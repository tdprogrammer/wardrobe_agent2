import os
from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# 1. Define the Tools (Skills)
# ==========================================

@tool
def Fetch_Calendar_Event(date: str) -> str:
    """Fetches calendar event details for a given date."""
    # Mocking an ambiguous event to force the agent to ask for clarification
    return '{"event_title": "Date with my gf", "location": "Downtown Coffee Shop", "time": "3:00 PM"}'

@tool
def Fetch_Weather(location: str, time: str) -> str:
    """Fetches weather for a given location and time."""
    return '{"temperature": "78F", "conditions": "Partly cloudy, slight breeze"}'

# For HITL, we define Request_Clarification as a Pydantic schema rather than an executable @tool.
# This ensures standard ToolNodes don't try to automatically run it.
class Request_Clarification(BaseModel):
    """Use this tool ONLY to ask the user a clarifying question when event type, dress code, or style is ambiguous or missing."""
    question_for_user: str = Field(description="The exact question you want to ask the user.")

# ==========================================
# 2. Setup the LangGraph State & Nodes
# ==========================================

# Initialize the Gemini Model (Requires GOOGLE_API_KEY in env)
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)


# Bind standard tools PLUS the schema for the HITL clarification tool
tools_list = [Fetch_Calendar_Event, Fetch_Weather]
llm_with_tools = llm.bind_tools(tools_list + [Request_Clarification])

SYSTEM_PROMPT = """You are a Wardrobe Recommendation Agent.
Your goal is to suggest an outfit based on weather, event type, and user style preferences.
Current Date: March 18, 2026. Location Context: Vietnam.

Strict Execution Rules:
1. Always start by using Fetch_Calendar_Event to get event details for the requested date.
2. Infer the event type/dress code from the event title. 
3. CRITICAL: If the event type is ambiguous (e.g., "Project Sync" could be casual or business formal), DO NOT GUESS. You must immediately call the Request_Clarification tool to ask the user what the vibe/dress code is.
4. Once you have a clear event type, location, and time, call Fetch_Weather.
5. Finally, provide a specific, contextual outfit recommendation.
"""

def agent_node(state: MessagesState):
    """The orchestrator node that calls the LLM."""
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
         messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
         
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def ask_human_placeholder(state: MessagesState):
    """
    Placeholder node. The graph is configured to interrupt BEFORE this node.
    The terminal loop will handle the logic and update the state as if this node executed.
    """
    pass

def router(state: MessagesState) -> Literal["tools", "ask_human", "__end__"]:
    """Routes based on the LLM's tool choices."""
    last_message = state["messages"][-1]
    
    # If the LLM didn't call any tools, we are done.
    if not last_message.tool_calls:
        return "__end__"
    
    # If the LLM wants to ask a question, route to the human breakpoint
    if any(tc["name"] == "Request_Clarification" for tc in last_message.tool_calls):
        return "ask_human"
    
    # Otherwise, execute standard tools
    return "tools"

# ==========================================
# 3. Build the Graph
# ==========================================

builder = StateGraph(MessagesState)

# Add Nodes
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools_list))
builder.add_node("ask_human", ask_human_placeholder)

# Add Edges
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", router)
builder.add_edge("tools", "agent")
builder.add_edge("ask_human", "agent") # After getting human input, go back to agent

# Compile with memory and a breakpoint on the ask_human node
memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["ask_human"])

# ==========================================
# 4. Terminal Execution Loop
# ==========================================

# ==========================================
# 4. Terminal Execution Loop
# ==========================================

def run_pilot():
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: Please set your GOOGLE_API_KEY environment variable.")
        return

    # Thread ID tracks the conversational state
    config = {"configurable": {"thread_id": "wardrobe_pilot_01"}}
    
    print("--- Wardrobe Agent Pilot Initialized ---")
    user_input = input("User: ")
    
    # FIX: Iterate through the stream to force the graph to execute
    for event in graph.stream({"messages": [HumanMessage(content=user_input)]}, config, stream_mode="values"):
        pass # We are just driving the execution forward until it pauses or finishes

    while True:
        # Check current state of the graph
        state = graph.get_state(config)
        
        # If there is no next node, the graph finished executing
        if not state.next:
            # Safely check if messages exist before trying to access them
            if "messages" in state.values:
                final_message = state.values["messages"][-1]
                print(f"\n[Agent Final Output]:\n{final_message.content}")
            else:
                print("\n[Error]: Graph finished but no messages were found in state.")
            break
            
        # If the graph paused because of our HITL breakpoint
        if state.next[0] == "ask_human":
            last_message = state.values["messages"][-1]
            
            # Extract the specific clarification question the LLM generated
            clarification_call = next(tc for tc in last_message.tool_calls if tc["name"] == "Request_Clarification")
            question = clarification_call["args"]["question_for_user"]
            
            print(f"\n[System Suspended] Agent requests clarification: '{question}'")
            human_response = input("Your Answer: ")
            
            # Construct a ToolMessage to feed the user's answer back into the LLM's context
            tool_message = ToolMessage(
                content=human_response,
                name="Request_Clarification",
                tool_call_id=clarification_call["id"]
            )
            
            # Update the state with the user's response, acting `as_node="ask_human"`
            graph.update_state(config, {"messages": [tool_message]}, as_node="ask_human")
            
            # FIX: Iterate through the stream again to resume execution
            for event in graph.stream(None, config, stream_mode="values"):
                pass

if __name__ == "__main__":
    run_pilot()