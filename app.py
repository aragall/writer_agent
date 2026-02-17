import streamlit as st
import os
import functools
from typing import Annotated, Literal, TypedDict
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Football Chronicle Generator", page_icon="⚽")

st.title("⚽ Football Chronicle Generator")
st.markdown("Generate detailed football match chronicles using AI agents.")

# Sidebar for API Keys
with st.sidebar:
    st.header("Configuration")
    google_api_key = st.text_input("Google API Key", type="password", help="Get your key from Google AI Studio")
    tavily_api_key = st.text_input("Tavily API Key", type="password", help="Get your key from Tavily")
    
    if not google_api_key:
        st.warning("Please enter your Google API Key to proceed.")
        st.stop()
        
    if not tavily_api_key:
        st.warning("Please enter your Tavily API Key to proceed.")
        st.stop()

# Set environment variables
os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key

# Match Input
match_topic = st.text_input("Enter the match details (e.g., 'Real Madrid vs Barcelona 2024 final score and summary')", placeholder="Real Madrid vs Barcelona 2024")

# --- LangGraph Setup ---

# 1. Define State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 2. Define Tools
tools = [TavilySearchResults(max_results=5)]
tool_node = ToolNode(tools)

# 3. Define Agents
def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    if tools:
        return prompt | llm.bind_tools(tools)
    else:
        return prompt | llm

# Agent Prompts
search_template = """Your job is to search the web for detailed football match information relevant to the user's request.
                  Focus on finding:
                  - Final score and scorers
                  - Key moments (goals, cards, VAR decisions)
                  - Team lineups and tactical setups
                  - Post-match reactions and stats
                  
                  NOTE: Do not write the chronicle. Just search the web for related facts and then forward that information to the outliner node.
                  """

outliner_template = """Your job is to take as input a list of search results about a football match and generate a structured outline for a match chronicle.
                       The outline should include:
                       - Introduction (Match context, venue, importance)
                       - First Half Summary
                       - Second Half Summary
                       - Key Players and Performances
                       - Tactical Analysis (brief)
                       - Conclusion (Final result, implications)
                    """

writer_template = """Your job is to write a compelling football match chronicle based on the provided outline.
                        Write in a journalistic sports style—engaging, dramatic, and factual.
                        
                        Format the output as follows:
                        
                        ## TITLE: <Catchy Title>
                        
                        **Date:** <Match Date>
                        **Venue:** <Stadium>
                        
                        <Body of the chronicle>
                        
                      NOTE: Do not copy the outline verbatim. Flesh it out into a full narrative using the details provided.
                       ```
                    """

# Initialize LLM
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# Create Agents
search_agent = create_agent(llm, tools, search_template)
outliner_agent = create_agent(llm, [], outliner_template)
writer_agent = create_agent(llm, [], writer_template)

# 4. Define Nodes
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {
        'messages': [result]
    }

search_node = functools.partial(agent_node, agent=search_agent, name="Search Agent")
outliner_node = functools.partial(agent_node, agent=outliner_agent, name="Outliner Agent")
writer_node = functools.partial(agent_node, agent=writer_agent, name="Writer Agent")

# 5. Define Edge Logic
def should_search(state) -> Literal["tools", "outliner"]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (send state to outliner)
    return "outliner"

# 6. Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("search", search_node)
workflow.add_node("tools", tool_node)
workflow.add_node("outliner", outliner_node)
workflow.add_node("writer", writer_node)

workflow.set_entry_point("search")

workflow.add_conditional_edges(
    "search",
    should_search
)
workflow.add_edge("tools", "search")
workflow.add_edge("outliner", "writer")
workflow.add_edge("writer", END)

graph = workflow.compile()

# --- Execution ---
if st.button("Generate Chronicle"):
    if not match_topic:
        st.warning("Please enter a match topic.")
    else:
        with st.spinner("Analyzing the match... (This might take a minute)"):
            try:
                initial_state = {"messages": [HumanMessage(content=match_topic)]}
                final_state = graph.invoke(initial_state)
                
                # Extract the final response from the writer agent
                final_message = final_state['messages'][-1].content
                
                st.markdown("### Match Chronicle")
                st.markdown(final_message)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

