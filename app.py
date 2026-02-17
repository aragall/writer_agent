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
search_template = """Tu trabajo es buscar en la web información detallada sobre partidos de fútbol relevante para la solicitud del usuario.
                  Si el usuario pide un partido específico, busca:
                  - Resultado final y goleadores.
                  - Momentos clave (goles, tarjetas, decisiones del VAR).
                  - Alineaciones y esquemas tácticos.
                  - Reacciones post-partido y estadísticas.
                  
                  Si el usuario hace una pregunta general o sobre otro tema relacionado con fútbol:
                  - Busca las noticias, estadísticas o datos históricos más relevantes y recientes que respondan a su duda.
                  
                  NOTA: Una vez hayas obtenido la información de las búsquedas, HAZ UN RESUMEN de los datos encontrados como respuesta final. NO escribas la crónica todavía, pero aporta los datos crudos y resumidos para que el siguiente agente (outliner) pueda trabajar.
                  """

outliner_template = """Tu trabajo es tomar como entrada una lista de resultados de búsqueda sobre un partido o tema de fútbol y generar un esquema estructurado para una crónica o artículo.
                       
                       Si es una crónica de partido, el esquema debe incluir:
                       - Introducción (Contexto del partido, estadio, importancia).
                       - Resumen de la Primera Parte.
                       - Resumen de la Segunda Parte.
                       - Jugadores Clave y Rendimiento.
                       - Análisis Táctico (breve).
                       - Conclusión (Resultado final, implicaciones).
                       
                       Si es otro tipo de artículo (ej. noticias, historia), estructura el esquema de manera lógica con:
                       - Introducción.
                       - Puntos clave / Desarrollo del tema.
                       - Conclusión / Resumen.
                    """

writer_template = """Tu trabajo es escribir una crónica o artículo de fútbol convincente basado en el esquema proporcionado.
                        Escribe en un estilo periodístico deportivo: atractivo, dramático y factual.
                        
                        **IMPORTANTE: EL RESULTADO DEBE ESTAR SIEMPRE EN ESPAÑOL.**
                        
                        Formatea la salida de la siguiente manera:
                        
                        ## TÍTULO: <Título Atractivo>
                        
                        **Fecha:** <Fecha del evento/hoy>
                        **Contexto:** <Estadio/Competición/Tema>
                        
                        <Cuerpo de la crónica/artículo>
                        
                      NOTA: No copies el esquema palabra por palabra. Desarrolla una narrativa completa usando los detalles proporcionados.
                       ```
                        
                        **INSTRUCCIONES DE REVISIÓN:**
                        Si recibes feedback de un REVISOR, tu trabajo es REESCRIBIR la crónica incorporando TODAS las sugerencias.
                        Mantén el mismo formato de salida.
                    """

reviewer_template = """Eres un editor senior de una prestigiosa revista deportiva. Tu trabajo es revisar la crónica escrita por el "Writer Agent".
                       
                       Criterios de revisión:
                       1. **Estilo:** Debe ser periodístico, emocionante y dramático. Evita el lenguaje robótico.
                       2. **Contenido:** Debe ser fiel a los datos proporcionados (no inventar hechos).
                       3. **Formato:** Debe seguir la estructura solicitada (Título, Fecha, Contexto, Cuerpo).
                       4. **Idioma:** Debe estar en un Español perfecto y natural.
                       
                       Instrucciones:
                       - Lee el último mensaje del Writer Agent.
                       - Si la crónica es excelente y cumple todos los criterios, responde ÚNICAMENTE con la palabra: **ACEPTADO**.
                       - Si hay aspectos a mejorar, proporciona una lista numerada de críticas constructivas y específicas para que el escritor lo corrija.
                       """

# Initialize LLM
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# Create Agents
search_agent = create_agent(llm, tools, search_template)
outliner_agent = create_agent(llm, [], outliner_template)
writer_agent = create_agent(llm, [], writer_template)
reviewer_agent = create_agent(llm, [], reviewer_template)

# 4. Define Nodes
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {
        'messages': [result]
    }

search_node = functools.partial(agent_node, agent=search_agent, name="Search Agent")
outliner_node = functools.partial(agent_node, agent=outliner_agent, name="Outliner Agent")
writer_node = functools.partial(agent_node, agent=writer_agent, name="Writer Agent")
reviewer_node = functools.partial(agent_node, agent=reviewer_agent, name="Reviewer Agent")

# 5. Define Edge Logic
def should_search(state) -> Literal["tools", "outliner"]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (send state to outliner)
    return "outliner"

def should_continue(state) -> Literal["writer", END]:
    messages = state['messages']
    last_message = messages[-1]
    
    # If the reviewer accepts the text, we end
    if "ACEPTADO" in last_message.content:
        return END
    
    # Safety mechanism: Limit the number of revisions to check specifically for Reviewer messages
    # Count how many times the Reviewer has spoken (approximate by content or role analysis if available, 
    # but here we can just count total messages if simpler, or check for specific agent flow).
    # Simple safeguard: if there are too many messages, stop.
    if len(messages) > 15: # Assuming ~3-4 messages per turn (search, outline, write, review) -> ~3 loops
        return END
        
    return "writer"

# 6. Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("search", search_node)
workflow.add_node("tools", tool_node)
workflow.add_node("outliner", outliner_node)
workflow.add_node("writer", writer_node)
workflow.add_node("reviewer", reviewer_node)

workflow.set_entry_point("search")

workflow.add_conditional_edges(
    "search",
    should_search
)
workflow.add_edge("tools", "search")
workflow.add_edge("outliner", "writer")
workflow.add_edge("writer", "reviewer")
workflow.add_conditional_edges(
    "reviewer",
    should_continue
)

graph = workflow.compile()

# --- Execution ---
if st.button("Generate Chronicle"):
    if not match_topic:
        st.warning("Please enter a match topic.")
    else:
        with st.spinner("Analyzing the match... (This might take a minute)"):
            try:
                initial_state = {"messages": [HumanMessage(content=match_topic)]}
                st.write("Starting graph execution...") 
                
                for event in graph.stream(initial_state):
                    for key, value in event.items():
                        st.write(f"Node finished: {key}") 
                        if 'messages' in value:
                             msg = value['messages'][-1]
                             st.write(f"Message content snippet: {msg.content[:100]}...")
                
                final_state = initial_state # Initialize fallback

                
                # Extract the final response from the writer agent
                if final_state and 'messages' in final_state:
                    messages = final_state['messages']
                    if messages:
                        last_msg = messages[-1]
                        # st.write("--- Debug: Last Message Info ---")
                        # st.write(f"Type: {type(last_msg)}")
                        # st.write(f"Content: '{last_msg.content}'") 
                        # st.write(f"Additional kwargs: {last_msg.additional_kwargs}")
                        # st.write("-------------------------------")
                        
                        final_message = last_msg.content
                        
                        if not final_message:
                            st.warning("The agent returned an empty response. Let's check the previous steps.")
                            st.json([m.content for m in messages]) # Show history to debug
                        else:
                            st.markdown("### Match Chronicle")
                            st.markdown(final_message)
                    else:
                         st.error("No messages returned.")
                else:
                    st.error("Invalid final state.")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                import traceback
                st.text(traceback.format_exc())

