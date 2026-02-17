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

# --- Custom CSS for MARCA Style ---
st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
    }
    h1 {
        color: #CC0000 !important;
        font-family: 'Arial Black', sans-serif;
        text-transform: uppercase;
        border-bottom: 4px solid #CC0000;
        padding-bottom: 10px;
    }
    h2, h3 {
        color: #000000 !important;
        font-family: 'Arial', sans-serif;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #CC0000 !important;
        color: white !important;
        border-radius: 0px !important;
        font-weight: bold !important;
        font-size: 18px !important;
        text-transform: uppercase;
        border: none !important;
    }
    .stMarkdown p {
        font-family: 'Georgia', serif;
        font-size: 18px;
        line-height: 1.6;
        color: #333333;
    }
    div[data-testid="stSidebar"] {
        background-color: #f0f0f0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚽ FUT CHRONICLE GENERATOR")
st.markdown("**La pasión del fútbol, generada por IA.**")

# Sidebar for API Keys
with st.sidebar:
    st.header("CONFIGURACIÓN")
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

writer_template = """Tu trabajo es escribir una crónica de fútbol con el ESTILO INCONFUNDIBLE DE "MARCA".
                        
                        **ESTILO Y TONO:**
                        - **SENSACIONALISTA Y APASIONADO:** Usa mayúsculas para enfatizar, exclamaciones y un lenguaje muy vivo ("PARTIDAZO", "RECITAL", "ESCÁNDALO", "ROBO", "HEROIDIDAD").
                        - **PERIODÍSTICO PERO "FOROFO":** Debes sonar como un redactor deportivo emocionado. No seas neutral.
                        - **FRASES CORTAS Y DIRECTAS:** Párrafos breves, mucho ritmo.
                        
                        **ESTRUCTURA OBLIGATORIA (FORMATO MARCA):**
                        
                        1. **TITULAR FLASH:** Una frase corta, impactante, en MAYÚSCULAS. (Ej: "¡EL MADRID NUNCA MUERE!")
                        2. **SUBTÍTULO:** Un resumen de una línea con un dato clave o la figura del partido.
                        3. **LA FOTO:** (Describe brevemente una imagen mental del momento clave, ej: *Vinicius besando el escudo...*)
                        4. **CRÓNICA (El cuerpo):**
                           - **Introducción:** ¡Directo a la yugular! Quién ganó y por qué fue épico.
                           - **El Crack:** Destaca al mejor jugador con adjetivos grandilocuentes.
                           - **El Dandy:** El jugador con más clase.
                           - **El Duro:** La acción más polémica o el jugador más agresivo.
                           - **Desarrollo:** Cuenta los goles y momentos clave con mucha emoción.
                        5. **FICHA TÉCNICA:**
                           - Goles: (Minuto y autor).
                           - Estadio: (Nombre y asistencia).
                        
                        **IMPORTANTE:**
                        - EL RESULTADO DEBE ESTAR SIEMPRE EN ESPAÑOL.
                        - SI HAY POLÉMICA, MÓJATE (Opina si fue penalti o no).
                        - NO SALUDES NI DIGAS "AQUÍ TIENES LA CRÓNICA". PÓNLA DIRECTAMENTE.
                        
                        **INSTRUCCIONES DE REVISIÓN:**
                        Si recibes feedback de un REVISOR, tu trabajo es REESCRIBIR la crónica incorporando TODAS las sugerencias.
                        Mantén el mismo formato de salida.
                    """

reviewer_template = """Eres el JEFE DE REDACCIÓN de MARCA. Tu trabajo es asegurar que la crónica tenga "GARRA" y venda periódicos.
                       
                       Criterios de revisión (ESTILO MARCA):
                       1. **¿ES ABURRIDO?:** Si suena a Wikipedia o a resumen formal, RECHÁZALO. Tiene que emocionar.
                       2. **TITULARES:** ¿Son impactantes? ¿Usan mayúsculas y exclamaciones?
                       3. **LENGUAJE:** Busca palabras como "Gesto", "Hecatombe", "Fiesta", "Rodillo". Si no las hay, pide más intensidad.
                       4. **ESTRUCTURA:** ¿Tiene las secciones de "El Crack", "El Dandy", "El Duro"? Son obligatorias.
                       
                       Instrucciones:
                       - Lee el último mensaje del Writer Agent.
                       - Si la crónica es un ESPECTÁCULO digno de portada, responde ÚNICAMENTE con la palabra: **ACEPTADO**.
                       - Si le falta "sangre", dile al redactor específicamente qué cambiar en una lista numerada.
                       - **IMPORTANTE:** TU NO ESCRIBES LA CRÓNICA. SOLO CRITICAS.
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
# --- Execution ---
if st.button("Generate Chronicle"):
    if not match_topic:
        st.warning("Please enter a match topic.")
    else:
        with st.spinner("Analyzing the match... (This might take a minute)"):
            try:
                initial_state = {"messages": [HumanMessage(content=match_topic)]}
                status_text = st.empty()
                status_text.text("Starting graph execution...") 
                
                all_messages = []
                if 'messages' in initial_state:
                    all_messages.extend(initial_state['messages'])

                # Iterate through the stream to show progress but capture the state
                for event in graph.stream(initial_state):
                    for key, value in event.items():
                        status_text.text(f"Processing step: {key}...")
                        # Accumulate new messages
                        if 'messages' in value:
                             all_messages.extend(value['messages'])
                
                status_text.empty() # Clear status
                
                # Find the last valid message from the Writer
                # We iterate backwards. We look for a message that:
                # 1. Is not "ACEPTADO"
                # 2. Has substantial content.
                
                final_chronicle = None
                
                if all_messages:
                    for msg in reversed(all_messages):
                        content = msg.content
                        if content and "ACEPTADO" not in content and len(content) > 50: # Assuming a chronicle is longer than 50 chars
                            final_chronicle = content
                            break
                
                if final_chronicle:
                     st.markdown(final_chronicle)
                else:
                     st.warning("No generated chronicle found. The agent might have failed or the content was filtered.")
                     # Fallback debugging
                     with st.expander("Debug History"):
                         for m in all_messages:
                             st.write(f"**{m.type}:** {m.content}")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                import traceback
                st.text(traceback.format_exc())
