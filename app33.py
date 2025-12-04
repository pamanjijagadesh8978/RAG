import asyncio
import time
import os
import logging
from typing import Annotated, Sequence, TypedDict, Any, List, Dict
from operator import add as add_messages
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import pickle
import time
# LangChain / LangGraph Imports
# Added AIMessage import for cleaner error handling in call_llm
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_mistralai import ChatMistralAI
from langchain_core.tools import tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langgraph.graph import StateGraph, END
from langchain_community.callbacks import get_openai_callback

# ==================== CONFIGURATION ====================
# Initialize logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting application initialization.")

load_dotenv()

MISTRAL_ENDPOINT = "https://mistral-small-2503-Pamanji-test.southcentralus.models.ai.azure.com"
MISTRAL_API_KEY = "5SKKbylMh5ueyeSfvUre68vknfYZMVAr"

if not MISTRAL_API_KEY or not MISTRAL_ENDPOINT:
    # Use st.error instead of raising EnvironmentError to allow Streamlit UI to display the error
    logging.error("Missing MISTRAL_API_KEY or MISTRAL_ENDPOINT. Stopping.")
    st.error("Missing required environment variables: MISTRAL_API_KEY or MISTRAL_ENDPOINT.")
    st.stop()
logging.info("Environment variables loaded.")


# ==================== LOAD NUTRITION DATA ====================
# Make sure FNDDS.csv is present in the same folder
try:
    df = pd.read_csv("FNDDS.csv")
    logging.info(f"Loaded FNDDS.csv successfully with {len(df)} rows.")
except FileNotFoundError:
    logging.error("FNDDS.csv not found. Stopping.")
    st.error("FNDDS.csv not found — put your USDA nutrition CSV file next to this script.")
    st.stop()
except Exception as e:
    logging.error(f"Error loading FNDDS.csv: {e}")
    st.error(f"Error loading FNDDS.csv: {e}")
    st.stop()

# Normalize description column name (defensive)
try:
    if "Main food description" not in df.columns:
        # attempt common alternatives or error
        possible = [c for c in df.columns if "food" in c.lower() or "description" in c.lower()]
        if possible:
            df.rename(columns={possible[0]: "Main food description"}, inplace=True)
            logging.info(f"Renamed column '{possible[0]}' to 'Main food description'.")
        else:
            logging.error("Could not find a food description column. Stopping.")
            st.error("Could not find a food description column in FNDDS.csv. Expected 'Main food description'.")
            st.stop()
except Exception as e:
    logging.error(f"Error processing FNDDS.csv columns: {e}")
    st.error(f"Error processing FNDDS.csv columns: {e}")
    st.stop()

# ==================== CONSTANTS ====================
EMBED_FILE = "food_embeddings.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ==================== DATAFRAME LOADING (cached) ====================
@st.cache_data(show_spinner=True)
def load_food_df(path: str = "FNDDS.csv") -> pd.DataFrame:
    logging.info(f"Loading CSV from {path}")
    try:
        df_local = pd.read_csv(path)
    except FileNotFoundError:
        logging.error("FNDDS.csv not found. Stopping.")
        st.error("FNDDS.csv not found — put your USDA nutrition CSV file next to this script.")
        st.stop()
    except Exception as e:
        logging.error(f"Error loading FNDDS.csv: {e}")
        st.error(f"Error loading FNDDS.csv: {e}")
        st.stop()
    logging.info(f"Loaded DataFrame with {len(df_local)} rows and {len(df_local.columns)} columns.")
    return df_local

df = load_food_df()
cols = df.columns.tolist()

# ==================== EMBEDDING MODEL + EMBEDDINGS (cached) ====================
@st.cache_resource(show_spinner=False)
def load_sentence_model(name: str = MODEL_NAME) -> SentenceTransformer:
    logging.info(f"Loading SentenceTransformer model: {name} (CPU)")
    # SentenceTransformer accepts device parameter internally; set to CPU explicitly
    model = SentenceTransformer(name, device="cpu")
    logging.info("SentenceTransformer loaded.")
    return model

@st.cache_data(show_spinner=True)
def load_embeddings(embed_file: str = EMBED_FILE):
    logging.info(f"Loading embeddings from {embed_file}")
    with open(embed_file, "rb") as f:
        saved = pickle.load(f)

    sentences = saved["sentences"]
    embeddings_raw = saved["embeddings"]  # likely a numpy array or list

    # Convert to torch tensor (CPU) and normalize
    emb_tensor = torch.tensor(embeddings_raw, dtype=torch.float32)
    emb_tensor = torch.nn.functional.normalize(emb_tensor, p=2, dim=1)
    logging.info(f"Embeddings loaded and normalized: shape={tuple(emb_tensor.shape)}")
    return sentences, emb_tensor

model = load_sentence_model()
sentences, corpus_embeddings = load_embeddings()

# ==================== FAST SEMANTIC SEARCH (CPU, no FAISS) ====================
# We'll cache recent queries using st.cache_data (simple caching)
@st.cache_data(show_spinner=False)
def semantic_search_cached(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    return semantic_search_impl(query, top_k=top_k)

def semantic_search_impl(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Compute a query embedding (normalized) and perform fast dot-product with pre-normalized corpus.
    Returns top_k matches as a list of dicts: {row_number, match, score}
    """
    logging.info(f"semantic_search: embedding and searching for query='{query}' (top_k={top_k})")
    # encode with convert_to_tensor and normalize via parameter if available
    try:
        q_emb = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    except TypeError:
        # older versions: compute and normalize manually
        q_emb = model.encode(query, convert_to_tensor=True)
        q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=0)

    # Ensure q_emb is CPU tensor float32
    if isinstance(q_emb, torch.Tensor):
        q_emb = q_emb.cpu().float()
    else:
        q_emb = torch.tensor(q_emb, dtype=torch.float32)

    # Dot product between corpus_embeddings (N x D) and q_emb (D,)
    scores = torch.mv(corpus_embeddings, q_emb)  # faster than matmul for vector multiply
    top_k = min(top_k, scores.size(0))
    top_values, top_indices = torch.topk(scores, k=top_k)

    results: List[Dict[str, Any]] = []
    for val, idx in zip(top_values.tolist(), top_indices.tolist()):
        results.append({"row_number": int(idx), "match": sentences[idx], "score": float(val)})
    logging.info(f"semantic_search: found {len(results)} results.")
    return results

def semantic_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    try:
        return semantic_search_cached(query, top_k=top_k)
    except Exception as e:
        logging.warning(f"semantic_search cache failed: {e}. Falling back to direct call.")
        return semantic_search_impl(query, top_k=top_k)

# ==================== SEARCH WRAPPER ====================
@st.cache_resource(show_spinner=False)
def get_ddg_wrapper():
    logging.info("Initializing DuckDuckGoSearchAPIWrapper (cached).")
    return DuckDuckGoSearchAPIWrapper(max_results=15)

wrapper = get_ddg_wrapper()

# ==================== AGENT STATE ====================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    token_usage: dict | None

# ==================== SYSTEM PROMPT ====================
SYSTEM_INSTRUCTION = """
You are a tool-dependent nutrition assistant. Do NOT answer from your own knowledge—always use a tool.

1. For every query, call either `get_matching_food` (specific food nutrition) or `search_web` (general nutrition/health topics).

2. Use `get_matching_food` when user asks nutrition of a specific food.
   - Input: "food name, column1, column2, ..." (e.g., energy, protein, fat, sugar—USDA names).
   - If portion differs, scale values proportionally.
   - If only food name is given, returns full USDA nutrition profile with 68 columns.
   - Always try not get all the 68 columns, try to give input for required columns only.
   - Pick only relevant columns based on user intent.
   - If results are poor, call `search_web`.
   - Always state: "Data from USDA FNDDS dataset."

3. Use `search_web` for general health/nutrition questions.
   - Query reliable sources (USDA, NIH ODS, EFSA, FAO/WHO INFOODS, CDC, Harvard, NCCDB, CoFID, CNF, AFCD, PubMed, etc.).
   - One source per query. Cite source name/domain (no URLs).
   - Construct the query with relavent single source name
   - And acknowledge the results

4. Responses must be **plain text**.

5. If no results, say so politely.

6. Only answer nutrition/health-related questions; refuse others.
"""
# ==================== TOOLS ====================
@tool
async def search_web(query: str) -> str:
    """Run DuckDuckGo search asynchronously and return text results."""
    logging.info(f"Tool: search_web called with query: '{query}'")
    try:
        loop = asyncio.get_event_loop()
        # run the blocking wrapper.run in an executor
        results = await loop.run_in_executor(None, wrapper.run, query)
        if not results:
            logging.warning("Tool: search_web returned no results.")
            return "No search results available."
        # wrapper.run typically returns a string or structured text; normalize to string
        logging.info("Tool: search_web succeeded.")
        return str(results)
    except Exception as exc:
        logging.error(f"Tool: search_web failed: {exc}")
        return f"Search failed. No information available. ({exc})"

@tool
async def get_matching_food(keyword: str) -> Any:
    """
    Return rows of nutritional information that match a given food name.
    """
    logging.info(f"Tool: get_matching_food called with keyword: '{keyword}'")

    # Split "food, col1, col2"
    parts = [p.strip() for p in keyword.split(",")]
    keyword = parts[0]
    col_names = parts[1:]  # optional user columns
    matched_cols = []

    try:
        # Normalize keyword
        if not isinstance(keyword, str):
            keyword = str(keyword)
        key = keyword.strip().lower()

        if not key:
            logging.warning("Tool: get_matching_food received an empty keyword.")
            return "No keyword provided."

        # Perform semantic search
        results = semantic_search(keyword, top_k=5)
        row_numbers = [r["row_number"] for r in results]

        # If only the food name was provided → return full rows
        if not col_names:
            matched = df.loc[row_numbers]
        else:
            # Add fixed essential columns
            fixed_cols = ['Main food description', 'Portion weight (g)', 'Portion description']
            col_names = list(set(fixed_cols + col_names))
            print(f"\nRequested columns: {col_names}")

            # Validate column names exist
            for name in col_names:
                for col in cols:
                    if name.lower() in col.lower():
                        matched_cols.append(col)
                        break   # stop after first match
            print(f"\nRequested columns: {matched_cols}")

            if not matched_cols:
                return f"No valid columns found among: {matched_cols}"

            matched = df.loc[row_numbers, matched_cols]

        print("\nMatched DataFrame rows:")
        print(matched)

        print("\nTop matches:")
        for r in results:
            print(f"[Row {r['row_number']}] {r['score']:.4f} → {r['match']}")

        if matched.empty:
            logging.info("Tool: get_matching_food found no matching rows.")
            return "No matching food found."

        logging.info(f"Tool: get_matching_food returning {len(matched)} records.")
        return matched.to_dict(orient="records")

    except Exception as exc:
        logging.error(f"Tool: get_matching_food failed: {exc}")
        return f"Food lookup failed: {exc}"        

# ==================== LLM INIT ====================
try:
    llm = ChatMistralAI(
        endpoint=MISTRAL_ENDPOINT,
        mistral_api_key=MISTRAL_API_KEY,
        temperature=0.2,
    )
    logging.info("ChatMistralAI initialized.")
except Exception as e:
    logging.error(f"Failed to initialize ChatMistralAI: {e}. Stopping.")
    st.error(f"Failed to initialize ChatMistralAI: {e}")
    st.stop()

# bind tools to the LLM parent
parent = llm.bind_tools([search_web, get_matching_food])

# ==================== WORKFLOW NODES ====================
# Make call_llm async so we can use await parent.ainvoke(...)
async def call_llm(state: AgentState) -> AgentState:
    """
    Send messages to the LLM and return a new state containing the LLM response
    and token usage (if available).
    """
    messages = [SystemMessage(content=SYSTEM_INSTRUCTION)] + list(state["messages"])
    logging.info(f"LLM Node: Invoking LLM with {len(messages)} messages (including System Instruction).")

    # use parent.ainvoke (async)
    try:
        with get_openai_callback() as cb:
            response = parent.invoke(messages)
            logging.info("LLM Node: Received response from LLM.")
    except Exception as exc:
        logging.error(f"LLM Node: LLM call failed with exception: {exc}")
        # return a friendly error message embedded as assistant text
        err_msg = f"LLM call failed: {exc}"
        # FIX: Changed HumanMessage to AIMessage. LLM error response should be an AIMessage.
        return {"messages": [AIMessage(content=err_msg)], "token_usage": None}
    
    if getattr(response, "tool_calls", None):
        logging.info(f"LLM Node: LLM decided to call {len(response.tool_calls)} tool(s).")
    else:
        logging.info("LLM Node: LLM produced a final answer.")

    # Return the model response as a message
    if isinstance(response, str):
        # This case is highly unlikely for LangChain LLM output, but retained for robustness.
        # Should ideally be wrapped as an AIMessage if it came from the LLM.
        msg = AIMessage(content=response)
    else:
        # If the response is a BaseMessage (expected), use it directly
        msg = response
    
    return {
        "messages": [response],
        "token_usage": {
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "total_tokens": cb.total_tokens,
        },
    }

# tool execution node — already async
async def take_action(state: AgentState) -> AgentState:
    """
    Executes the tools requested by the last LLM message's `tool_calls` attribute.
    Correctly extracts the single string argument from the kwargs dictionary.
    """
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []

    if not tool_calls:
        # nothing to do
        return {"messages": [], "token_usage": None}

    logging.info(f"Tool Executor Node: Executing {len(tool_calls)} tool call(s).")

    # map of tool name -> function
    tool_map = {
        "search_web": search_web,
        "get_matching_food": get_matching_food,
    }

    tasks = []
    # build tasks using the correct tool function
    for t in tool_calls:
        name = t.get("name")
        args = t.get("args", {})  # Expect a dictionary of kwargs, default to empty dict
        tool_fn = tool_map.get(name)

        if tool_fn is None:
            logging.error(f"Tool Executor Node: Unknown tool '{name}'. Skipping.")
            # unknown tool — return a ToolMessage saying so
            tasks.append(asyncio.sleep(0, result=f"Unknown tool: {name}"))
            continue

        arg_to_pass = "" 

        # CRITICAL FIX: LangChain models return arguments as a dictionary (kwargs).
        # Since the tool functions expect a single positional string argument, we must extract the value.
        if isinstance(args, dict):
            if name == "search_web":
                # Expects 'query' key
                arg_to_pass = args.get('query', '')
            elif name == "get_matching_food":
                # Expects 'keyword' key
                arg_to_pass = args.get('keyword', '')
        # Fallback for old/non-standard LLM tool call formats (e.g., raw string/list if not a dict)
        elif isinstance(args, (list, tuple)):
             # Join list/tuple items into a single string
             arg_to_pass = " ".join(map(str, args))
        else:
             # Assume raw string argument
             arg_to_pass = str(args)
             
        logging.info(f"Tool Executor Node: Preparing to call tool '{name}' with argument: '{arg_to_pass}'")

        # Call the tool function with the extracted positional argument value
        tasks.append(tool_fn.ainvoke(arg_to_pass))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    logging.info("Tool Executor Node: Tool execution complete. Processing results.")

    tool_messages = []
    for t, result in zip(tool_calls, results):
        # Convert exceptions to strings
        if isinstance(result, Exception):
            content = f"Tool '{t.get('name')}' failed: {result}"
            logging.error(f"Tool Executor Node: Tool '{t.get('name')}' failed: {result}")
        else:
            content = result
            logging.info(f"Tool Executor Node: Tool '{t.get('name')}' succeeded. Result size: {len(str(result))} chars.")
        # Create a ToolMessage that references tool name & id
        tool_messages.append(
            ToolMessage(tool_call_id=t.get("id", "unknown"), name=t.get("name", "unknown"), content=str(content))
        )

    return {"messages": tool_messages, "token_usage": None}

# ==================== GRAPH ====================
# Use StateGraph similarly to your original but we're compiling a graph that uses async functions
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tool_executor", take_action)
# conditional function same as before
def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    
    # Log the decision process
    if getattr(last, "tool_calls", None):
        logging.info("Graph Decision: Tool calls detected. Routing to 'tool_executor'.")
        return "tool_executor"
    
    logging.info("Graph Decision: No tool calls. Routing to 'END'.")
    return END

graph.add_conditional_edges("llm", should_continue, {"tool_executor": "tool_executor", END: END})
graph.add_edge("tool_executor", "llm")
graph.set_entry_point("llm")
assistant_agent = graph.compile()
logging.info("LangGraph agent compiled and ready.")


# ==================== EXECUTION WRAPPER ====================
async def run_agent(query: str) -> tuple[str, dict]:
    logging.info(f"Agent Execution started for query: '{query}'")
    initial = {"messages": [HumanMessage(content=query)]}

    usage = None
    final_content = "Unexpected error: no final output produced."

    async for step in assistant_agent.astream(initial):
        node, state = list(step.items())[0]

        # Update usage if present
        current_usage = state.get("token_usage", None)
        if current_usage:
            # We track the last known usage for the final message
            usage = current_usage

        logging.info(f"Agent Execution: Completed node '{node}'. Next decision...")

        # If we reached END node or the llm node decided to end, extract content
        if node == END or (node == "llm" and should_continue(state) == END):
            final_msg = state["messages"][-1]
            # Ensure content is extracted robustly
            final_content = getattr(final_msg, "content", str(final_msg))
            logging.info("Agent Execution: Reached END. Final content extracted.")
            break # Exit loop once final message is found

    logging.info(f"Agent Execution finished. Final usage: {usage}")
    return final_content, usage

# ==================== STREAMLIT UI ====================
st.set_page_config(page_title="Nutrition Assistant", page_icon="🥗")

st.markdown(
    """
<div style="padding:20px;border-radius:15px;background:#F0FFF0;border:1px solid #DFF0D8;text-align:center;">
    <h1 style="color:#2E8B57;margin-bottom:0;">🥦 Nutrition Expert</h1>
    <p style="margin-top:4px;color:#3A5F0B;">Ask any nutrition or health-related question.</p>
</div>
""",
    unsafe_allow_html=True,
)

# Init session variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
    st.session_state.total_prompt = 0
    st.session_state.total_completion = 0

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
query = st.chat_input("Ask something related to Nutritional information...")
start_time = time.time()

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    response = ""
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching and analyzing..."):
            # Console feedback that the agent run is starting
            print(f"\n--- Starting Agent Run for User Query: {query} ---")
            
            # Run the agent asynchronously using a fresh loop 
            loop = asyncio.new_event_loop()
            try:
                # Set event loop for run_agent
                asyncio.set_event_loop(loop)
                response, usage_data = loop.run_until_complete(run_agent(query))
                if usage_data:
                    usage = usage_data # Assign the usage data if available
            except Exception as e:
                response = f"An error occurred during agent execution: {e}"
                print(f"--- Agent Run Error: {e} ---")
            finally:
                # Always ensure the loop is closed to prevent resource leakage
                try:
                    loop.close()
                except Exception:
                    pass
            print("--- Agent Run Finished ---")

        end_time = time.time()
        total_time = end_time - start_time

        st.write(response)
        st.info(f"⏱️ Time taken: {total_time:.3f} seconds")

    # Save response and token usage into session (if available)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.session_state.total_prompt += usage.get("prompt_tokens", 0)
    st.session_state.total_completion += usage.get("completion_tokens", 0)
    st.session_state.total_tokens += usage.get("total_tokens", 0)

    # ====================== TOKEN METER ======================
    st.sidebar.title("📊 Token Usage")
    st.sidebar.info(
        f"""
### 🔹 This response
- **Prompt tokens:** {usage.get("prompt_tokens", 0)}
- **Completion tokens:** {usage.get("completion_tokens", 0)}
- **Total tokens:** {usage.get("total_tokens", 0)}

---

### 🔸 Session totals
- **Prompt tokens:** {st.session_state.total_prompt}
- **Completion tokens:** {st.session_state.total_completion}
- **Total tokens:** {st.session_state.total_tokens}
"""
    )

# ==================== END OF FILE ====================

