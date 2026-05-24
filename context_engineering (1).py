"""
Context Engineering with Pixeltable
====================================
A sophisticated AI system combining RAG, tool calling, short-term memory,
long-term vector memory, context summarization, and response generation.

Requirements:
    uv add pixeltable pixelagent openai lancedb sentence-transformers tiktoken

Usage:
    Set your OpenAI API key, then run:
        python context_engineering.py
"""

import os
import getpass
from pathlib import Path

import numpy as np
import lancedb
import tiktoken
from sentence_transformers import SentenceTransformer

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.iterators import DocumentSplitter
from pixeltable.functions.huggingface import sentence_transformer
from pixelagent.openai import Agent


# ---------------------------------------------------------------------------
# 0. API key setup  (called inside main, not at import time)
# ---------------------------------------------------------------------------

def setup_api_keys() -> None:
    """Prompt for missing API keys so the script can be safely imported."""
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key: ")


# ---------------------------------------------------------------------------
# 1. RAG — document ingestion & embedding index
# ---------------------------------------------------------------------------

def setup_rag() -> tuple:
    """
    Create the Pixeltable directory, documents table, and chunked view.
    Returns (documents_t, documents_chunks, embed_model).
    """
    pxt.drop_dir("context_engineering", if_not_exists="ignore", force=True)
    pxt.create_dir("context_engineering")

    documents_t = pxt.create_table(
        "context_engineering.documents",
        {"pdf": pxt.Document},
    )

    documents_chunks = pxt.create_view(
        "context_engineering.document_chunks",
        documents_t,
        iterator=DocumentSplitter.create(
            document=documents_t.pdf,
            separators="token_limit",
            limit=300,
        ),
    )

    embed_model = sentence_transformer.using(model_id="intfloat/e5-large-v2")
    documents_chunks.add_embedding_index(column="text", string_embed=embed_model)

    return documents_t, documents_chunks, embed_model


def ingest_documents(documents_t):
    """Load financial PDFs from GitHub into the documents table."""
    base_url = (
        "https://github.com/pixeltable/pixeltable/raw/release/docs/resources/rag-demo/"
    )
    document_urls = [
        base_url + doc
        for doc in [
            "Argus-Market-Digest-June-2024.pdf",
            "Company-Research-Alphabet.pdf",
        ]
    ]
    documents_t.insert({"pdf": url} for url in document_urls)
    print(f"Ingested {len(document_urls)} documents.")


# ---------------------------------------------------------------------------
# 2. RAG tool — vector similarity search over document chunks
# ---------------------------------------------------------------------------

def make_find_documents_tool(documents_chunks):
    """
    Returns a Pixeltable query UDF that retrieves the top-5 most relevant
    document chunks for a given query string.
    """

    @pxt.query
    def find_documents(query: str) -> dict:
        """Return top-5 chunks from financial documents ranked by vector similarity."""
        sim = documents_chunks.text.similarity(query)
        return (
            documents_chunks.order_by(sim, asc=False)
            .select(documents_chunks.text, similarity=sim)
            .limit(5)
        )

    return find_documents


# ---------------------------------------------------------------------------
# 3. MCP tool — real-time exchange rates via Alpha Vantage
# ---------------------------------------------------------------------------

def load_exchange_rate_tool(api_key: str):
    """
    Connect to the Alpha Vantage MCP server and return the exchange-rate UDF.
    Replace <api-key> with a real key from https://www.alphavantage.co/support/#api-key
    """
    udfs = pxt.mcp_udfs(f"https://mcp.alphavantage.co/mcp?apikey={api_key}")
    # Index 30 is the CURRENCY_EXCHANGE_RATE tool; run print(list(enumerate(udfs))) to verify
    get_exchange_rate = udfs[30]
    return get_exchange_rate


# ---------------------------------------------------------------------------
# 4. Tools agent — orchestrates RAG + exchange-rate tool calls
# ---------------------------------------------------------------------------

def create_tools_agent(find_documents, get_exchange_rate) -> Agent:
    """
    Build a Pixelagent that decides when to call find_documents (RAG)
    vs get_exchange_rate (live market data) based on the user query.
    """
    tools_agent = Agent(
        name="context_engineering.agent",
        system_prompt=(
            "You are a tools-enabled AI assistant. "
            "Use `find_documents` for RAG/PDF-Search to retrieve the top-5 chunks "
            "as per user query and answer ONLY from those. "
            "If context is insufficient, say so explicitly. "
            "Use `get_exchange_rate` for real-time exchange rate for any pair of "
            "cryptocurrency (e.g., Bitcoin) or physical currency (e.g., USD). "
            "Be concise, factual, and avoid recursive tool calls."
        ),
        tools=pxt.tools(find_documents, get_exchange_rate),
        reset=False,
        n_latest_messages=None,
    )
    return tools_agent


# ---------------------------------------------------------------------------
# 5. Short-term memory — read conversation history from Pixeltable
# ---------------------------------------------------------------------------

def get_chat_history() -> str:
    """Return the full conversation history as a formatted string."""
    memory = pxt.get_table("context_engineering.agent.memory")
    df = memory.select(memory.role, memory.content).collect()
    history = ""
    for role, content in zip(df["role"], df["content"]):
        history += f"{role}: {content}\n"
    return history


# ---------------------------------------------------------------------------
# 6. Long-term memory — embed conversation turns & export to LanceDB
# ---------------------------------------------------------------------------

def build_long_term_memory(embed_model, vector_db_path: str = "vector_db"):
    """
    Add embedding columns to the agent memory table and export to LanceDB
    for persistent, cross-session semantic search.
    """
    memory = pxt.get_table("context_engineering.agent.memory")

    # Computed column: "role: content" string for embedding
    memory.add_computed_column(
        user_content=pxtf.string.format("{0}: {1}", memory.role, memory.content),
        if_exists="ignore",
    )

    # Computed column: sentence-transformer embedding vector
    from pixeltable.functions import huggingface
    memory.add_computed_column(
        embedding=huggingface.sentence_transformer(
            memory.user_content, model_id="intfloat/e5-large-v2"
        ),
        if_exists="ignore",
    )

    # Embedding index for in-table vector search
    memory.add_embedding_index(
        column="user_content",
        idx_name="user_content_idx",
        embedding=embed_model,
        if_exists="ignore",
    )

    # Export to LanceDB for cross-session long-term memory
    pxt.io.export_lancedb(
        memory.select(memory.message_id, memory.user_content, memory.embedding),
        Path(vector_db_path),
        "semantic_memory",
        if_exists="append",
    )
    print(f"Long-term memory exported to '{vector_db_path}/semantic_memory'.")


# ---------------------------------------------------------------------------
# 7. Vector search — semantic retrieval from LanceDB
# ---------------------------------------------------------------------------

def search_memory(
    query: str,
    vector_db_path: str = "vector_db",
    top_k: int = 5,
) -> list[dict]:
    """
    Encode the query with the same sentence-transformer used during ingestion,
    search the LanceDB table, and return ranked results with cosine similarity.
    """
    # Lazy-load the model (cached on first call)
    if not hasattr(search_memory, "_model"):
        search_memory._model = SentenceTransformer("intfloat/e5-large-v2")
    model = search_memory._model

    q = model.encode("query: " + query, normalize_embeddings=False).astype("float32")
    q_norm = np.linalg.norm(q) + 1e-12

    db = lancedb.connect(vector_db_path)
    tbl = db.open_table("semantic_memory")
    df = tbl.search(q, vector_column_name="embedding").limit(top_k).to_pandas()

    def _cosine(v):
        v = np.asarray(v, dtype="float32")
        return float(np.dot(v, q) / ((np.linalg.norm(v) + 1e-12) * q_norm))

    df["cosine_similarity"] = [_cosine(v) for v in df["embedding"]]
    df = df.sort_values("cosine_similarity", ascending=False)

    return [
        {
            "message_id": row["message_id"],
            "user_content": row["user_content"],
            "cosine_similarity": float(row["cosine_similarity"]),
        }
        for _, row in df.iterrows()
    ]


# ---------------------------------------------------------------------------
# 8. Summarization agent — compress context blocks to fit token budgets
# ---------------------------------------------------------------------------

def create_summarizer_agent() -> Agent:
    """
    A lightweight agent that compresses any context block to a target token size
    while preserving key facts, numbers, and identifiers.
    """
    return Agent(
        name="summarizer_agent",
        system_prompt=(
            "You are a compression assistant.\n"
            "Condense the given block into a concise, lossless summary sized to the target.\n"
            "Keep original identifiers, references. Remove redundancy and off-topic lines.\n"
            "Preserve key facts and numbers. Output only the summary text."
        ),
    )


# Token counting helpers (uses cl100k_base, same tokenizer as GPT-4o)
_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text or ""))


def within_budget(text: str, max_tokens: int) -> bool:
    return count_tokens(text) <= max_tokens


def maybe_summarize_block(
    text: str,
    block_name: str,
    max_tokens: int,
    target_tokens: int | None = None,
    model: str = "gpt-4o-mini",
    summarizer_agent: Agent = None,
) -> str:
    """
    If `text` exceeds `max_tokens`, compress it to approximately `target_tokens`
    using the summarizer agent. Returns the original text if it fits.
    """
    text = (str(text) or "").strip()
    if not text or within_budget(text, max_tokens):
        return text

    tgt = target_tokens or max(256, max_tokens // 2)
    user_msg = (
        f"BLOCK NAME: {block_name}\n"
        f"TARGET SIZE: ~{tgt} tokens\n\n"
        f"CONTENT START\n{text}\nCONTENT END"
    )
    resp = summarizer_agent.chat(messages=user_msg, model=model)
    return resp.strip() if isinstance(resp, str) else resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# 9. Context prompt builder — assemble + optionally compress all context sources
# ---------------------------------------------------------------------------

def build_context_prompt(
    user_query: str,
    tool_ctx: str,
    chat_history: str,
    long_term_ctx: str,
) -> str:
    """
    Assemble raw context without any compression.
    Useful when total token count is already within budget.
    """
    return f"""========================
USER QUERY
========================
{user_query}

========================
CONTEXT BLOCKS (read-only)
========================
[TOOL OUTPUT]
- May include:
  • RAG output: response that includes PDF/Document search
  • Exchange rate: exchange details between physical and crypto currency finance
{tool_ctx or "(none)"}

[USER CHAT HISTORY]
- Prefer recent messages if conflicting
- Messages are sorted; latest messages are at the bottom
{chat_history or "(none)"}

[LONG-TERM USER DB HITS]
- Vector search results (highest cosine first)
- Only include items that look relevant to the query
{long_term_ctx or "(none)"}""".strip()


def build_context_prompt_with_summarization(
    user_query: str,
    tool_ctx: str,
    chat_history: str,
    long_term_ctx: str,
    summarizer_agent: Agent,
    tool_budget: int = 1200,
    chat_budget: int = 900,
    mem_budget: int = 900,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Like build_context_prompt, but each block is compressed with the
    summarizer agent if it exceeds its individual token budget.
    """
    tool_ctx_s = maybe_summarize_block(
        tool_ctx, "TOOL OUTPUT", max_tokens=tool_budget,
        model=model, summarizer_agent=summarizer_agent,
    )
    chat_hist_s = maybe_summarize_block(
        chat_history, "USER CHAT HISTORY", max_tokens=chat_budget,
        model=model, summarizer_agent=summarizer_agent,
    )
    mem_ctx_s = maybe_summarize_block(
        long_term_ctx, "LONG-TERM USER DB HITS", max_tokens=mem_budget,
        model=model, summarizer_agent=summarizer_agent,
    )

    return build_context_prompt(user_query, tool_ctx_s, chat_hist_s, str(mem_ctx_s))


# ---------------------------------------------------------------------------
# 10. Response agent — synthesizes all context into a final answer
# ---------------------------------------------------------------------------

RESPONSE_AGENT_SYSTEM_PROMPT = """
**1. Role & Objective**

You are a specialized AI assistant for context-based response generation.
Your primary objective is to synthesize information from a given set of
context blocks to answer a user's query directly and accurately.

**2. Core Workflow**

1. Analyze the Query: Deconstruct the user's request to identify the specific information needed.
2. Review Context: Scrutinize all provided context blocks: [TOOL OUTPUT], [USER CHAT HISTORY],
   and [LONG-TERM USER DB HITS].
3. Synthesize Information: Extract and combine only the relevant facts from the context.
4. Resolve Conflicts: Adhere to the following hierarchy of authority:
   1. [TOOL OUTPUT]  — most definitive (RAG / live market data)
   2. [LONG-TERM USER DB HITS] — all-session vector DB hits
   3. [USER CHAT HISTORY] — current session context
5. Handle Insufficiency: If the combined context is insufficient, answer from existing knowledge.

**3. Strict Constraints**

* No External Knowledge: Derive responses exclusively from the provided context.
* No Speculation: Do not infer or assume anything not explicitly stated in the context.

**4. Output Style**

* Concise & Factual: Short, direct answers. No filler.
* Actionable: Help the user complete a task or make a decision.
* Neutral Tone: Impartial, data-driven.
"""


def create_response_agent() -> Agent:
    return Agent(
        name="response_generation",
        system_prompt=RESPONSE_AGENT_SYSTEM_PROMPT,
    )


# ---------------------------------------------------------------------------
# 11. Full pipeline — wire everything together
# ---------------------------------------------------------------------------

def run_pipeline(
    user_query: str,
    tools_agent: Agent,
    summarizer_agent: Agent,
    response_agent: Agent,
    vector_db_path: str = "vector_db",
    use_summarization: bool = True,
) -> str:
    """
    End-to-end context engineering pipeline:
      1. Tool call  → tool output (RAG / exchange rate)
      2. Short-term → chat history from Pixeltable memory
      3. Long-term  → semantic search from LanceDB
      4. Assembly   → combined context prompt (with optional summarization)
      5. Generation → response agent produces the final answer
    """
    print(f"\n{'='*60}")
    print(f"Query: {user_query}")
    print("="*60)

    # Step 1: tool output
    print("[1/5] Running tool call...")
    tool_ctx = tools_agent.tool_call(user_query)

    # Step 2: short-term memory
    print("[2/5] Reading chat history...")
    chat_history = get_chat_history()

    # Step 3: long-term memory
    print("[3/5] Searching long-term memory...")
    long_term_results = search_memory(user_query, vector_db_path=vector_db_path)
    long_term_ctx = str(long_term_results)

    # Step 4: assemble context prompt
    print("[4/5] Building context prompt...")
    if use_summarization:
        context_prompt = build_context_prompt_with_summarization(
            user_query=user_query,
            tool_ctx=tool_ctx,
            chat_history=chat_history,
            long_term_ctx=long_term_ctx,
            summarizer_agent=summarizer_agent,
        )
    else:
        context_prompt = build_context_prompt(
            user_query=user_query,
            tool_ctx=tool_ctx,
            chat_history=chat_history,
            long_term_ctx=long_term_ctx,
        )

    # Step 5: generate response
    print("[5/5] Generating final response...")
    final_answer = response_agent.chat(context_prompt)

    print("\nAnswer:")
    print(final_answer)
    return final_answer


# ---------------------------------------------------------------------------
# 12. Main entry point
# ---------------------------------------------------------------------------

def main():
    # ── API keys ──────────────────────────────────────────────────────────────
    setup_api_keys()

    # ── Setup ────────────────────────────────────────────────────────────────
    print("Setting up RAG infrastructure...")
    documents_t, documents_chunks, embed_model = setup_rag()

    print("Ingesting financial documents...")
    ingest_documents(documents_t)

    # ── Tools ─────────────────────────────────────────────────────────────────
    find_documents = make_find_documents_tool(documents_chunks)

    # Replace with a real key from https://www.alphavantage.co/support/#api-key
    ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "<api-key>")
    get_exchange_rate = load_exchange_rate_tool(ALPHA_VANTAGE_API_KEY)

    # ── Agents ────────────────────────────────────────────────────────────────
    print("Initializing agents...")
    tools_agent = create_tools_agent(find_documents, get_exchange_rate)
    summarizer_agent = create_summarizer_agent()
    response_agent = create_response_agent()

    # ── Warm up short-term memory with a couple of turns ─────────────────────
    print("\nRunning initial queries to populate memory...")
    tools_agent.tool_call("Give me a brief summary about Alphabet earnings.")
    tools_agent.tool_call("What is the exchange rate between USD and bitcoin?")

    # ── Build long-term memory ────────────────────────────────────────────────
    print("\nBuilding long-term memory...")
    build_long_term_memory(embed_model)

    # ── Full pipeline demo ────────────────────────────────────────────────────
    queries = [
        "Summarize my previous conversations with you!",
        "What did Alphabet report for Q1 2024 earnings per share?",
        "What is the current BTC to USD rate?",
    ]

    for query in queries:
        run_pipeline(
            user_query=query,
            tools_agent=tools_agent,
            summarizer_agent=summarizer_agent,
            response_agent=response_agent,
            use_summarization=True,
        )


if __name__ == "__main__":
    main()
