# Context Engineering with Pixeltable

A sophisticated AI pipeline that combines **RAG**, **tool calling**, **short & long-term memory**, and **context summarization** to produce accurate, context-aware responses.

---

## How it works

1. **Document Ingestion** — Financial PDFs are loaded into Pixeltable and split into chunks
2. **RAG Setup** — Chunks are embedded using `intfloat/e5-large-v2` and indexed for semantic search
3. **Tool Integration** — Two tools are available: document search (RAG) and live exchange rates (Alpha Vantage MCP)
4. **Agent** — An AI agent decides which tool to call based on the user query
5. **Short-term Memory** — Conversation history is stored in Pixeltable's agent memory table
6. **Long-term Memory** — Conversation turns are embedded and exported to LanceDB for cross-session semantic search
7. **Context Engineering** — Tool output, chat history, and long-term memory hits are assembled and summarized to fit token budgets
8. **Response Generation** — A final agent synthesizes all context into a concise, accurate answer

---

## Tech stack

- [Pixeltable](https://pixeltable.com) — AI data infrastructure and agent memory
- [Pixelagent](https://github.com/pixeltable/pixelagent) — Stateful agents with tool calling
- [LanceDB](https://lancedb.com) — Vector database for long-term memory
- [sentence-transformers](https://www.sbert.net) — `intfloat/e5-large-v2` for embeddings
- [OpenAI](https://openai.com) — GPT-4o / GPT-4o-mini for response and summarization
- [Alpha Vantage MCP](https://www.alphavantage.co) — Real-time currency and crypto exchange rates

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/sthirisha02003/context-engineering-pixeltable.git
cd context-engineering-pixeltable
```

### 2. Install dependencies

```bash
uv add pixeltable pixelagent openai lancedb sentence-transformers tiktoken
```

Or with pip:

```bash
pip install pixeltable pixelagent openai lancedb sentence-transformers tiktoken
```

### 3. Set your API keys

```bash
export OPENAI_API_KEY=your_openai_key_here
export ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
```

Get a free Alpha Vantage key at [alphavantage.co](https://www.alphavantage.co/support/#api-key)

### 4. Run

```bash
python context_engineering.py
```

---

## Project structure

```
context-engineering-pixeltable/
├── context_engineering.py   # Main pipeline script
├── README.md
└── .gitignore
```

---

## Example queries

The pipeline runs these queries out of the box:

- *"Summarize my previous conversations with you!"*
- *"What did Alphabet report for Q1 2024 earnings per share?"*
- *"What is the current BTC to USD rate?"*

---

## License

MIT
