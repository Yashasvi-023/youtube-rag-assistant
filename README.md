# 🎥 YouTube RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers questions about YouTube videos — paste a link, ask anything, and get answers grounded in the actual transcript, complete with timestamps.

**Live demo:** _add your Streamlit Cloud URL here once deployed_

---

## What it does

1. Paste a YouTube video URL
2. The app fetches the transcript, chunks it, and builds a searchable vector index
3. Ask questions in natural language
4. The assistant retrieves the most relevant transcript excerpts and generates an answer — citing the approximate timestamp where it's discussed

No fine-tuning, no training — just retrieval + prompting, the core idea behind RAG.

---

## Architecture

```
YouTube URL
    │
    ▼
Transcript fetch (youtube-transcript-api)
    │
    ▼
Chunking (RecursiveCharacterTextSplitter, with timestamp tracking)
    │
    ▼
Embeddings (sentence-transformers/all-mpnet-base-v2)
    │
    ▼
Vector store (Chroma)
    │
    ▼
User question ──► Similarity search (top-k chunks) ──► Prompt ──► LLM (Groq, Llama 3.3 70B) ──► Answer
```

---

## Tech stack

| Component        | Choice                                  |
|-------------------|------------------------------------------|
| Transcript fetch  | `youtube-transcript-api`                 |
| Framework         | LangChain (`langchain-core`, `langchain-text-splitters`, `langchain-huggingface`, `langchain-chroma`, `langchain-groq`) |
| Embeddings        | `sentence-transformers/all-mpnet-base-v2` |
| Vector store      | Chroma                                   |
| LLM               | Groq — Llama 3.3 70B (free tier, fast inference) |
| Frontend          | Streamlit                                |

---

## Running locally

```bash
git clone https://github.com/Yashasvi-023/youtube-rag-assistant.git
cd youtube-rag-assistant
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` with your [Groq API key](https://console.groq.com):
```toml
GROQ_API_KEY = "your_key_here"
```

Then run:
```bash
streamlit run app.py
```

---

## Development process

This project was built in stages, deliberately:

- **v1 — raw pipeline** (no LangChain): every step written by hand — transcript fetching, chunking, embeddings via `sentence-transformers`, Chroma storage, and generation via local Ollama — to make sure no part of the pipeline was a black box before introducing abstractions.
- **v2 — LangChain refactor**: same pipeline, rebuilt on LangChain's standard interfaces (`Document`, `Embeddings`, `VectorStore`, `Retriever`), to compare abstraction overhead vs. flexibility.
- **v3 — embedding model comparison**: swapped `all-MiniLM-L6-v2` for `all-mpnet-base-v2` to investigate a retrieval quality issue (below).
- **v4 — timestamps + cloud deployment**: restored per-chunk timestamps lost during the LangChain refactor, and swapped local Ollama for Groq's hosted API for deployability.

Exploratory notebooks for each stage are in `/notebooks`.

---

## Known limitation: semantic search can miss stylistically unusual content

While testing retrieval on a Formula 1 broadcast transcript, a major race incident (a crash, described in the commentary as *"INTO THE GRAVEL... FULL SAFETY CAR"*) was consistently ranked **last** among retrieved chunks for the query *"what did they say about [driver]?"* — even though it was arguably the most important moment involving that driver in the entire video.

**Root cause:** the crash was described in shouted, fragmented broadcast language, stylistically different from the calmer, descriptive commentary in surrounding chunks. This pattern held across two different embedding models (`all-MiniLM-L6-v2` and `all-mpnet-base-v2`) and multiple query phrasings — meaning it wasn't a weak-model problem, but a genuine limitation of pure semantic similarity search on stylistically unusual source text.

**Mitigation used:** increasing `k` (chunks retrieved) ensured the relevant chunk was still included in the LLM's context, even when ranked low — verified across five repeated test runs.

**Not yet implemented, but a natural next step:** hybrid search (combining semantic search with keyword-based methods like TF-IDF/BM25) would likely surface this kind of chunk more reliably, since exact/rare-word matching doesn't depend on stylistic similarity the way embeddings do.

---

## Other notes

- The assistant is instructed to answer only from the transcript, but smaller/local LLMs (tested with local Llama 3.2) were observed to occasionally fall back on outside training knowledge when the transcript had no relevant content — a known grounding challenge in RAG systems, not unique to this project.
- Retrieval quality depends on `k` and query specificity; a `k` of 8 was chosen after empirical testing on a ~20-chunk video transcript.
