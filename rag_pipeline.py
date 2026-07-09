import os
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# ---------------------------------------------------------
# Models are loaded ONCE when this file is imported, not every
# time a function runs — loading them is slow, so we do it here.
# ---------------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# GROQ_API_KEY is read from the environment, NOT hardcoded here.
# When running locally: set it in your terminal before running streamlit, e.g.
#   setx GROQ_API_KEY "your_key_here"      (Windows, one-time)
# When deployed on Streamlit Cloud: set it in the app's Secrets settings
# (st.secrets), which app.py will load into the environment for you.
llm = ChatGroq(model="llama-3.3-70b-versatile")


def get_transcript(video_url):
    """Fetch the transcript for a YouTube video from its URL."""
    if "youtu.be/" in video_url:
        video_id = video_url.split("youtu.be/")[1].split("?")[0]
    elif "v=" in video_url:
        video_id = video_url.split("v=")[1].split("&")[0]
    else:
        raise ValueError("Couldn't extract video ID from this URL")

    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id)
    return transcript


def build_documents(transcript):
    """Turn a fetched transcript into timestamped chunks (LangChain Documents)."""
    full_text = ""
    offset_map = []

    for snippet in transcript.snippets:
        offset_map.append((len(full_text), snippet.start))
        full_text += snippet.text + " "

    doc = Document(page_content=full_text, metadata={"source": "youtube"})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True
    )
    chunks = splitter.split_documents([doc])

    for chunk in chunks:
        chunk_start = chunk.metadata["start_index"]
        best_time = offset_map[0][1]
        for offset, timestamp in offset_map:
            if offset <= chunk_start:
                best_time = timestamp
            else:
                break
        chunk.metadata["start_time"] = best_time

    return chunks


def build_pipeline(video_url):
    """
    Full indexing pipeline for one video: fetch transcript, chunk it,
    embed + store it, and return a retriever ready to be queried.
    This is the slow part — only run it once per video, not per question.
    """
    transcript = get_transcript(video_url)
    chunks = build_documents(transcript)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="video_transcript"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    return retriever


def generate_answer(question, retriever):
    """
    Answer a question using a given video's retriever.
    retriever is passed in explicitly (not a global), so the app can
    hold a different retriever per video in st.session_state.
    """
    retrieved_docs = retriever.invoke(question)

    context = ""
    for doc in retrieved_docs:
        start_time = doc.metadata["start_time"]
        context += f"[{start_time:.1f}s] {doc.page_content}\n\n"

    prompt = f"""Answer the question using only the transcript excerpts below.
Mention the approximate timestamp where the answer is discussed.

Transcript excerpts:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    return response.content