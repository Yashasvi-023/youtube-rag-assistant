import os
import streamlit as st

# ---------------------------------------------------------
# API key setup — must happen BEFORE importing rag_pipeline,
# since rag_pipeline.py creates the ChatGroq client as soon as
# it's imported, and ChatGroq reads GROQ_API_KEY from the
# environment at that moment.
#
# Locally: put your key in a file called .streamlit/secrets.toml
#   GROQ_API_KEY = "your_key_here"
# On Streamlit Community Cloud: set the same key in
#   App settings -> Secrets, in the same TOML format.
# Either way, st.secrets reads it, and we copy it into the
# environment so rag_pipeline.py (and ChatGroq) can find it.
# ---------------------------------------------------------
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from rag_pipeline import build_pipeline, generate_answer

st.set_page_config(page_title="YouTube Video Assistant", page_icon="🎥")
st.title("🎥 YouTube Video Assistant")
st.write("Paste a YouTube link, then ask questions about the video.")

# ---------------------------------------------------------
# session_state persists across Streamlit reruns (every click/keystroke
# reruns this whole file). We use it to remember:
#   - which video is currently loaded
#   - that video's retriever (so we don't rebuild it per question)
#   - the running chat history
# Without this, asking a second question would silently re-fetch the
# transcript and rebuild the whole vector store from scratch every time.
# ---------------------------------------------------------
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "current_video" not in st.session_state:
    st.session_state.current_video = None

if "messages" not in st.session_state:
    st.session_state.messages = []

video_url = st.text_input("YouTube video URL")

if video_url and video_url != st.session_state.current_video:
    with st.spinner("Reading transcript and building index... this can take a minute"):
        try:
            st.session_state.retriever = build_pipeline(video_url)
            st.session_state.current_video = video_url
            st.session_state.messages = []  # fresh chat for a new video
            st.success("Video indexed! Ask a question below.")
        except Exception as e:
            st.error(f"Couldn't process this video: {e}")
            st.session_state.retriever = None
            st.session_state.current_video = None

# Only show the question box once a video has been successfully indexed
if st.session_state.retriever is not None:

    # Replay previous messages in this session, so the chat feels continuous
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask something about the video...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = generate_answer(question, st.session_state.retriever)
                st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})