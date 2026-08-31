import os
import uuid

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")


def _error_detail(e: requests.RequestException) -> str:
    """
    requests' default exception string (e.g. "503 Server Error: ...") drops
    the structured {"error": ..., "detail": ...} body the API's error
    handler sends back. Prefer that when it's present so failures are
    actually diagnosable from the UI instead of just "something failed".
    """
    resp = getattr(e, "response", None)
    if resp is not None:
        try:
            body = resp.json()
            if isinstance(body, dict) and ("detail" in body or "error" in body):
                return body.get("detail") or body.get("error")
        except ValueError:
            pass
    return str(e)

st.set_page_config(page_title="Local AI Assistant", page_icon="🤖", layout="centered")

if "session_id" not in st.session_state:
    # Round-trip the session id through the URL so a page reload (or
    # sharing/bookmarking the URL) reuses the same conversation instead of
    # silently starting a brand-new, empty one every time the Streamlit
    # session resets.
    existing_id = st.query_params.get("sid")
    st.session_state.session_id = existing_id or str(uuid.uuid4())
    st.query_params["sid"] = st.session_state.session_id
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.title("🤖 Local AI Assistant")
st.sidebar.caption(f"Backend: {API_BASE_URL}")
mode = st.sidebar.radio("Mode", ["💬 Chat (RAG)", "📄 Ingest Document", "🖼️ Classify Image"])

if mode == "💬 Chat (RAG)":
    if st.sidebar.button("🗑️ Clear conversation"):
        try:
            requests.delete(f"{API_BASE_URL}/chat/{st.session_state.session_id}", timeout=10)
        except requests.RequestException:
            pass
        st.session_state.chat_history = []
        st.rerun()

# --- Chat page ---
if mode == "💬 Chat (RAG)":
    st.header("Chat")

    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(text)

    user_message = st.chat_input("Ask something...")
    if user_message:
        st.session_state.chat_history.append(("user", user_message))
        with st.chat_message("user"):
            st.write(user_message)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/chat",
                        json={"message": user_message, "session_id": st.session_state.session_id},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    st.write(data["answer"])
                    if data.get("used_tools"):
                        st.caption(f"Tools used: {', '.join(data['used_tools'])}")
                    if data.get("sources"):
                        with st.expander(f"Sources ({len(data['sources'])})"):
                            for s in data["sources"]:
                                st.markdown(f"**{s['chunk_id']}** (score: {s['score']})")
                                st.text(s["text"])
                    st.session_state.chat_history.append(("assistant", data["answer"]))
                except requests.RequestException as e:
                    error_msg = f"Request failed: {_error_detail(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(("assistant", error_msg))

# --- Ingest page ---
elif mode == "📄 Ingest Document":
    st.header("Ingest a document")
    st.write("Upload a `.txt` or `.md` file to add it to the assistant's knowledge base.")
    uploaded_file = st.file_uploader("Choose a document", type=["txt", "md"])
    if uploaded_file is not None and st.button("Ingest"):
        with st.spinner("Chunking and embedding..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                resp = requests.post(f"{API_BASE_URL}/ingest", files=files, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                st.success(f"Ingested {data['chunks_created']} chunks from {data['filename']}")
                st.info("Switch to the Chat tab and ask about this document — it's now searchable.")
            except requests.RequestException as e:
                st.error(f"Ingest failed: {_error_detail(e)}")

# --- Classify page ---
elif mode == "🖼️ Classify Image":
    st.header("Classify an image")
    st.write("Upload an image to classify it with the CIFAR-10 ResNet50 model.")
    uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        st.image(uploaded_image, width=200)
        if st.button("Classify"):
            with st.spinner("Running inference..."):
                try:
                    files = {"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
                    resp = requests.post(f"{API_BASE_URL}/classify", files=files, timeout=60)
                    resp.raise_for_status()
                    data = resp.json()
                    if data.get("error"):
                        st.warning(data["error"])
                    else:
                        st.success(f"Prediction: **{data['predicted_class']}** ({data['confidence']:.1%} confidence)")
                        st.write("Top 3:")
                        for item in data["top_k"]:
                            st.write(f"- {item['class']}: {item['confidence']:.1%}")
                except requests.RequestException as e:
                    st.error(f"Classification failed: {_error_detail(e)}")