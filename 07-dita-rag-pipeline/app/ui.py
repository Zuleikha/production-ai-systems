"""
ui.py
Streamlit UI for the DITA RAG pipeline.
"""

import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="DITA Documentation Assistant", page_icon="📄", layout="centered")

st.title("DITA Documentation Assistant")
st.caption("Ask questions about your structured documentation")

with st.sidebar:
    st.header("Controls")
    n_results = st.slider("Chunks to retrieve", min_value=1, max_value=8, value=4)
    if st.button("Re-index Documents"):
        with st.spinner("Indexing..."):
            resp = requests.post(f"{API_URL}/index")
            if resp.status_code == 200:
                st.success(f"Indexed {resp.json()['indexed']} chunks")
            else:
                st.error(resp.text)
    st.markdown("---")
    st.markdown("**Sample questions**")
    st.markdown("- What are the system requirements?")
    st.markdown("- How do I fix a license error?")
    st.markdown("- How do I configure units?")
    st.markdown("- What causes application crashes?")

query = st.text_input("Ask a question", placeholder="e.g. How do I install OpenRoads Designer?")

if query:
    with st.spinner("Searching documentation..."):
        resp = requests.post(
            f"{API_URL}/ask",
            json={"query": query, "n_results": n_results}
        )

    if resp.status_code == 200:
        data = resp.json()
        st.markdown("### Answer")
        st.markdown(data["answer"])

        st.markdown("### Sources")
        for source in data["sources"]:
            with st.expander(f"{source['topic']} > {source['section']} (score: {source['score']})"):
                st.markdown(f"**Tags:** {source['tags']}")
    else:
        st.error(f"Error: {resp.text}")
