import os
from urllib.parse import urlparse

import streamlit as st
from rag import generate_answer, process_urls


st.set_page_config(page_title="Real Estate Research Tool", page_icon="🏠", layout="wide")


def load_api_key_from_streamlit_secrets():
    """Load GROQ_API_KEY from Streamlit Cloud secrets when available."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        if "GROQ_MODEL" in st.secrets:
            os.environ["GROQ_MODEL"] = st.secrets["GROQ_MODEL"]
    except Exception:
        # Local runs without .streamlit/secrets.toml are okay.
        pass


def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def parse_urls(raw_text: str) -> list[str]:
    """Accept URLs written one-per-line or comma separated."""
    possible_urls = []
    for line in raw_text.replace(",", "\n").splitlines():
        url = line.strip()
        if url:
            possible_urls.append(url)

    # Remove duplicates while preserving order.
    unique_urls = list(dict.fromkeys(possible_urls))
    return unique_urls


load_api_key_from_streamlit_secrets()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_urls" not in st.session_state:
    st.session_state.processed_urls = []
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

st.title("🏠 Real Estate Research Assistant")
st.caption("Paste public article URLs, process them, and ask questions from those pages.")

with st.sidebar:
    st.header("1) Paste URLs")
    urls_text = st.text_area(
        "Enter URLs",
        placeholder="https://example.com/article-1\nhttps://example.com/article-2",
        height=160,
        help="Paste one URL per line. Public pages work best.",
    )

    process_button = st.button("Process URLs", type="primary", use_container_width=True)

    if st.button("Clear Current Session", use_container_width=True):
        st.session_state.vector_store = None
        st.session_state.processed_urls = []
        st.session_state.chunk_count = 0
        st.success("Session cleared. Paste new URLs and process again.")

if process_button:
    urls = parse_urls(urls_text)
    invalid_urls = [url for url in urls if not is_valid_url(url)]

    if not urls:
        st.error("Please paste at least one URL.")
    elif invalid_urls:
        st.error("These URLs are not valid. Please use full links starting with http:// or https://")
        for bad_url in invalid_urls:
            st.write(f"- {bad_url}")
    elif not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is missing. Add it in Streamlit Cloud Secrets or in a local .env file.")
    else:
        progress_box = st.empty()
        try:
            with st.spinner("Loading pages and creating embeddings..."):
                vector_store, chunk_count = process_urls(urls, progress_callback=progress_box.info)

            st.session_state.vector_store = vector_store
            st.session_state.processed_urls = urls
            st.session_state.chunk_count = chunk_count
            progress_box.success(f"Done. Processed {len(urls)} URL(s) and created {chunk_count} text chunks.")
        except Exception as exc:
            st.session_state.vector_store = None
            st.error("Could not process these URLs. Some websites block scraping or the page may require login.")
            st.exception(exc)

if st.session_state.processed_urls:
    st.success(
        f"Ready: {len(st.session_state.processed_urls)} URL(s) processed "
        f"with {st.session_state.chunk_count} chunks."
    )
    with st.expander("Processed sources"):
        for source_url in st.session_state.processed_urls:
            st.write(source_url)
else:
    st.info("Paste URLs in the sidebar and click **Process URLs** before asking a question.")

st.header("2) Ask a Question")
query = st.text_input(
    "Question",
    placeholder="Example: What are the main real estate trends mentioned in these articles?",
)

if query:
    if st.session_state.vector_store is None:
        st.warning("Please process URLs first.")
    elif not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is missing. Add it in Streamlit Cloud Secrets or in a local .env file.")
    else:
        try:
            with st.spinner("Generating answer..."):
                answer, sources = generate_answer(query, st.session_state.vector_store)

            st.subheader("Answer")
            st.write(answer)

            if sources:
                st.subheader("Sources")
                for source in sources:
                    st.write(source)
        except Exception as exc:
            st.error("Could not generate answer.")
            st.exception(exc)
