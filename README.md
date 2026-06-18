# 🏠 Real Estate RAG Assistant

> An AI-powered real estate research chatbot that uses **Retrieval-Augmented Generation (RAG)** to answer questions from any real estate article or web page — in real time.

---

## 📌 Project Overview

This project is a full-stack RAG (Retrieval-Augmented Generation) application built with **Streamlit**, **LangChain**, and **Groq LLMs**. Users paste URLs of real estate articles, the app scrapes and embeds the content into a vector store, and then answers natural language questions grounded in those sources.

---

## ✨ Key Features

- 🔗 **URL-based ingestion** — Paste any public real estate article URL and the app loads and processes it automatically
- 🧠 **RAG pipeline** — Uses semantic search over embedded chunks to retrieve relevant context before answering
- ⚡ **Groq LLM** — Powered by `llama-3.3-70b-versatile` via Groq for ultra-fast inference
- 🗂️ **In-memory vector store** — Uses Chroma (in-memory) per session — no persistent DB required
- 🔍 **Source citations** — Every answer links back to the original source URLs
- 🧹 **Session management** — Clear and reload sources anytime without restarting the app
- 📓 **Companion notebooks** — Includes Jupyter notebooks for document loading and text splitting concepts

---

## 🛠️ Technologies Used

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB (in-memory) |
| Document Loading | LangChain `UnstructuredURLLoader` |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| Environment | `python-dotenv` |
| Notebooks | Jupyter / IPython |

---

## 📁 Project Structure

```
real-estate-rag-assistant/
│
├── real-estate-tool/           # Main Streamlit application
│   ├── main.py                 # Streamlit UI & app logic
│   ├── rag.py                  # RAG pipeline (loading, embedding, retrieval, generation)
│   ├── requirements.txt        # Python dependencies
│   ├── .env.example            # Environment variable template (copy to .env)
│   └── resources/              # Static assets
│
├── notebooks/                  # Jupyter notebooks (learning/exploration)
│   ├── 1_document_loader.ipynb # Document loading with LangChain
│   ├── 2_text_splitter.ipynb   # Text splitting strategies
│   └── patient_records.csv     # Sample CSV for notebook demos
│
├── screenshots/                # App screenshots for documentation
│   └── app_screenshot.png
│
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/real-estate-rag-assistant.git
cd real-estate-rag-assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r real-estate-tool/requirements.txt
```

### 4. Set up environment variables

```bash
cp real-estate-tool/.env.example real-estate-tool/.env
```

Then open `.env` and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

> 🔑 Get your free Groq API key at [console.groq.com](https://console.groq.com)

---

## 📦 Requirements

```
python-dotenv
streamlit
unstructured
langchain-chroma
langchain-groq
langchain-community
langchain-huggingface
langchain-text-splitters
sentence-transformers
protobuf
```

Install everything with:

```bash
pip install -r real-estate-tool/requirements.txt
```

---

## 🚀 How to Run

```bash
cd real-estate-tool
streamlit run main.py
```

Then open your browser at `http://localhost:8501`

**Usage:**
1. Paste one or more public real estate article URLs into the sidebar
2. Click **Process URLs**
3. Type your question in the main input box
4. Get an AI-generated answer with source citations

---

## 📸 Screenshots

### Main Interface
![Real Estate RAG Assistant - Main UI](screenshots/app_screenshot.png)

---

## 🔒 API Key Security

- The `.env` file is listed in `.gitignore` and will **never** be committed to this repository
- For **Streamlit Cloud deployment**, add your key under `Settings → Secrets`:
  ```toml
  GROQ_API_KEY = "your_key_here"
  GROQ_MODEL = "llama-3.3-70b-versatile"
  ```
- The app reads secrets automatically from `st.secrets` when deployed on Streamlit Cloud

---

## 🔮 Future Improvements

- [ ] **PDF upload support** — Allow users to upload PDF documents in addition to URLs
- [ ] **Persistent vector store** — Option to save and reload processed sources across sessions
- [ ] **Multi-language support** — Handle articles in languages other than English
- [ ] **Answer confidence scores** — Display similarity scores for retrieved chunks
- [ ] **Export answers** — Download Q&A history as PDF or CSV
- [ ] **Authentication** — Add user login for Streamlit Cloud deployments
- [ ] **Streaming responses** — Stream LLM output token-by-token for better UX

---

## 👤 Author

**MOAZ**  
- 📧 Contact via GitHub Issues or Pull Requests
- 🔗 Built as part of a Generative AI / RAG learning project

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with ❤️ using LangChain, Groq, and Streamlit*
