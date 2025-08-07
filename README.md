# 🧠 Local RAG Chatbot with Ollama on Mac

This base of this code was taken from [eplit](https://github.com/eplt/RAG_Ollama_Mac) based on the [medium article](https://eplt.medium.com/local-rag-chatbot-with-ollama-on-mac-4d4b37f977b4) 
  and [pruthvirajcyn](https://github.com/pruthvirajcyn/RAG_Ollama_Deepseek_Streamlit) and his [Medium article](https://medium.com/@pruthvirajc/implementing-a-local-rag-chat-bot-with-ollama-streamlit-and-deepseek-r1-a-practical-guide-46b1903f011f).

---

## ⚙️ About This Project

> Lightweight, private, and customizable retrieval-augmented chatbot locally on your computer.

- [Ollama](https://ollama.com/) for running open-source LLMs and embedding models locally.
- [Streamlit](https://streamlit.io/) for a clean and interactive chat UI.
- [ChromaDB](https://www.trychroma.com/) for storing and querying vector embeddings.

Current configuration:

- 🔍 Embedding model: `mxbai-embed-large` (reliable and stable)
- 🧠 LLM: `llama3.1:8b` (excellent balance of quality and performance)
- 🔄 **Hybrid RAG**: Uses both your documents AND general AI knowledge

---

## 💡 Why Run a RAG Locally?

- **🔒 Privacy**: No data is sent to the cloud. Upload and query your documents entirely offline.
- **💸 Cost-effective**: No API tokens or cloud GPU costs. You only pay electricity.
- **📚 Better than summarizing**: With long PDFs or multiple documents, even summaries may not contain the context you need. A RAG chatbot can drill deeper and provide contextual answers.
- **🧠 Hybrid Intelligence**: Combines your private documents with general AI knowledge for comprehensive answers.

## 💾 System Requirements

- **Minimum**: 8GB RAM (use `phi3.5:3.8b` model)
- **Recommended**: 16GB RAM (use `llama3.1:8b` model) ✅
- **Optimal**: 32GB+ RAM (use `mixtral:8x7b` or larger models)
- **Storage**: ~6GB for models + your document data

---

## ⚡ Quick Start

```bash
# 1. Install Ollama and models
brew install ollama
ollama serve  # Keep running in terminal 1

# 2. In terminal 2: Setup project
git clone <your-repo-url>
cd local-rag
pipenv install

# 3. Pull models
ollama pull llama3.1:8b
ollama pull mxbai-embed-large

# 4. Load your documents
pipenv run python ./src/load_docs.py

# 5. Start chatting with tunable parameters! 🆕
pipenv run python ./src/gradio_ui.py
```

---

## 🛠️ 1. Detailed Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd local-rag
```

### 2. Create a Venv and Install Dependencies

```bash
pipenv install
```
**on mac venv are usually created under the `~/.local/virtualenv/` directory 

### 3. Activate Venv 

```bash
pipenv shell
```
---

## 🚀 2. Usage

### 1. Install Ollama
```bash
brew install ollama
```

### 2. Start Ollama and Pull the Models

```bash
ollama serve
ollama pull llama3.1:8b
ollama pull mxbai-embed-large
```

### 3. Load Documents

Place your documents in the `data/` directory. Supported formats:
- **PDF files** (`.pdf`) - Research papers, books, documentation
- **Markdown files** (`.md`) - Notes, documentation, structured content

```bash
pipenv run python ./src/load_docs.py
```

To reset and reload the vector database:

```bash
pipenv run python ./src/load_docs.py --reset
```

### 4. Launch the Chatbot Interface

**Option A: Enhanced Gradio Interface (Recommended)** 🆕
```bash
pipenv run python ./src/gradio_ui.py
```

**Option B: Simple Streamlit Interface**
```bash
pipenv run streamlit run ./src/UI.py
```

### 5. Start Chatting

Ask questions and the chatbot will respond intelligently using:
- **Your documents** when relevant information is found
- **General AI knowledge** to supplement or when no relevant docs exist
- **Both sources** for comprehensive answers

---

## 🧩 3. Features & Customization

### 📝 **Enhanced Document Support**
- **Smart Markdown Processing**: Preserves document structure with header-aware chunking
- **Mixed Document Types**: Process PDFs and Markdown files together in one knowledge base
- **Intelligent Chunk IDs**: Section-based naming for Markdown (e.g., `section_core_ai_concepts`) and page-based for PDFs
- **Hybrid RAG**: Combines document knowledge with general AI knowledge for comprehensive answers

### 🎛️ **Interactive Parameter Tuning** (Gradio Interface)
- **Real-time Sliders**: Adjust similarity threshold, document count, and retrieval settings
- **Model Selection**: Switch between different chat and embedding models on-the-fly
- **Debug Panel**: See exactly which documents are used and similarity scores
- **Live Configuration**: No need to restart - parameters change immediately
- **Processing Insights**: View context length, processing time, and source attribution

### ⚙️ **Customization Options**

- **✏️ Modify Prompts**  
  Update prompt templates in `src/rag_query.py` to guide the chatbot's tone or behavior.

- **🔄 Try Different Models**  
  Ollama supports various LLMs and embedding models. Run `ollama list` to see what's available or try pulling new ones.

- **⚙️ Tune Retrieval Parameters**  
  Adjust `SIMILARITY_THRESHOLD` and `MAX_CONTEXT_DOCS` in `src/rag_query.py`, or chunk size/overlaps in `src/load_docs.py`.

- **🚀 Extend the Interface**  
  Add features like file upload, chat history, user authentication, or export options using Streamlit's powerful features.

---

## 🧯 4. Troubleshooting

- **Ollama not running?**  
  Make sure `ollama serve` is active in a terminal tab.

- **Missing models?**  
  Run `ollama list` to verify models are downloaded correctly.

- **"decode: cannot decode batches" error?**  
  Switch to `mxbai-embed-large` embedding model (more compatible than `nomic-embed-text-v2-moe`).

- **Out of memory errors?**  
  Try smaller models: `phi3.5:3.8b` for chat or `llama3.2:3b` for very low memory.

- **No relevant documents found?**  
  The system will use general AI knowledge. Adjust `SIMILARITY_THRESHOLD` in `src/rag_query.py`.

- **Dependency issues?**  
  Ensure Python 3.12+ and run `pipenv install` to recreate the virtual environment.  

- **Streamlit errors?**  
  Ensure you're running commands with `pipenv run` prefix from the project directory.

---
