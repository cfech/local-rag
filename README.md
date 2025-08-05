# 🧠 Local RAG Chatbot with Ollama on Mac

This base of this code was taken from [eplit](https://github.com/eplt/RAG_Ollama_Mac) based on the [medium article](https://eplt.medium.com/local-rag-chatbot-with-ollama-on-mac-4d4b37f977b4) 
  and [pruthvirajcyn](https://github.com/pruthvirajcyn/RAG_Ollama_Deepseek_Streamlit) and his [Medium article](https://medium.com/@pruthvirajc/implementing-a-local-rag-chat-bot-with-ollama-streamlit-and-deepseek-r1-a-practical-guide-46b1903f011f).

---

## ⚙️ About This Project

> Lightweight, private, and customizable retrieval-augmented chatbot locally on your computer.

- [Ollama](https://ollama.com/) for running open-source LLMs and embedding models locally.
- [Streamlit](https://streamlit.io/) for a clean and interactive chat UI.
- [ChromaDB](https://www.trychroma.com/) for storing and querying vector embeddings.

As of **2025-07-17**, I'm using:

- 🔍 Embedding model: `nomic-embed-text-v2-moe`
- 🧠 LLM: `gemma3n`

---

## 💡 Why Run a RAG Locally?

- **🔒 Privacy**: No data is sent to the cloud. Upload and query your documents entirely offline.
- **💸 Cost-effective**: No API tokens or cloud GPU costs. You only pay electricity.
- **📚 Better than summarizing**: With long PDFs or multiple documents, even summaries may not contain the context you need. A RAG chatbot can drill deeper and provide contextual answers.

> ✅ Recommended: At least **16GB of RAM** on your Mac. Preferably 24GB+ for smoother experience.

---

## 🛠️ 1. Installation

### 1. Clone the Repository

```bash
git clone TODO - Add Link
cd local_rag
```

### 2. Create a Venv and Install Dependencies

```bash
pipenv insatll
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

### 1. Start Ollama and Pull the Models

```bash
ollama serve
ollama pull gemma3n
ollama pull llama3.1:8b
ollama pull toshk0/nomic-embed-text-v2-moe:Q6_K
ollama pull mxbai-embed-large
```

### 2. Load Documents

Place your documents in the `data/` directory. Supported formats:
- **PDF files** (`.pdf`) - Research papers, books, documentation
- **Markdown files** (`.md`) - Notes, documentation, structured content

```bash
python ./src/load_docs.py
```

To reset and reload the vector database:

```bash
python ./src/load_docs.py --reset
```

### 3. Launch the Chatbot Interface

```bash
streamlit run ./src/UI.py
```

### 4. Start Chatting

Ask questions and the chatbot will respond using relevant context retrieved from your documents.

---

## 🧩 3. Features & Customization

### 📝 **Enhanced Document Support**
- **Smart Markdown Processing**: Preserves document structure with header-aware chunking
- **Mixed Document Types**: Process PDFs and Markdown files together in one knowledge base
- **Intelligent Chunk IDs**: Section-based naming for Markdown (e.g., `section_core_ai_concepts`) and page-based for PDFs

### ⚙️ **Customization Options**

- **✏️ Modify Prompts**  
  Update prompt templates in `UI.py` to guide the chatbot's tone or behavior.

- **🔄 Try Different Models**  
  Ollama supports various LLMs and embedding models. Run `ollama list` to see what's available or try pulling new ones.

- **⚙️ Tune Retrieval Parameters**  
  Adjust chunk size, overlaps, or top-K retrieval values in `load_docs.py` for improved performance.

- **🚀 Extend the Interface**  
  Add features like file upload, chat history, user authentication, or export options using Streamlit's powerful features.

---

## 🧯 4. Troubleshooting

- **Ollama not running?**  
  Make sure `ollama serve` is active in a terminal tab.

- **Missing models?**  
  Run `ollama list` to verify models are downloaded correctly.

- **Dependency issues?**  
  Double-check your Python version (3.7+) and re-create the virtual environment.

- **Streamlit errors?**  
  Ensure you're running the app from the correct path and activate your virtual environment.

---
