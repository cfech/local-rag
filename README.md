# 🧠 Local RAG Chatbot with Ollama on Mac

This base of this code was taken from [eplit](https://github.com/eplt/RAG_Ollama_Mac) based on the [medium article](https://eplt.medium.com/local-rag-chatbot-with-ollama-on-mac-4d4b37f977b4) 
  and [pruthvirajcyn](https://github.com/pruthvirajcyn/RAG_Ollama_Deepseek_Streamlit) and his [Medium article](https://medium.com/@pruthvirajc/implementing-a-local-rag-chat-bot-with-ollama-streamlit-and-deepseek-r1-a-practical-guide-46b1903f011f).

---

## ⚙️ About This Project

> Lightweight, private, and customizable retrieval-augmented chatbot locally on your computer.

- [Ollama](https://ollama.com/) for running open-source LLMs and embedding models locally.
- [Gradio](https://gradio.app/) for a modern, interactive chat UI with real-time parameter tuning.
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

## ⚡ Quick Start (Complete Setup)

### Prerequisites
- **Python 3.8+** (check with `python3 --version`)
- **macOS** with Homebrew installed
- **8GB+ RAM** recommended

### Step-by-Step Setup

```bash
# 1. Install pipenv if you don't have it
pip3 install pipenv

# 2. Clone and setup project
git clone https://github.com/cfech/local-rag.git  
cd local-rag
pipenv install  # Creates virtual environment and installs dependencies

# 3. Install and start Ollama
brew install ollama
ollama serve  # Keep this running in terminal 1

# 4. In a NEW terminal (terminal 2), pull required models
ollama pull llama3.1:8b      # ~4.9GB download
ollama pull mxbai-embed-large # ~669MB download

# 5. Verify models are installed
ollama list  # Should show both models

# 6. Create data directory and add your documents
mkdir -p data
# Copy your PDF and Markdown files to the data/ folder

# 7. Build the vector database from your documents
pipenv run python ./src/load_docs.py

# 8. Launch the chatbot interface
pipenv run python ./src/gradio_ui.py

# 9. Open your browser and go to: http://localhost:7860
```

🎉 **You're ready to chat!** The interface will show all your documents are loaded and you can start asking questions.

---

## 🛠️ 1. Detailed Installation

### 1. Clone the Repository

```bash
git clone https://github.com/cfech/local-rag.git
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

**Primary Interface: Enhanced Gradio UI** ⭐
```bash
pipenv run python ./src/gradio_ui.py
```
- **Access**: Open http://localhost:7860 in your browser
- **Features**: Real-time parameter tuning, model selection, debug info
- **Usage**: Type questions and press Enter to chat

**Alternative: Simple Streamlit Interface** (Basic)
```bash
pipenv run streamlit run ./src/UI.py
```
- **Access**: Opens automatically in browser
- **Features**: Basic chat interface without parameter tuning

### 5. Start Chatting

**How to Use the Interface:**
1. **Type your question** in the message box
2. **Press Enter** to send (or click Send button)
3. **Adjust parameters** in real-time using the expanded control panel
4. **Monitor debug info** to see which documents are being used

**The chatbot responds intelligently using:**
- **Your documents** when relevant information is found (shows sources)
- **General AI knowledge** to supplement incomplete document info
- **Both sources** for comprehensive, contextual answers

**Parameter Quick Guide:**
- **Temperature**: Higher = more creative, Lower = more focused
- **Top-P**: Controls response diversity (0.9 recommended)
- **Max Tokens**: Response length (500 = ~400 words)
- **Similarity**: Lower = stricter document matching

---

## 🧩 3. Features & Customization

### 📝 **Enhanced Document Support**
- **Smart Markdown Processing**: Preserves document structure with header-aware chunking
- **Mixed Document Types**: Process PDFs and Markdown files together in one knowledge base
- **Intelligent Chunk IDs**: Section-based naming for Markdown (e.g., `section_core_ai_concepts`) and page-based for PDFs
- **Hybrid RAG**: Combines document knowledge with general AI knowledge for comprehensive answers

### 🎛️ **Interactive Parameter Tuning** (Gradio Interface)
- **Response Generation**: Control creativity (temperature), focus (top-p), and length (max tokens)
- **Real-time Sliders**: Adjust similarity threshold, document count, and retrieval settings  
- **Smart Model Selection**: Automatically detects installed models with ✅ indicators
- **🔄 Model Refresh**: Click refresh button to detect newly installed models without restart
- **Debug Panel**: See exactly which documents are used and similarity scores
- **Expanded View**: All parameter sections open by default for easy access
- **Enter Key**: Type and press Enter to send messages (no clicking required)
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

### Common Setup Issues

- **"pipenv: command not found"**  
  Install pipenv: `pip3 install pipenv` or `brew install pipenv`

- **"ollama: command not found"**  
  Install Ollama: `brew install ollama` then start: `ollama serve`

- **Ollama not running?**  
  Make sure `ollama serve` is active in a terminal tab. Check with `curl http://localhost:11434`

- **Models not downloading?**  
  Ensure Ollama is running first, then: `ollama pull llama3.1:8b && ollama pull mxbai-embed-large`

- **"No documents found" in data/ directory**  
  Add PDF or Markdown files to the `data/` folder, then run `pipenv run python ./src/load_docs.py --reset`

### Runtime Issues

- **Port 7860 already in use**  
  Kill existing processes: `lsof -ti:7860 | xargs kill` or change port in `src/gradio_ui.py`

- **"decode: cannot decode batches" error**  
  Switch to `mxbai-embed-large` embedding model (more compatible than others)

- **Out of memory errors**  
  Try smaller models: `phi3.5:3.8b` for chat or `llama3.2:3b` for very low memory

- **No relevant documents found**  
  The system will use general AI knowledge. Lower similarity threshold in the UI or add more documents

- **Interface not loading**  
  Check terminal for errors, ensure all dependencies installed: `pipenv install --dev`

### Model Management

- **Check installed models**: `ollama list`
- **Remove unused models**: `ollama rm model-name`  
- **Refresh models in UI**: Click the 🔄 button next to model dropdowns

### Performance Tips

- **First response slow?** Models load into memory on first use (30-60s)
- **Memory usage high?** Restart Ollama service: `killall ollama && ollama serve`
- **Database issues?** Reset and rebuild: `pipenv run python ./src/load_docs.py --reset`

---
