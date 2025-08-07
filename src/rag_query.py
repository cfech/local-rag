import argparse
from langchain_chroma import Chroma  # Updated import for Chroma
from langchain_ollama import OllamaLLM  # Updated import for Ollama
from langchain.prompts import ChatPromptTemplate
from embedding import get_embedding_function
from config import RAGConfig, DEFAULT_CONFIG
import re
from typing import Optional, Tuple, List, Dict, Any

CHROMA_PATH = "chroma"

PROMPT= """
You are a knowledgeable AI assistant. Use the following context from documents to answer the question, but you can also supplement with your general knowledge when helpful.

Document Context:
{context}

---

Instructions:
- If the document context contains relevant information, prioritize and reference it
- If the context is incomplete or doesn't fully answer the question, supplement with your general knowledge
- If the context doesn't contain relevant information, answer using your general knowledge but mention this
- Always be clear about what information comes from the documents vs. your general knowledge
- Provide helpful, comprehensive answers

Question: {question}
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    response, debug_info = query_rag(query_text)
    print(f"\nResponse: {response}")
    print(f"Debug Info: {debug_info}")



def remove_think_tags(response):
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

def query_rag(query: str, config: Optional[RAGConfig] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Enhanced RAG query function with tunable parameters.
    
    Args:
        query: The user's question
        config: RAGConfig object with tunable parameters (uses defaults if None)
    
    Returns:
        Tuple of (response_text, debug_info)
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    debug_info = {
        "sources": [],
        "similarity_scores": [],
        "context_length": 0,
        "processing_time": 0,
        "used_rag": False,
        "num_docs_retrieved": 0,
        "num_docs_used": 0
    }
    
    import time
    start_time = time.time()
    
    # Prepare the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB with configurable top-k
    results = db.similarity_search_with_score(query, k=config.top_k_retrieval)
    debug_info["num_docs_retrieved"] = len(results)

    # Check if we have good quality matches using configurable threshold
    relevant_docs = [(doc, score) for doc, score in results if score < config.similarity_threshold]
    
    if relevant_docs and config.use_rag:
        # Use relevant documents with configurable max docs
        context_docs = relevant_docs[:config.max_context_docs]
        context = "\n\n---\n\n".join([doc.page_content for doc, _score in context_docs])
        sources = [doc.metadata.get("id", "Unknown") for doc, _score in context_docs]
        similarity_scores = [float(score) for _doc, score in context_docs]
        
        debug_info["used_rag"] = True
        debug_info["sources"] = sources
        debug_info["similarity_scores"] = similarity_scores
        debug_info["num_docs_used"] = len(context_docs)
        debug_info["context_length"] = len(context)
        
        print(f"Using {len(context_docs)} relevant documents from your knowledge base")
    else:
        # No relevant documents found or RAG disabled - use general knowledge
        context = "No relevant information found in the document knowledge base." if config.use_rag else "RAG disabled - using general AI knowledge only."
        sources = ["General AI Knowledge"]
        debug_info["sources"] = sources
        debug_info["context_length"] = len(context)
        print("No relevant documents found - using general knowledge" if config.use_rag else "RAG disabled - using general knowledge")

    prompt_template = ChatPromptTemplate.from_template(PROMPT)
    prompt = prompt_template.format(context=context, question=query)

    # Use configurable model with generation parameters
    model = OllamaLLM(
        model=config.chat_model,
        temperature=config.temperature,
        top_p=config.top_p,
        num_predict=config.max_tokens  # Ollama's parameter for max tokens
    )
    response_text = model.invoke(prompt)
    response = remove_think_tags(response_text)

    debug_info["processing_time"] = time.time() - start_time

    if config.enable_debug:
        formatted_response = f"Response: {response_text}\nSources: {sources}\nDebug: {debug_info}"
        print(formatted_response)
    
    return response, debug_info

# Backward compatibility wrapper
def query_rag_simple(query: str) -> str:
    """Simple wrapper for backward compatibility with existing code."""
    response, _ = query_rag(query)
    return response


if __name__ == "__main__":
    main()