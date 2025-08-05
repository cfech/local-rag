import argparse
from langchain_chroma import Chroma  # Updated import for Chroma
from langchain_ollama import OllamaLLM  # Updated import for Ollama
from langchain.prompts import ChatPromptTemplate
from embedding import get_embedding_function
import re

CHROMA_PATH = "chroma"

# Configuration
SIMILARITY_THRESHOLD = 1.5  # Lower scores = better matches. Adjust based on your needs
MAX_CONTEXT_DOCS = 3        # Maximum number of documents to include in context

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
    query_rag(query_text)



def remove_think_tags(response):
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

def query_rag(query: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query, k=5)

    # Check if we have good quality matches (similarity score threshold)
    relevant_docs = [(doc, score) for doc, score in results if score < SIMILARITY_THRESHOLD]
    
    if relevant_docs:
        # Use relevant documents
        context = "\n\n---\n\n".join([doc.page_content for doc, _score in relevant_docs[:MAX_CONTEXT_DOCS]])
        sources = [doc.metadata.get("id", None) for doc, _score in relevant_docs[:MAX_CONTEXT_DOCS]]
        print(f"Using {len(relevant_docs[:MAX_CONTEXT_DOCS])} relevant documents from your knowledge base")
    else:
        # No relevant documents found - use general knowledge
        context = "No relevant information found in the document knowledge base."
        sources = ["General AI Knowledge"]
        print("No relevant documents found - using general knowledge")

    prompt_template = ChatPromptTemplate.from_template(PROMPT)
    prompt = prompt_template.format(context=context, question=query)
    # print(prompt)

    #model = OllamaLLM(model="mistral")
    # model = OllamaLLM(model="gemma3n:latest")
    model = OllamaLLM(model="llama3.1:8b")
    response_text = model.invoke(prompt)
    response = remove_think_tags(response_text)

    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response


if __name__ == "__main__":
    main()