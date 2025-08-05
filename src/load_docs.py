from langchain_community.document_loaders import PyPDFLoader, TextLoader
import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.schema.document import Document
from embedding import get_embedding_function
from langchain_chroma import Chroma
import re

# Get current working directory and data path
cwd = os.getcwd()
DATA_PATH = os.path.join(cwd, "data")  # This is now a folder, not a single file
CHROMA_PATH = "chroma"

def get_loader(file_path, file_extension):
    """Factory function to get appropriate document loader based on file extension."""
    loaders = {
        '.pdf': lambda path: PyPDFLoader(path),
        '.md': lambda path: TextLoader(path, encoding='utf-8')
    }
    
    loader_func = loaders.get(file_extension.lower())
    if loader_func:
        return loader_func(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

def load_documents():
    documents = []
    supported_extensions = {'.pdf', '.md'}
    
    # Group files by extension for efficient processing
    files_by_extension = {}
    for filename in os.listdir(DATA_PATH):
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension in supported_extensions:
            if file_extension not in files_by_extension:
                files_by_extension[file_extension] = []
            files_by_extension[file_extension].append(filename)
    
    # Process files by extension
    for extension, filenames in files_by_extension.items():
        print(f"Processing {len(filenames)} {extension} files...")
        for filename in filenames:
            file_path = os.path.join(DATA_PATH, filename)
            try:
                loader = get_loader(file_path, extension)
                file_docs = loader.load()
                print(f"Loaded {len(file_docs)} documents from {filename}")
                documents.extend(file_docs)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return documents


def split_documents(documents: list[Document]):
    """Split documents with markdown-aware splitting for better structure preservation."""
    all_chunks = []
    
    for doc in documents:
        # Determine if this is a markdown document based on source extension
        source_path = doc.metadata.get('source', '')
        is_markdown = source_path.lower().endswith('.md')
        
        if is_markdown:
            # For markdown files, use header-aware splitting first
            chunks = split_markdown_document(doc)
        else:
            # For PDF and other files, use standard recursive splitting
            chunks = split_standard_document(doc)
        
        all_chunks.extend(chunks)
    
    return all_chunks

def split_markdown_document(document: Document):
    """Split markdown document using header-aware splitting."""
    # Define markdown headers to split on
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    
    # First split by headers
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False  # Keep headers in the chunks for context
    )
    
    # Split by headers first
    header_chunks = markdown_splitter.split_text(document.page_content)
    
    # Then apply recursive splitting for size constraints
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    
    final_chunks = []
    for chunk in header_chunks:
        # Preserve the original metadata and add header metadata
        chunk.metadata.update(document.metadata)
        
        # If chunk is still too large, split it further
        if len(chunk.page_content) > 800:
            sub_chunks = text_splitter.split_documents([chunk])
            # Preserve the header metadata in sub-chunks
            for sub_chunk in sub_chunks:
                # Copy all header metadata from the parent chunk
                for key, value in chunk.metadata.items():
                    if key.startswith('Header') or key in ['source']:
                        sub_chunk.metadata[key] = value
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)
    
    return final_chunks

def split_standard_document(document: Document):
    """Split non-markdown documents using standard recursive splitting."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents([document])

def add_to_db(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    chunks_with_ids = calculate_chunk_ids(chunks)  # Giving each chunk an ID

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("All documents are already added.")



def calculate_chunk_ids(chunks):
    """Calculate unique IDs for chunks with different strategies for markdown vs PDF."""
    # Group chunks by source file for processing
    chunks_by_source = {}
    for chunk in chunks:
        source = chunk.metadata.get("source")
        if source not in chunks_by_source:
            chunks_by_source[source] = []
        chunks_by_source[source].append(chunk)
    
    # Process each source file separately
    for source, source_chunks in chunks_by_source.items():
        is_markdown = source.lower().endswith('.md')
        
        if is_markdown:
            calculate_markdown_chunk_ids(source_chunks)
        else:
            calculate_pdf_chunk_ids(source_chunks)
    
    return chunks

def calculate_markdown_chunk_ids(chunks):
    """Calculate chunk IDs for markdown files using section-based naming."""
    section_counters = {}
    
    for chunk in chunks:
        source = chunk.metadata.get("source")
        
        # Extract the most specific header from metadata
        section_name = "content"
        
        # Look for the deepest header level (Header 4 -> Header 3 -> Header 2 -> Header 1)
        for header_level in ["Header 4", "Header 3", "Header 2", "Header 1"]:
            if header_level in chunk.metadata and chunk.metadata[header_level]:
                header_value = chunk.metadata[header_level]
                # Clean the header value to make it filename-safe
                clean_header = re.sub(r'[^\w\s-]', '', header_value).strip()
                clean_header = re.sub(r'[-\s]+', '_', clean_header).lower()
                section_name = clean_header[:50]  # Limit length
                break
        
        # Create section key for counting
        section_key = f"{source}:section_{section_name}"
        
        # Increment counter for this section
        if section_key not in section_counters:
            section_counters[section_key] = 0
        else:
            section_counters[section_key] += 1
        
        # Create the chunk ID
        chunk_id = f"{section_key}:{section_counters[section_key]}"
        chunk.metadata["id"] = chunk_id

def calculate_pdf_chunk_ids(chunks):
    """Calculate chunk IDs for PDF files using the original page-based naming."""
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)



def main():
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_db(chunks)



if __name__ == "__main__":
    main()