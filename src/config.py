"""
Configuration file for RAG system parameters with defaults and validation.
"""
from dataclasses import dataclass, field
from typing import List, Tuple
import os

@dataclass
class RAGConfig:
    """Configuration class for RAG system parameters."""
    
    # Core RAG Parameters
    similarity_threshold: float = 1.5
    max_context_docs: int = 3
    top_k_retrieval: int = 5
    
    # Text Splitting Parameters
    chunk_size: int = 800
    chunk_overlap: int = 80
    
    # Model Parameters
    chat_model: str = "llama3.1:8b"
    embedding_model: str = "mxbai-embed-large"
    
    # Response Generation Parameters
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 500
    
    # Advanced Parameters
    enable_debug: bool = False
    use_rag: bool = True  # If False, uses pure LLM mode
    
    # Parameter Ranges for UI Validation (using field with default_factory)
    SIMILARITY_RANGE: Tuple[float, float] = (0.5, 3.0)
    MAX_DOCS_RANGE: Tuple[int, int] = (1, 10)
    TOP_K_RANGE: Tuple[int, int] = (1, 20)
    CHUNK_SIZE_OPTIONS: List[int] = field(default_factory=lambda: [400, 600, 800, 1000, 1200])
    CHUNK_OVERLAP_RANGE: Tuple[int, int] = (0, 200)
    
    # Response Generation Parameter Ranges
    TEMPERATURE_RANGE: Tuple[float, float] = (0.0, 2.0)
    TOP_P_RANGE: Tuple[float, float] = (0.1, 1.0)
    MAX_TOKENS_RANGE: Tuple[int, int] = (50, 2000)
    
    def validate(self) -> bool:
        """Validate all parameters are within acceptable ranges."""
        if not (self.SIMILARITY_RANGE[0] <= self.similarity_threshold <= self.SIMILARITY_RANGE[1]):
            return False
        if not (self.MAX_DOCS_RANGE[0] <= self.max_context_docs <= self.MAX_DOCS_RANGE[1]):
            return False
        if not (self.TOP_K_RANGE[0] <= self.top_k_retrieval <= self.TOP_K_RANGE[1]):
            return False
        if self.chunk_size not in self.CHUNK_SIZE_OPTIONS:
            return False
        if not (self.CHUNK_OVERLAP_RANGE[0] <= self.chunk_overlap <= self.CHUNK_OVERLAP_RANGE[1]):
            return False
        if self.max_context_docs > self.top_k_retrieval:
            return False  # Can't use more docs than we retrieve
        # Validate generation parameters
        if not (self.TEMPERATURE_RANGE[0] <= self.temperature <= self.TEMPERATURE_RANGE[1]):
            return False
        if not (self.TOP_P_RANGE[0] <= self.top_p <= self.TOP_P_RANGE[1]):
            return False
        if not (self.MAX_TOKENS_RANGE[0] <= self.max_tokens <= self.MAX_TOKENS_RANGE[1]):
            return False
        return True
    
    def auto_adjust(self):
        """Auto-adjust dependent parameters to ensure validity."""
        # Ensure max_context_docs doesn't exceed top_k_retrieval
        if self.max_context_docs > self.top_k_retrieval:
            self.max_context_docs = min(self.max_context_docs, self.top_k_retrieval)
        
        # Ensure chunk_overlap is not greater than chunk_size
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = min(self.chunk_overlap, self.chunk_size // 2)

# Default configuration instance
DEFAULT_CONFIG = RAGConfig()

def get_installed_models() -> Tuple[List[str], List[str]]:
    """Get lists of installed Ollama models."""
    import subprocess
    try:
        # Get installed models from ollama list
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            installed_models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]  # First column is model name
                    installed_models.append(model_name)
            
            # Separate chat and embedding models
            chat_models = [m for m in installed_models if 'embed' not in m.lower()]
            embedding_models = [m for m in installed_models if 'embed' in m.lower()]
            
            return chat_models, embedding_models
        else:
            # Fallback to defaults if ollama command fails
            return [], []
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # Fallback if ollama not available
        return [], []

def get_model_choices() -> Tuple[List[str], List[str]]:
    """Get model choices with installed models first, then recommended ones."""
    installed_chat, installed_embed = get_installed_models()
    
    # Recommended models that might not be installed
    recommended_chat = [
        "llama3.2:3b",
        "phi3.5:3.8b", 
        "mixtral:8x7b",
        "qwen2.5:14b"
    ]
    
    recommended_embed = [
        "mxbai-embed-large",
        "nomic-embed-text",
        "all-minilm",
        "snowflake-arctic-embed"
    ]
    
    # Build chat model choices
    chat_choices = []
    if installed_chat:
        chat_choices.extend([f"✅ {model}" for model in installed_chat])
        chat_choices.append("--- Recommended (Install with 'ollama pull <model>') ---")
    
    # Add recommended models that aren't already installed
    for model in recommended_chat:
        if model not in installed_chat:
            chat_choices.append(f"📦 {model} (not installed)")
    
    # Build embedding model choices
    embed_choices = []
    if installed_embed:
        embed_choices.extend([f"✅ {model}" for model in installed_embed])
        embed_choices.append("--- Recommended (Install with 'ollama pull <model>') ---")
    
    # Add recommended models that aren't already installed
    for model in recommended_embed:
        if model not in installed_embed:
            embed_choices.append(f"📦 {model} (not installed)")
    
    return chat_choices, embed_choices

def extract_model_name(choice: str) -> str:
    """Extract actual model name from UI choice string."""
    if choice.startswith("✅ "):
        return choice[2:]  # Remove checkmark prefix
    elif choice.startswith("📦 "):
        # Remove package prefix and " (not installed)" suffix
        return choice[2:].replace(" (not installed)", "")
    elif "---" in choice:
        return ""  # Separator, not a model
    return choice

def load_config_from_file(filepath: str) -> RAGConfig:
    """Load configuration from a JSON file (future enhancement)."""
    # Placeholder for loading config from file
    return DEFAULT_CONFIG

def save_config_to_file(config: RAGConfig, filepath: str) -> bool:
    """Save configuration to a JSON file (future enhancement)."""
    # Placeholder for saving config to file
    return True

# Get model choices dynamically - after all functions are defined
AVAILABLE_CHAT_MODELS, AVAILABLE_EMBEDDING_MODELS = get_model_choices()