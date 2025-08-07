"""
Gradio-based UI for the RAG system with tunable parameters.
"""

import gradio as gr
import json
from typing import List, Tuple, Dict, Any
from rag_query import query_rag
from config import RAGConfig, DEFAULT_CONFIG, AVAILABLE_CHAT_MODELS, AVAILABLE_EMBEDDING_MODELS, extract_model_name, get_model_choices
import time

class RAGInterface:
    def __init__(self):
        self.current_config = RAGConfig()
        
    def chat_with_rag(
        self,
        message: str,
        history: List[dict],
        similarity_threshold: float,
        max_context_docs: int,
        top_k_retrieval: int,
        chat_model: str,
        embedding_model: str,
        use_rag: bool,
        chunk_size: int,
        chunk_overlap: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        enable_debug: bool
    ) -> Tuple[str, List[dict], str]:
        """
        Main chat function that handles user queries with tunable parameters.
        """
        if not message.strip():
            return "", history, ""
        
        # Update configuration with current slider values
        self.current_config.similarity_threshold = similarity_threshold
        self.current_config.max_context_docs = max_context_docs
        self.current_config.top_k_retrieval = top_k_retrieval
        self.current_config.chat_model = extract_model_name(chat_model)
        self.current_config.embedding_model = extract_model_name(embedding_model)
        self.current_config.use_rag = use_rag
        self.current_config.chunk_size = chunk_size
        self.current_config.chunk_overlap = chunk_overlap
        self.current_config.temperature = temperature
        self.current_config.top_p = top_p
        self.current_config.max_tokens = max_tokens
        self.current_config.enable_debug = enable_debug
        
        # Auto-adjust dependent parameters
        self.current_config.auto_adjust()
        
        try:
            # Get response from RAG system
            response, debug_info = query_rag(message, self.current_config)
            
            # Update chat history (convert to messages format for new Gradio)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            
            # Format debug information
            debug_text = self.format_debug_info(debug_info) if enable_debug else ""
            
            return "", history, debug_text
            
        except Exception as e:
            error_response = f"Error: {str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_response})
            return "", history, f"Error occurred: {str(e)}"
    
    def format_debug_info(self, debug_info: Dict[str, Any]) -> str:
        """Format debug information for display."""
        debug_lines = [
            f"🔍 **Debug Information**",
            f"⏱️ Processing Time: {debug_info['processing_time']:.2f}s",
            f"📄 Documents Retrieved: {debug_info['num_docs_retrieved']}",
            f"📋 Documents Used: {debug_info['num_docs_used']}",
            f"🧠 Used RAG: {'Yes' if debug_info['used_rag'] else 'No'}",
            f"📏 Context Length: {debug_info['context_length']} chars",
        ]
        
        if debug_info['sources']:
            debug_lines.append(f"📚 **Sources:**")
            for i, source in enumerate(debug_info['sources']):
                score_text = f" (score: {debug_info['similarity_scores'][i]:.3f})" if i < len(debug_info['similarity_scores']) else ""
                debug_lines.append(f"  {i+1}. {source}{score_text}")
        
        return "\n".join(debug_lines)
    
    def reset_chat(self) -> Tuple[List[dict], str]:
        """Reset the chat history."""
        return [], ""
    
    def refresh_models(self) -> Tuple[List[str], List[str]]:
        """Refresh the list of available models from Ollama."""
        try:
            chat_choices, embed_choices = get_model_choices()
            return chat_choices, embed_choices
        except Exception as e:
            # Return current choices if refresh fails
            return AVAILABLE_CHAT_MODELS, AVAILABLE_EMBEDDING_MODELS
    
    def create_interface(self):
        """Create and configure the Gradio interface."""
        
        with gr.Blocks(
            title="🧠 Local RAG Chatbot with Tunable Parameters",
            theme=gr.themes.Soft()
        ) as interface:
            
            gr.Markdown("# 🧠 Local RAG Chatbot")
            gr.Markdown("Ask questions and tune parameters in real-time! The system uses both your documents and general AI knowledge.")
            
            with gr.Row():
                with gr.Column(scale=3):  # Increased from 2 to 3 for bigger chat window
                    # Main chat interface
                    chatbot = gr.Chatbot(
                        height=600,  # Increased from 500 to 600
                        placeholder="Chat history will appear here...",
                        label="Chat with RAG System",
                        type="messages"  # Fix deprecation warning
                    )
                    
                    msg = gr.Textbox(
                        placeholder="Type your question here and press Enter...",
                        label="Message",
                        lines=1,  # Changed to single line for better Enter key behavior
                        autofocus=True,
                        show_label=False,
                        container=False,
                        max_lines=1  # Ensure single line behavior
                    )
                    
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat", variant="secondary")
                        send_btn = gr.Button("Send", variant="primary")
                
                with gr.Column(scale=1):
                    # Parameter controls panel - consolidated into one box
                    with gr.Group():
                        gr.Markdown("## ⚙️ RAG Parameters")
                        
                        with gr.Accordion("🎛️ Response Generation", open=True):
                            temperature = gr.Slider(
                                minimum=DEFAULT_CONFIG.TEMPERATURE_RANGE[0],
                                maximum=DEFAULT_CONFIG.TEMPERATURE_RANGE[1],
                                step=0.1,
                                value=DEFAULT_CONFIG.temperature,
                                label="Temperature",
                                info="Higher = more creative, Lower = more focused"
                            )
                            
                            top_p = gr.Slider(
                                minimum=DEFAULT_CONFIG.TOP_P_RANGE[0],
                                maximum=DEFAULT_CONFIG.TOP_P_RANGE[1],
                                step=0.1,
                                value=DEFAULT_CONFIG.top_p,
                                label="Top-P",
                                info="Nucleus sampling - lower = more focused"
                            )
                            
                            max_tokens = gr.Slider(
                                minimum=DEFAULT_CONFIG.MAX_TOKENS_RANGE[0],
                                maximum=DEFAULT_CONFIG.MAX_TOKENS_RANGE[1],
                                step=50,
                                value=DEFAULT_CONFIG.max_tokens,
                                label="Max Tokens",
                                info="Maximum response length"
                            )
                        
                        with gr.Accordion("🔍 Core RAG Settings", open=True):
                            similarity_threshold = gr.Slider(
                                minimum=0.5,
                                maximum=3.0,
                                step=0.1,
                                value=DEFAULT_CONFIG.similarity_threshold,
                                label="Similarity Threshold",
                                info="Lower = more strict matching"
                            )
                            
                            max_context_docs = gr.Slider(
                                minimum=1,
                                maximum=10,
                                step=1,
                                value=DEFAULT_CONFIG.max_context_docs,
                                label="Max Context Documents",
                                info="Number of documents to include in context"
                            )
                            
                            top_k_retrieval = gr.Slider(
                                minimum=1,
                                maximum=20,
                                step=1,
                                value=DEFAULT_CONFIG.top_k_retrieval,
                                label="Top-K Retrieval",
                                info="Number of documents to retrieve from database"
                            )
                        
                        with gr.Accordion("🤖 Model Settings", open=True):
                            # Find the correct default values from the choices
                            default_chat = next((choice for choice in AVAILABLE_CHAT_MODELS if DEFAULT_CONFIG.chat_model in choice), AVAILABLE_CHAT_MODELS[0] if AVAILABLE_CHAT_MODELS else DEFAULT_CONFIG.chat_model)
                            default_embed = next((choice for choice in AVAILABLE_EMBEDDING_MODELS if DEFAULT_CONFIG.embedding_model in choice), AVAILABLE_EMBEDDING_MODELS[0] if AVAILABLE_EMBEDDING_MODELS else DEFAULT_CONFIG.embedding_model)
                            
                            with gr.Row():
                                chat_model = gr.Dropdown(
                                    choices=AVAILABLE_CHAT_MODELS,
                                    value=default_chat,
                                    label="Chat Model",
                                    info="✅ = Installed, 📦 = Needs 'ollama pull <model>'",
                                    interactive=True,
                                    scale=4
                                )
                                refresh_models_btn = gr.Button("🔄", size="sm", variant="secondary", scale=1, min_width=50)
                            
                            embedding_model = gr.Dropdown(
                                choices=AVAILABLE_EMBEDDING_MODELS,
                                value=default_embed,
                                label="Embedding Model",
                                info="✅ = Installed, 📦 = Needs 'ollama pull <model>'",
                                interactive=True
                            )
                            
                            use_rag = gr.Checkbox(
                                value=DEFAULT_CONFIG.use_rag,
                                label="Enable RAG",
                                info="Use document knowledge (vs pure LLM)"
                            )
                        
                        with gr.Accordion("📄 Text Processing", open=True):
                            chunk_size = gr.Dropdown(
                                choices=DEFAULT_CONFIG.CHUNK_SIZE_OPTIONS,
                                value=DEFAULT_CONFIG.chunk_size,
                                label="Chunk Size",
                                info="Size of text chunks (requires reload)"
                            )
                            
                            chunk_overlap = gr.Slider(
                                minimum=0,
                                maximum=200,
                                step=20,
                                value=DEFAULT_CONFIG.chunk_overlap,
                                label="Chunk Overlap",
                                info="Overlap between chunks (requires reload)"
                            )
                        
                        with gr.Accordion("🔧 Debug Settings", open=True):
                            enable_debug = gr.Checkbox(
                                value=DEFAULT_CONFIG.enable_debug,
                                label="Enable Debug Info",
                                info="Show detailed processing information"
                            )
                    
                    # Debug information display (outside the main parameter group)
                    debug_info = gr.Markdown(
                        label="Debug Information",
                        visible=True
                    )
            
            # Event handlers
            def submit_message(message, history, sim_thresh, max_docs, top_k, model, emb_model, rag_enabled, temp, top_p_val, max_tok, chunk_sz, chunk_ovlp, debug_enabled):
                return self.chat_with_rag(message, history, sim_thresh, max_docs, top_k, model, emb_model, rag_enabled, chunk_sz, chunk_ovlp, temp, top_p_val, max_tok, debug_enabled)
            
            # Send button click
            send_btn.click(
                fn=submit_message,
                inputs=[
                    msg, chatbot, similarity_threshold, max_context_docs, 
                    top_k_retrieval, chat_model, embedding_model, use_rag, 
                    temperature, top_p, max_tokens, chunk_size, chunk_overlap, enable_debug
                ],
                outputs=[msg, chatbot, debug_info]
            )
            
            # Enter key in textbox
            msg.submit(
                fn=submit_message,
                inputs=[
                    msg, chatbot, similarity_threshold, max_context_docs,
                    top_k_retrieval, chat_model, embedding_model, use_rag,
                    temperature, top_p, max_tokens, chunk_size, chunk_overlap, enable_debug
                ],
                outputs=[msg, chatbot, debug_info]
            )
            
            # Clear button
            clear_btn.click(
                fn=self.reset_chat,
                outputs=[chatbot, debug_info]
            )
            
            # Auto-adjust max_docs when top_k changes
            def adjust_max_docs(top_k_val, max_docs_val):
                return min(max_docs_val, top_k_val)
            
            top_k_retrieval.change(
                fn=adjust_max_docs,
                inputs=[top_k_retrieval, max_context_docs],
                outputs=[max_context_docs]
            )
            
            # Refresh models when dropdowns are clicked/focused
            def refresh_chat_models():
                chat_choices, _ = self.refresh_models()
                # Find current selection in new choices or default to first
                current_val = next((choice for choice in chat_choices if DEFAULT_CONFIG.chat_model in choice), 
                                  chat_choices[0] if chat_choices else DEFAULT_CONFIG.chat_model)
                return gr.Dropdown(choices=chat_choices, value=current_val)
            
            def refresh_embed_models():
                _, embed_choices = self.refresh_models()
                # Find current selection in new choices or default to first
                current_val = next((choice for choice in embed_choices if DEFAULT_CONFIG.embedding_model in choice), 
                                  embed_choices[0] if embed_choices else DEFAULT_CONFIG.embedding_model)
                return gr.Dropdown(choices=embed_choices, value=current_val)
            
            
            # Refresh models when button is clicked
            def refresh_all_models():
                chat_choices, embed_choices = self.refresh_models()
                # Find current selections in new choices or default to first
                current_chat = next((choice for choice in chat_choices if DEFAULT_CONFIG.chat_model in choice), 
                                   chat_choices[0] if chat_choices else DEFAULT_CONFIG.chat_model)
                current_embed = next((choice for choice in embed_choices if DEFAULT_CONFIG.embedding_model in choice), 
                                    embed_choices[0] if embed_choices else DEFAULT_CONFIG.embedding_model)
                return (
                    gr.Dropdown(choices=chat_choices, value=current_chat),
                    gr.Dropdown(choices=embed_choices, value=current_embed)
                )
            
            refresh_models_btn.click(
                fn=refresh_all_models,
                outputs=[chat_model, embedding_model]
            )
            
            # Footer
            gr.Markdown("""
            ---
            💡 **Tips:**
            - Lower similarity threshold = more strict document matching
            - Higher top-K = more documents to choose from
            - Enable debug info to see what's happening under the hood
            - Try different models for different response styles
            - **Note**: Chunk size/overlap changes require running `pipenv run python ./src/load_docs.py --reset` to rebuild the database
            """)
        
        return interface

def main():
    """Launch the Gradio interface."""
    rag_interface = RAGInterface()
    interface = rag_interface.create_interface()
    
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()