from rag_query import query_rag_simple as query_rag
import streamlit as st

    
st.set_page_config(page_title="Local Rag")
with st.sidebar:
    st.title('Local Rag')

# Function for generating LLM response
def generate_response(input):
    result = query_rag(input)
    return result

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! I can answer questions using both your uploaded documents and my general knowledge. Ask me anything!"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Querying LLM..."):
            response = generate_response(input) 
            
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)