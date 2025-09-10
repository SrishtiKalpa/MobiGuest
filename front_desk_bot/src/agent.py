import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

def handle_chat_input(prompt, rag_chain):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Prepare chat history for the RAG chain
        langchain_chat_history = []
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                langchain_chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_chat_history.append(AIMessage(content=msg["content"]))

        # Stream response from RAG chain
        response_generator = rag_chain.stream(
            {"input": prompt, "chat_history": langchain_chat_history}
        )
        full_response = st.write_stream(response_generator)
        
        # Append assistant's full response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
