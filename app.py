import streamlit as st
# Ensure the import path matches your folder structure. 
# If helper.py is in the same folder, use: from helper import ...
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process a PDF first.")
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Display the conversation
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"**User:** {message.content}")
        else:
            st.write(f"**Bot:** {message.content}")

def main():
    st.set_page_config(page_title="Information Retrieval", page_icon="🦜")
    st.header("Information-Retrieval-System 🦜")

    # Initialize session state variables if they don't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # User Input Area
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        handle_userinput(user_question)

    # Sidebar for File Upload
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if pdf_docs:
                    # 1. Get PDF Text
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # 2. Get Text Chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    # 3. Create Vector Store
                    vector_store = get_vector_store(text_chunks)
                    
                    # 4. Create Conversation Chain
                    st.session_state.conversation = get_conversational_chain(vector_store)

                    st.success("Done! You can now ask questions.")
                else:
                    st.error("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()