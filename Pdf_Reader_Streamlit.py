import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
import tempfile

st.set_page_config(page_title="AI PDF Chatbot", layout="wide")

st.title("Chat with Your PDF")

# Sidebar for settings
st.sidebar.header("Settings")

api_key = st.sidebar.text_input("Your Gemini API Key", type="password")
chunk_size = st.sidebar.number_input("Chunk Size", min_value=100, max_value=2000, value=1000, step=100)
chunk_overlap = st.sidebar.number_input("Chunk Overlap", min_value=0, max_value=1000, value=400, step=100)
collection_name = st.sidebar.text_input("Qdrant Collection Name (New or Existing)", value="learning_vectors")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

embedding_model = None
vector_store = None
vector_db = None
chat_model = None

if api_key:
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )

    try:
        vector_db = QdrantVectorStore.from_existing_collection(
            url="https://e59dfb81-bb98-4eb6-9806-f172c977a89f.us-east-1-0.aws.cloud.qdrant.io:6333",
            api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.EhU0hmKfZ9p-LYubvLHcF7aQg-piIYam2L1qK7CdAnE",
            collection_name=collection_name,
            embedding=embedding_model
        )
        st.success(f"Connected to existing vector store: {collection_name}")
    except Exception as e:
        st.warning("Collection not found or failed to connect. Please upload a PDF to create it.")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        split_docs = text_splitter.split_documents(documents=docs)

        vector_store = QdrantVectorStore.from_documents(
            documents=split_docs,
            url="https://e59dfb81-bb98-4eb6-9806-f172c977a89f.us-east-1-0.aws.cloud.qdrant.io:6333",
            api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.EhU0hmKfZ9p-LYubvLHcF7aQg-piIYam2L1qK7CdAnE",
            collection_name=collection_name,
            embedding=embedding_model,
            force_recreate=False
        )

        st.success(f"File indexed and added to vector store: {collection_name}")
        vector_db = vector_store

    if vector_db:
        st.markdown("---")
        st.subheader("Chat with your PDF")

        genai.configure(api_key=api_key)

        user_input = st.chat_input("Ask a question")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                st.markdown(msg["parts"][0])

        if user_input:

            with st.chat_message("user"):
                st.markdown(user_input)

            search_results = vector_db.similarity_search(query=user_input)

            context = "\n\n\n".join([
                f"Page Content: {result.page_content}\nPage Number: {result.metadata.get('page_label', 'N/A')}\nFile Location: {result.metadata.get('source', 'Uploaded')}"
                for result in search_results
            ])

            SYSTEM_PROMPT = f"""
                You are a helpfull AI Assistant who answers user query based on the available context
    retrieved from a PDF file along with page_contents and page number.
    You should only ans the user based on the following context and navigate the user
    to open the right page number to know more. Give useful answer fom the context and strictly the page number.

                Context:
                {context}
            """
            chat_model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config={"temperature": 0},
                system_instruction=SYSTEM_PROMPT
            )

            chat = chat_model.start_chat(history=st.session_state.chat_history)

            # print(SYSTEM_PROMPT)

            response = chat.send_message(user_input)

            with st.chat_message("assistant"):
                st.markdown(response.text)

            # print(response.text)

            st.session_state.chat_history.append({"role": "user", "parts": [user_input]})
            st.session_state.chat_history.append({"role": "model", "parts": [response.text]})
else:
    st.info("Please enter your Gemini API key to begin.")
