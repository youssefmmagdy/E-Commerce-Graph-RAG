import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import Optional, List, Any
from pydantic import BaseModel, Field
from pypdf import PdfReader
from huggingface_hub import InferenceClient

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 RAG Chatbot with LangChain & HuggingFace")
st.markdown("Ask questions about your documents using Retrieval Augmented Generation")

# Initialize session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    hf_token = st.text_input(
        "HuggingFace API Token",
        type="password",
        help="Your HuggingFace API token"
    )
    
    pdf_file = st.file_uploader(
        "Upload PDF file",
        type="pdf",
        help="Choose a PDF document to analyze"
    )
    
    chunk_size = st.slider(
        "Chunk Size",
        min_value=100,
        max_value=1000,
        value=400,
        step=100,
        help="Size of text chunks for embedding"
    )
    
    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=200,
        value=100,
        step=10,
        help="Overlap between chunks"
    )
    
    max_tokens = st.slider(
        "Max Output Tokens",
        min_value=100,
        max_value=1000,
        value=500,
        step=100,
        help="Maximum tokens in LLM response"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="Controls randomness of responses"
    )
    
    st.divider()
    st.markdown("### About")
    st.markdown(
        """
        This app uses:
        - **LangChain** for RAG orchestration
        - **HuggingFace Inference API** (Gemma 2B)
        - **FAISS** for vector similarity search
        - **Sentence Transformers** for embeddings
        """
    )


# Custom LLM wrapper for HuggingFace Inference Client
class GemmaLangChainWrapper(LLM):
    client: Any = Field(...)
    max_tokens: int = 500
    temperature: float = 0.2

    @property
    def _llm_type(self) -> str:
        return "gemma_hf_api"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message["content"]


# Main content
col1, col2 = st.columns([2, 1])

with col2:
    if st.button("🔄 Initialize RAG Chain", use_container_width=True):
        if not hf_token:
            st.error("❌ Please provide HuggingFace API token")
        elif pdf_file is None:
            st.error("❌ Please upload a PDF file")
        else:
            with st.spinner("Processing document..."):
                try:
                    # Save uploaded file temporarily
                    import tempfile
                    import os
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(pdf_file.read())
                        tmp_path = tmp_file.name
                    
                    # Extract text from PDF
                    reader = PdfReader(tmp_path)
                    all_text = ""
                    
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            all_text += text + "\n"
                    
                    if not all_text.strip():
                        st.error("❌ No text could be extracted from the PDF")
                    else:
                        # Create text splitter
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        documents = splitter.create_documents([all_text])
                        
                        # Create embeddings
                        embedding_model = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2"
                        )
                        
                        # Create vector store
                        vectorstore = LCFAISS.from_documents(
                            documents=documents,
                            embedding=embedding_model
                        )
                        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
                        
                        # Initialize LLM
                        client = InferenceClient(
                            model="google/gemma-2-2b-it",
                            token=hf_token
                        )
                        gemma_llm = GemmaLangChainWrapper(
                            client=client,
                            max_tokens=max_tokens,
                            temperature=temperature
                        )
                        
                        # Create QA chain
                        st.session_state.qa_chain = RetrievalQA.from_chain_type(
                            llm=gemma_llm,
                            retriever=retriever,
                            chain_type="stuff"
                        )
                        
                        # Clean up
                        os.unlink(tmp_path)
                        
                        st.success("✅ RAG Chain initialized successfully!")
                        st.session_state.chat_history = []
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Chat interface
with col1:
    st.subheader("💬 Chat with Your Document")
    
    if st.session_state.qa_chain is not None:
        # Display chat history
        chat_container = st.container(height=400, border=True)
        
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.chat_message("user").write(message["content"])
                else:
                    st.chat_message("assistant").write(message["content"])
        
        # User input
        user_input = st.chat_input("Ask a question about your document...")
        
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.qa_chain.run(user_input)
                    
                    # Add assistant response to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
    else:
        st.info("👈 Please initialize the RAG chain first using the button on the right")


# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center'>
    <small>Built with ❤️ using LangChain, Streamlit, and HuggingFace</small>
    </div>
    """,
    unsafe_allow_html=True
)
