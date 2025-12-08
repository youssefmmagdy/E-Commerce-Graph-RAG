from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import ChatHuggingFace
from langchain.llms.base import LLM
from typing import Optional, List, Any
from pydantic import BaseModel, Field
from pypdf import PdfReader
from huggingface_hub import InferenceClient

HF_TOKEN = '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'

pdf_path = "Checklist.pdf"

reader = PdfReader(pdf_path)
all_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        all_text += text + "\n"


# Custom LLM wrapper for HuggingFace Inference Client (Gemma conversational)
class GemmaLangChainWrapper(LLM):
    client: Any = Field(...)
    max_tokens: int = 500 #sets a default max output length

    @property
    def _llm_type(self) -> str:
        return "gemma_hf_api" #Identify the LLM type

    #what LangChain calls when it needs the LLM to answer something
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat_completion( #call the HuggingFace API
            messages=[{"role": "user", "content": prompt}],  #Wrap the plain text prompt into chat format because Gemma ONLY understands chat messages.
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message["content"]
client = InferenceClient(
    model="google/gemma-2-2b-it",
    token=HF_TOKEN
)

# Instantiate the wrapper
gemma_llm = GemmaLangChainWrapper(client=client)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100
)
documents = splitter.create_documents([all_text])

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = LCFAISS.from_documents(
    documents=documents,
    embedding=embedding_model
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

qa_chain = RetrievalQA.from_chain_type(
    llm=gemma_llm,
    retriever=retriever,
    chain_type="stuff" #concatenate all retrieved documents and feed them to the LLM as one big prompt.
)

response = qa_chain.run("What should be included in report Milestone 1?")
print(response)