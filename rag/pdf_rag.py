import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_knowledge_base(pdf_paths):
    all_docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        for d in docs:
            d.metadata["source_file"] = os.path.basename(pdf_path)
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH,
    )
    return len(all_docs), len(chunks)

def get_retriever():
    if not os.path.exists(DB_PATH):
        return None
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    return store.as_retriever(search_kwargs={"k": 5})

def retrieve_context(question):
    retriever = get_retriever()
    if retriever is None:
        return [], "No PDF knowledge base has been built yet."
    docs = retriever.invoke(question)
    context = "\n\n".join(
        f"[Source: {d.metadata.get('source_file','PDF')}, page {d.metadata.get('page','?')}]\n{d.page_content}"
        for d in docs
    )
    return docs, context
