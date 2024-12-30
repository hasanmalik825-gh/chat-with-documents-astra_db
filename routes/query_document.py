from fastapi import APIRouter, Query, File, UploadFile
from services.rag import load_document, split_document, embedder_by_huggingface, get_astra_db_vector_store
from utils.document_comparison import add_unique_documents
from services.llm_chain import inference_chain_rag
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

query_document_router = APIRouter()


@query_document_router.post("/query_document")
async def query_document(
    query: str = Query(..., description="Query for the document"),
    file: UploadFile = File(..., description="File to be queried"),
):
    documents = load_document(file)
    documents = split_document(documents)
    embeddings = embedder_by_huggingface()
    vector_store = get_astra_db_vector_store(collection_name="astra_vectordb_langchain", embeddings=embeddings)
    add_unique_documents(documents, vector_store)
    template = [
        ("system", "You are a helpful assistant that answers concisely. You are given the following context: {context}."),
        ("human", "{input}"),
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages=template)
    llm = ChatGroq(model="llama3-8b-8192")
    chain = inference_chain_rag(
        vectorstorage=vector_store, 
        llm=llm,
        prompt_template=prompt_template,
    )
    response = chain.invoke({"input": query})
    return {"chain_response": response["answer"]}
