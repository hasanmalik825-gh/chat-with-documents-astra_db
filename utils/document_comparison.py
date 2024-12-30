import hashlib
from typing import List
from langchain_community.vectorstores import VectorStore
from langchain_core.documents import Document

def _get_string_hash(input_string: str) -> str:
    sha256_hash = hashlib.sha256()
    # Convert the string to bytes and update the hash
    sha256_hash.update(input_string.encode('utf-8'))
    return sha256_hash.hexdigest()

def add_unique_documents(documents: List[Document], vector_store: VectorStore):
    doc_ids = [_get_string_hash(doc.page_content) for doc in documents]
    documents = [(doc, doc_ids[idx]) for idx, doc in enumerate(documents) if vector_store.get_by_document_id(doc_ids[idx]) is None]
    if len(documents) > 0:
        vector_store.add_documents(documents=[doc[0] for doc in documents], ids=[doc[1] for doc in documents])
        print(f"Added {len(documents)} unique documents to the vector store.")
    else:
        print("All documents already exist in the vector store.")