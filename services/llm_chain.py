from langchain_core.runnables.base import Runnable
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.vectorstores import VectorStore


def inference_chain_rag(
        llm : BaseLanguageModel, 
        vectorstorage : VectorStore, 
        prompt_template : PromptTemplate, 
        k=3
    ) -> Runnable:
    """
    This function is used to create RetrievalQ with "stuff" type and also take retriever for rag.
    args:
        llm: langchain llm
        vectorstorage: vector store
        prompt_template: prompt template
        output_parser: output parser
        return_source_documents: return source documents
        k: number of documents to return
    """

    # Create a retriever from the vector store
    retriever = vectorstorage.as_retriever(search_kwargs={'k': k})
    # Create the chain with the specified parameters
    qa_chain = create_stuff_documents_chain(
        llm=llm, 
        prompt=prompt_template,
    )
    retrieval_chain = create_retrieval_chain(retriever, qa_chain)
    return retrieval_chain