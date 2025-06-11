from src.helper_fn import create_faiss_vector_store_and_context
from llama_index.core import  VectorStoreIndex
from src.dynamicKG_builder import knowledge_graph_construction
import asyncio
from llama_index.core import Settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_retriever(
    question,
    nodes=None,
    max_nodes=None,
    type="hybrid",
    llm=None,
    extractor=None,
    load_persist=None,
    persist_path=None,
    max_triplets_per_chunk=20,
    similarity_top_k=15 
):
    """
    Create a retriever index based on the specified type.

    Args:
        nodes (list): List of nodes to use for building the retriever.
        max_nodes (int, optional): Maximum number of nodes for the vector store (used in hybrid mode). Defaults to 100.
        type (str, optional): Type of retriever to create: "vector_store", "knowledge_graph", or "hybrid". Defaults to "vector_store".
        llm (object, optional): Language model for knowledge graph construction. Defaults to None.
        extractor (object, optional): Extractor for knowledge graph construction. Defaults to None.
        load_persist (bool, optional): Whether to load persisted knowledge graph data. Defaults to None.
        persist_path (str, optional): Path for persisting or loading knowledge graph data. Defaults to None.
        max_triplets_per_chunk (int, optional): Max triplets per chunk for knowledge graph construction. Defaults to 20.

    Returns:
        If type is "vector_store":
            tuple: (VectorStoreIndex, vector_store, storage_context)
        If type is "knowledge_graph":
            kg_index
        If type is "hybrid":
            tuple: (VectorStoreIndex, kg_index, vector_store, storage_context)
    """

    if max_nodes:
        nodes = nodes[:max_nodes]
    if type == "vector_store":
        # Create a FAISS vector store and storage context
        vector_store, storage_context = create_faiss_vector_store_and_context(nodes)
        vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
        vec_rec = vector_index.as_retriever(
            similarity_top_k=similarity_top_k,
            show_progress=True,
            include_text=True
        )
        vec_docs = vec_rec.retrieve(question)
        # Adding metadata to distinguish source
        for doc in vec_docs:
            doc.metadata['source'] = 'vector_store'
        

        return vec_docs
    
    elif type == "knowledge_graph":
        # Create a knowledge graph index
        kg_index = asyncio.run(
            knowledge_graph_construction(
                nodes,
                llm=llm,
                extractor=extractor,
                load_persist=load_persist,
                persist_path=persist_path,
                max_triplets_per_chunk=max_triplets_per_chunk,
            )
        )
        kg_ret = kg_index.as_retriever(
            similarity_top_k=similarity_top_k, 
            show_progress=True,
            include_text=True
        )

        kg_docs = kg_ret.retrieve(question)
        # Adding metadata to distinguish source
        for doc in kg_docs:
            doc.metadata['source'] = 'knowledge_graph'
        return kg_docs
    
    elif type == "hybrid":
        # Create both vector store and knowledge graph index
        vector_store, storage_context = create_faiss_vector_store_and_context(nodes)
        vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

        kg_index = asyncio.run(
            knowledge_graph_construction(
                nodes,
                llm=llm,
                extractor=extractor,
                load_persist=load_persist,
                persist_path=persist_path,
                max_triplets_per_chunk=max_triplets_per_chunk,
            )
        )
        vec_rec = vector_index.as_retriever(
            similarity_top_k=similarity_top_k,  
            show_progress=True,
            include_text=True
        )
        kg_rec = kg_index.as_retriever(
            similarity_top_k=similarity_top_k,
            show_progress=True,
            include_text=True
        )
        vec_docs = vec_rec.retrieve(question)
        kg_docs = kg_rec.retrieve(question)

        print(f"Vector Store Retrieved {len(vec_docs)} documents")
        print(f"Knowledge Graph Retrieved {len(kg_docs)} documents")

        # # use TfidfVectorizer to compute similarity scores
        # # if the cosine similarity is equal to 1, it means the documents are identical
        # # so we can filter them out
        # vectorizer = TfidfVectorizer()
        # vec_texts = [doc.text for doc in vec_docs]
        # kg_texts = [doc.text for doc in kg_docs]
        # vec_tfidf = vectorizer.fit_transform(vec_texts)
        # kg_tfidf = vectorizer.transform(kg_texts)
        # similarity_matrix = cosine_similarity(vec_tfidf, kg_tfidf)
        # # Filter out documents with cosine similarity of 1
        # vec_docs = [doc for i, doc in enumerate(vec_docs) if not any(similarity_matrix[i, j] >= 1 for j in range(len(kg_docs)))]
        # kg_docs = [doc for i, doc in enumerate(kg_docs) if not any(similarity_matrix[j, i] >= 1 for j in range(len(vec_docs)))]

        # Adding metadata to distinguish sources
        for doc in vec_docs:
            doc.metadata['source'] = 'vector'
        for doc in kg_docs:
            doc.metadata['source'] = 'knowledge_graph'

        # Combine results from both retrievers
        combined_docs = vec_docs + kg_docs

        return combined_docs
    

    
