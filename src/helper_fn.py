from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
import faiss

def create_faiss_vector_store_and_context(nodes=None, persist_path=None, load_persist=None):
    """
    Initializes a Faiss vector store and a storage context.

    Args:
        nodes: A list of nodes, where the first node has an 'embedding' attribute 
               to determine the dimension.

    Returns:
        A tuple containing (vector_store, storage_context).
    """
    if not nodes or not hasattr(nodes[0], 'embedding') or nodes[0].embedding is None:
        raise ValueError("Nodes list must not be empty and the first node must have a valid embedding.")
    
    # dimensions of BAAI/bge-small-en-v1.5 is 384, dynamically determined
    dim = len(nodes[0].embedding)
    faiss_index = faiss.IndexFlatL2(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    
    if persist_path:
        vector_store.persist(persist_path=persist_path)
        print(f"✅ Vector store persisted to {persist_path}")
    elif load_persist:
        try:
            vector_store = FaissVectorStore.from_persist_path(load_persist)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            print(f"🔄 Successfully loaded persisted vector store from {load_persist}")
            return vector_store, storage_context
        except Exception as e:
            print(f"❌ Error loading persisted vector store: {e}")
            return None, None
    storage_context = StorageContext.from_defaults(vector_store=vector_store)


    return vector_store, storage_context