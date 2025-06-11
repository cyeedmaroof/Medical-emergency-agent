from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core import StorageContext, load_index_from_storage
from src.PGExtractor import knowledge_graph_extractor
from llama_index.core import Settings
from src.medical_kg_prompt import MedicalEmergencyKGExtractor
from llama_index.core.graph_stores import SimplePropertyGraphStore
import nest_asyncio
nest_asyncio.apply()

async def knowledge_graph_construction(nodes=None, llm=None, extractor = None, load_persist=None, persist_path=None, max_triplets_per_chunk=20):
    """Build complete knowledge graph construction pipeline."""
    
    #Check persistence path
    if load_persist:
        print(f"🔄 Loading persisted knowledge graph from {load_persist}")
        try:
            storage_context = StorageContext.from_defaults(persist_dir=load_persist)
            kg_index = load_index_from_storage(storage_context)
            print("✅ Successfully loaded persisted knowledge graph!")
            return kg_index
        except Exception as e:
            print(f"❌ Error loading persisted knowledge graph: {e}")
            return None, None
    
    if not nodes:
        print("❌ No nodes available for knowledge graph construction")
        print("Please provide a list of nodes to build the knowledge graph.")
        return None, None
    
    print(f"📄 {len(nodes)} medical emergency nodes found")
    
    # Setup the extractor
    if extractor == "Custom": 
        print("🔧 Using custom extractor configuration")
        extractor = MedicalEmergencyKGExtractor(
        llm=llm,
        use_properties=True,  
        max_triplets_per_chunk=max_triplets_per_chunk 
        )

    else:
        extractor = knowledge_graph_extractor(llm=llm, max_triplets_per_chunk=max_triplets_per_chunk)

    # Create PropertyGraphIndex
    try:
        kg_index = PropertyGraphIndex(
            nodes=nodes,
            llm=llm or Settings.llm,  # Use default LLM if not provided
            embed_model=Settings.embed_model,  # Use default embedding model
            kg_extractors=[extractor],
            embed_kg_nodes=True,
            use_async=True,
            show_progress=True
        )
        print("✅ Successfully created PropertyGraphIndex!") 

        if persist_path:
            kg_index.storage_context.persist(persist_dir=persist_path)
            print(f"✅ Property graph store persisted to {persist_path}")
        
        return kg_index
        
        
    except Exception as e:
        print(f"❌ Error creating knowledge graph: {e}")
        return None, None
    


    