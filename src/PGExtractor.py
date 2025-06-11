from llama_index.core.indices.property_graph import DynamicLLMPathExtractor
from llama_index.core import Settings

def knowledge_graph_extractor(llm=None, max_triplets_per_chunk=20):
    """Configure the DynamicLLMPathExtractor for medical emergency data."""
    
    # Define medical emergency ontology
    allowed_entity_types = [
        "Person",           # Patient, Rescuer, Medical Professional
        "MedicalCondition", # Unconscious, Cardiac Arrest, Breathing Issues
        "MedicalProcedure", # CPR, Rescue Breathing, AED Usage
        "Equipment",        # AED, Electrode Pads, Medical Supplies
        "BodyPart",         # Chest, Airway, Heart
        "Measurement",      # Rate, Depth, Duration, Ratio
        "Location",         # Hospital, Emergency Scene
        "Organization"      # Emergency Services, Medical Team
    ]
    
    allowed_relation_types = [
        "requires",         # CPR requires chest compressions
        "follows",          # Step A follows Step B
        "treats",           # CPR treats cardiac arrest
        "uses",             # Rescuer uses AED
        "positioned_on",    # Hands positioned on chest
        "causes",           # Compressions cause blood circulation
        "prevents",         # CPR prevents brain damage
        "indicates",        # No pulse indicates cardiac arrest
        "measured_by",      # Compression depth measured by inches
        "performed_by",     # CPR performed by rescuer
        "contraindicated_for" # AED contraindicated for pacemaker patients
    ]
    
    # Define properties for entities and relations
    entity_properties = [
        ("age_group", "Target age demographic (adult, child, infant)"),
        ("skill_level", "Required skill level (basic, advanced)"),
        ("duration", "Time duration for procedure or measurement"),
        ("frequency", "How often something occurs (per minute, per cycle)"),
        ("severity", "Severity level of condition or procedure")
    ]
    
    relation_properties = [
        ("sequence_order", "Order in a sequence of steps"),
        ("confidence_level", "Medical confidence level for relationship"),
        ("timing", "When this relationship applies"),
        ("conditions", "Conditions under which relationship is valid")
    ]
    
    # Initialize the extractor
    extractor = DynamicLLMPathExtractor(
        llm=llm or Settings.llm,  # Make sure to set your LLM in Settings
        allowed_entity_types=allowed_entity_types,
        allowed_relation_types=allowed_relation_types,
        allowed_entity_props=entity_properties,
        allowed_relation_props=relation_properties,
        max_triplets_per_chunk=max_triplets_per_chunk  # Increase for medical procedures
    )
    
    return extractor