from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core import Settings
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor

# Custom prompt for medical emergency knowledge graph extraction
MEDICAL_EMERGENCY_KG_EXTRACT_TMPL = (
    "Extract up to {max_knowledge_triplets} medical knowledge triplets from the given emergency response text. "
    "Focus on procedures, symptoms, treatments, equipment usage, and safety protocols.\n"
    "Each triplet should be in the form of (head, relation, tail) with their respective types.\n"
    "---------------------\n"
    "MEDICAL EMERGENCY ONTOLOGY:\n"
    "Entity Types: {allowed_entity_types}\n"
    "Relation Types: {allowed_relation_types}\n"
    "\n"
    "MEDICAL-SPECIFIC GUIDELINES:\n"
    "- Prioritize actionable medical procedures and their relationships\n"
    "- Extract step sequences in emergency protocols (first, then, after, etc.)\n"
    "- Identify medical equipment and their specific usage contexts\n"
    "- Capture timing, measurements, and critical safety information\n"
    "- Link symptoms to appropriate responses\n"
    "- Connect contraindications and precautions to procedures\n"
    "\n"
    "OUTPUT FORMAT:\n"
    "- JSON format: [{{'head': '', 'head_type': '', 'relation': '', 'tail': '', 'tail_type': ''}}]\n"
    "- Use medical terminology when appropriate\n"
    "- Be specific about body parts, measurements, and timing\n"
    "- Maintain clinical accuracy\n"
    "---------------------\n"
    "MEDICAL EMERGENCY EXAMPLES:\n"
    "\n"
    "Text: 'Place your hands in the center of the chest and push down at least 2 inches at a rate of 100-120 compressions per minute.'\n"
    "Output:\n"
    "[{{'head': 'hands', 'head_type': 'BODY_PART', 'relation': 'positioned_on', 'tail': 'center of chest', 'tail_type': 'ANATOMICAL_LOCATION'}},\n"
    " {{'head': 'chest compressions', 'head_type': 'MEDICAL_PROCEDURE', 'relation': 'requires_depth', 'tail': '2 inches minimum', 'tail_type': 'MEASUREMENT'}},\n"
    " {{'head': 'chest compressions', 'head_type': 'MEDICAL_PROCEDURE', 'relation': 'performed_at_rate', 'tail': '100-120 per minute', 'tail_type': 'RATE'}}]\n"
    "\n"
    "Text: 'If the patient is unconscious and not breathing normally, immediately begin CPR and call emergency services.'\n"
    "Output:\n"
    "[{{'head': 'unconscious patient', 'head_type': 'MEDICAL_CONDITION', 'relation': 'indicates_need_for', 'tail': 'CPR', 'tail_type': 'MEDICAL_PROCEDURE'}},\n"
    " {{'head': 'abnormal breathing', 'head_type': 'SYMPTOM', 'relation': 'triggers', 'tail': 'emergency response', 'tail_type': 'PROTOCOL'}},\n"
    " {{'head': 'CPR', 'head_type': 'MEDICAL_PROCEDURE', 'relation': 'performed_simultaneously_with', 'tail': 'emergency services call', 'tail_type': 'COMMUNICATION_ACTION'}}]\n"
    "\n"
    "Text: 'Tilt the head back and lift the chin to open the airway before giving rescue breaths.'\n"
    "Output:\n"
    "[{{'head': 'head tilt', 'head_type': 'PHYSICAL_MANEUVER', 'relation': 'opens', 'tail': 'airway', 'tail_type': 'ANATOMICAL_STRUCTURE'}},\n"
    " {{'head': 'chin lift', 'head_type': 'PHYSICAL_MANEUVER', 'relation': 'facilitates', 'tail': 'airway opening', 'tail_type': 'PHYSIOLOGICAL_PROCESS'}},\n"
    " {{'head': 'airway opening', 'head_type': 'PHYSIOLOGICAL_PROCESS', 'relation': 'prerequisite_for', 'tail': 'rescue breaths', 'tail_type': 'MEDICAL_PROCEDURE'}}]\n"
    "---------------------\n"
    "Text: {text}\n"
    "Output:\n"
)

MEDICAL_EMERGENCY_KG_EXTRACT_PROMPT = PromptTemplate(
    MEDICAL_EMERGENCY_KG_EXTRACT_TMPL, 
    prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT
)

# Enhanced prompt with properties for more detailed extraction
MEDICAL_EMERGENCY_KG_EXTRACT_WITH_PROPS_TMPL = (
    "Extract up to {max_knowledge_triplets} detailed medical knowledge triplets from the given emergency response text. "
    "Include relevant properties for entities and relationships to capture clinical context.\n"
    "---------------------\n"
    "MEDICAL EMERGENCY ONTOLOGY:\n"
    "Entity Types: {allowed_entity_types}\n"
    "Entity Properties: {allowed_entity_properties}\n"
    "Relation Types: {allowed_relation_types}\n"
    "Relation Properties: {allowed_relation_properties}\n"
    "\n"
    "MEDICAL EXTRACTION FOCUS:\n"
    "- Emergency procedures with timing and sequence\n"
    "- Equipment specifications and usage contexts\n"
    "- Patient demographics and condition severity\n"
    "- Safety protocols and contraindications\n"
    "- Measurements, rates, and clinical thresholds\n"
    "- Professional roles and responsibilities\n"
    "\n"
    "PROPERTY GUIDELINES:\n"
    "- timing: immediate, after_30_compressions, continuous\n"
    "- severity: critical, moderate, minor\n"
    "- age_group: adult, child, infant\n"
    "- duration: 1_second, 2_minutes, until_help_arrives\n"
    "- frequency: per_minute, per_cycle, once\n"
    "- skill_level: basic_life_support, advanced_life_support\n"
    "- equipment_type: manual, automated, disposable\n"
    "- body_position: supine, recovery_position, sitting\n"
    "\n"
    "OUTPUT FORMAT:\n"
    "[{{'head': '', 'head_type': '', 'head_props': {{'property': 'value'}}, 'relation': '', 'relation_props': {{'property': 'value'}}, 'tail': '', 'tail_type': '', 'tail_props': {{'property': 'value'}}}}]\n"
    "---------------------\n"
    "DETAILED MEDICAL EXAMPLE:\n"
    "\n"
    "Text: 'For adults, perform chest compressions at a depth of at least 2 inches at 100-120 compressions per minute until advanced life support arrives.'\n"
    "Output:\n"
    "[{{'head': 'chest compressions', 'head_type': 'MEDICAL_PROCEDURE', 'head_props': {{'age_group': 'adult', 'skill_level': 'basic_life_support'}}, 'relation': 'requires_depth', 'relation_props': {{'measurement_type': 'minimum_threshold'}}, 'tail': '2 inches', 'tail_type': 'MEASUREMENT', 'tail_props': {{'unit': 'inches', 'measurement_context': 'compression_depth'}}}},\n"
    " {{'head': 'chest compressions', 'head_type': 'MEDICAL_PROCEDURE', 'head_props': {{'age_group': 'adult', 'technique': 'continuous'}}, 'relation': 'performed_at_rate', 'relation_props': {{'timing': 'continuous', 'measurement_type': 'frequency'}}, 'tail': '100-120 per minute', 'tail_type': 'RATE', 'tail_props': {{'unit': 'per_minute', 'range': '100-120'}}}},\n"
    " {{'head': 'chest compressions', 'head_type': 'MEDICAL_PROCEDURE', 'head_props': {{'skill_level': 'basic_life_support'}}, 'relation': 'continues_until', 'relation_props': {{'termination_condition': 'professional_takeover'}}, 'tail': 'advanced life support arrival', 'tail_type': 'EVENT', 'tail_props': {{'professional_level': 'advanced', 'event_type': 'medical_handover'}}}}]\n"
    "---------------------\n"
    "Text: {text}\n"
    "Output:\n"
)

MEDICAL_EMERGENCY_KG_EXTRACT_WITH_PROPS_PROMPT = PromptTemplate(
    MEDICAL_EMERGENCY_KG_EXTRACT_WITH_PROPS_TMPL, 
    prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT
)

# Custom extractor class using the medical-specific prompts

class MedicalEmergencyKGExtractor(DynamicLLMPathExtractor):
    """
    Specialized knowledge graph extractor for medical emergency response content.
    Uses medical-specific prompts and ontology for better extraction quality.
    """
    
    def __init__(self, llm=None, use_properties=True, **kwargs):
        # Define comprehensive medical emergency ontology
        medical_entity_types = [
            "MEDICAL_PROCEDURE", "MEDICAL_CONDITION", "SYMPTOM", "EQUIPMENT", 
            "BODY_PART", "ANATOMICAL_LOCATION", "ANATOMICAL_STRUCTURE",
            "MEASUREMENT", "RATE", "DURATION", "PERSON", "PROFESSIONAL_ROLE",
            "MEDICATION", "TREATMENT", "PROTOCOL", "EMERGENCY_SERVICE",
            "PHYSIOLOGICAL_PROCESS", "PHYSICAL_MANEUVER", "COMMUNICATION_ACTION",
            "EVENT", "LOCATION", "AGE_GROUP", "SEVERITY_LEVEL"
        ]
        
        medical_relation_types = [
            "requires", "prerequisite_for", "follows_after", "performed_by", 
            "positioned_on", "measured_by", "performed_at_rate", "requires_depth",
            "indicates_need_for", "triggers", "treats", "prevents", "causes",
            "contraindicated_for", "performed_simultaneously_with", "opens",
            "facilitates", "continues_until", "replaces", "assists_with",
            "monitors", "detects", "administers", "applies_to", "used_for",
            "located_at", "connects_to", "activates", "deactivates"
        ]
        
        if use_properties:
            entity_properties = [
                ("age_group", "Target age demographic (adult, child, infant)"),
                ("severity", "Severity level (critical, moderate, minor)"),
                ("timing", "When performed (immediate, after_compressions, continuous)"),
                ("duration", "How long (seconds, minutes, until_condition_met)"),
                ("skill_level", "Required training (basic_life_support, advanced)"),
                ("equipment_type", "Equipment category (manual, automated, disposable)"),
                ("body_position", "Patient position (supine, recovery, sitting)"),
                ("measurement_unit", "Unit of measurement (inches, per_minute, seconds)")
            ]
            
            relation_properties = [
                ("sequence_order", "Step number in procedure sequence"),
                ("timing", "When relationship applies (before, during, after)"),
                ("conditions", "Conditions when relationship is valid"),
                ("measurement_type", "Type of measurement (minimum, maximum, range)"),
                ("professional_level", "Level of medical professional required"),
                ("equipment_required", "Equipment needed for this relationship")
            ]
            
            extract_prompt = MEDICAL_EMERGENCY_KG_EXTRACT_WITH_PROPS_PROMPT
        else:
            entity_properties = None
            relation_properties = None
            extract_prompt = MEDICAL_EMERGENCY_KG_EXTRACT_PROMPT
        
        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt,
            allowed_entity_types=medical_entity_types,
            allowed_relation_types=medical_relation_types,
            allowed_entity_props=entity_properties,
            allowed_relation_props=relation_properties,
            **kwargs
        )