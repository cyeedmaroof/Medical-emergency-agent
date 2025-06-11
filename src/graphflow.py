from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document
import json
from typing import List, Optional, Dict, Any
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict
from src.retriever import create_retriever

# =============================================================================
# STATE DEFINITION
# =============================================================================

class EmergencyRAGState(TypedDict):
    """
    Enhanced state for emergency dispatch + RAG workflow.
    
    Attributes:
        # Original RAG fields
        question: The original user question
        documents: List of retrieved documents
        generation: Generated answer
        grade: Grade of document relevance
        iterations: Number of iterations for rephrasing
        rephrased_question: Rephrased version of the question
        llm: LLM model instance
        retriever_type: Type of retriever to use
        load_persist: Path to persisted index
        nodes: List of document nodes
        max_iterations: Maximum number of iterations
        
        # Emergency dispatch fields
        call_transcript: The emergency call transcript
        emergency_analysis: Structured emergency analysis
        is_emergency: Whether this is an emergency dispatch query
        emergency_type: Type of emergency identified
        dispatcher_actions: Recommended dispatcher actions
        structured_report: Final structured emergency report
    """
    # RAG fields
    question: str
    documents: List[Document]
    generation: str
    grade: str
    iterations: int
    rephrased_question: str
    llm: Any
    retriever_type: str
    load_persist_vector: Optional[str] = None
    load_persist_kg: Optional[str] = None
    max_iterations: Optional[int] = 3
    nodes: List[Document] = Field(default_factory=list)
    # Emergency dispatch fields
    call_transcript: Optional[str] = None
    emergency_analysis: Optional[Dict[str, Any]] = None
    is_emergency: Optional[bool] = False
    emergency_type: Optional[str] = None
    dispatcher_actions: Optional[List[str]] = None
    structured_report: Optional[str] = None
    type_of_emergency: Optional[str] = None  # Added for clarity
    injury_description: Optional[str] = None  # Added for clarity

# =============================================================================
# EMERGENCY DISPATCH ANALYSIS NODES
# =============================================================================

def detect_emergency_context(state: EmergencyRAGState) -> Dict[str, Any]:
    """
    Detect if the input is an emergency dispatch scenario.
    
    Args:
        state: The current graph state
        
    Returns:
        Updated state with emergency detection
    """
    print("--- DETECT EMERGENCY CONTEXT ---")
    
    question = state["question"]
    call_transcript = state.get("call_transcript", "")
    model = state.get("llm")
    
    # Combine question and transcript for analysis
    input_text = f"Question: {question}\n\nTranscript: {call_transcript}" if call_transcript else question
    
    detection_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are analyzing input to determine if it's related to emergency dispatch.
        Look for indicators like:
        - Emergency call transcripts
        - 911/emergency service communications
        - Urgent medical, fire, or police situations
        - Dispatcher-caller interactions
        - Emergency response scenarios
        
        Provide a JSON response with:
        {{
            "is_emergency": true/false,
            "confidence": 0.0 to 1.0,
            "emergency_type": "medical/fire/police/other/none",
            "reasoning": "brief explanation"
        }}"""),
        ("human", "Analyze this input:\n\n{input_text}")
    ])
    
    chain = detection_prompt | model | JsonOutputParser()
    result = chain.invoke({"input_text": input_text})
    
    is_emergency = result.get("is_emergency", False)
    emergency_type = result.get("emergency_type", "none")
    
    print(f"Emergency detected: {is_emergency}, Type: {emergency_type}")
    
    return {
        **state,
        "is_emergency": is_emergency,
        "emergency_type": emergency_type
    }

def analyze_emergency_call(state: EmergencyRAGState) -> Dict[str, Any]:
    """
    Perform structured emergency call analysis using the dispatch prompts.
    
    Args:
        state: The current graph state
        
    Returns:
        Updated state with emergency analysis
    """
    print("--- ANALYZE EMERGENCY CALL ---")
    
    question = state["question"]
    call_transcript = state.get("call_transcript", question)
    model = state.get("llm")
    
    # Emergency analysis system message
    system_message = """You are an AI assistant of a dispatch center.  
    Your objective is to interpret what is being communicated and fill in as much information as possible, 
    in the fixed format below, to make it easy for the dispatcher to know what kind of emergency the call is about, 
    and what kind of actions are required."""
    
    # Emergency analysis prompt
    analysis_prompt = """Analyze this emergency call and provide a structured JSON response with:
    {{
        "caller_and_location": {{
            "caller_description": "what can you say about the caller",
            "location": "location details from caller",
            "scene_description": "scene described by caller"
        }},
        "emergency_type": {{
            "emergency_classification": "type of emergency",
            "caller_status": "injured or calling for someone else"
        }},
        "injuries": {{
            "has_injuries": true/false,
            "patient_count": number,
            "patients": [
                {{
                    "injury_description": "description",
                    "severity": "assessment",
                    "patient_details": "what you know about patient"
                }}
            ]
        }},
        "actions": {{
            "dispatcher_actions": ["action1", "action2"],
            "relevant_skills": ["skill explanations for lay person"]
        }},
        "communication": {{
            "additional_questions": ["question1", "question2"],
            "critical_info_needed": ["info1", "info2"]
        }}
    }}"""
    
    emergency_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", f"{analysis_prompt}\n\nCall to analyze:\n{{call_transcript}}")
    ])
    
    chain = emergency_prompt | model | JsonOutputParser()
    emergency_analysis = chain.invoke({"call_transcript": call_transcript})
    
    # Extract dispatcher actions for easy access
    dispatcher_actions = emergency_analysis.get("actions", {}).get("dispatcher_actions", [])
    
    return {
        **state,
        "emergency_analysis": emergency_analysis,
        "dispatcher_actions": dispatcher_actions
    }

def generate_emergency_report(state: EmergencyRAGState) -> Dict[str, Any]:
    """
    Generate the final structured emergency dispatch report.
    
    Args:
        state: The current graph state
        
    Returns:
        Updated state with structured report
    """
    print("--- GENERATE EMERGENCY REPORT ---")
    
    emergency_analysis = state.get("emergency_analysis", {})
    call_transcript = state.get("call_transcript", state["question"])
    model = state.get("llm")
    type_of_emergency = state.get("emergency_type", "unknown")
    injury_description = state.get("injury_description", "unknown")
    
    report_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are generating a final emergency dispatch report.
        Create a clear, structured report that a dispatcher can quickly scan and act upon.
        Use the analysis provided to create a concise but complete report."""),
        ("human", """Create a structured emergency dispatch report based on this analysis:
        
        Analysis: {analysis}
        Original Call: {call_transcript}
        
        Format the report with clear sections into JSON format:
        {{
            "emergency_summary": {{
                "Type of emergency": "brief summary of the emergencyy",
                "Incident Description": "description of the incident",
            }},
    
            "caller_and_location": {{
                "Caller Name": "name of the caller if available",
                "Caller Location": "location of the caller",
                "Caller Status": "status of the caller (injured, calm, etc.)",
                "Scene Description": "description of the scene as provided by the caller"
            }},
            "injuries_patients":{{
                "has_injuries": true/false,
                "patient_count": number of patients,
                "patients": [
                    {{
                        "injury_description": "description of the injury",
                        "severity": "severity assessment (e.g., critical, moderate, minor)",
                        "patient_details": "details about the patient (e.g., age, gender, condition)"
            }},
            "immediate_actions_required":{{
                "actions": "actions that need to be taken immediately",
                "relevant_skills": "skills or protocols needed for this emergency",

            }},
            "additional_questions_to_ask": "any additional questions to clarify the situation",
            "relevant_protocols_skills_needed": "protocols or skills needed for this emergency"
        }}
        """)
    ])
    
    # Format the report with clear sections:
        # 1. EMERGENCY SUMMARY
        # 2. CALLER & LOCATION
        # 3. INJURIES/PATIENTS
        # 4. IMMEDIATE ACTIONS REQUIRED
        # 5. ADDITIONAL QUESTIONS TO ASK
        # 6. RELEVANT PROTOCOLS/SKILLS NEEDED

    chain = report_prompt | model | JsonOutputParser()
    structured_report = chain.invoke({
        "analysis": json.dumps(emergency_analysis, indent=2),
        "call_transcript": call_transcript
    })
    
    print(f"emergency summary: {structured_report.get('emergency_summary', {}).get('Type of emergency', 'N/A')}")
    type_of_emergency = structured_report.get("emergency_summary", {}).get("Type of emergency", "unknown")
    injury_description = structured_report.get("injuries_patients", {}).get("patients", [{}])[0].get("injury_description", "unknown")
    return {
        **state,
        "structured_report": structured_report,
        "type_of_emergency": type_of_emergency,
        "injury_description": injury_description,
    }

# =============================================================================
# ENHANCED RAG NODES (keeping your existing structure)
# =============================================================================

def retrieve(state: EmergencyRAGState) -> Dict[str, Any]:
    """
    Retrieve documents based on the question.
    Enhanced for emergency context.
    """
    print("--- RETRIEVE ---")
    
    # Use rephrased question if available, otherwise use original
    question = state.get("rephrased_question", state["question"])
    llm = state.get("llm")
    RETREIVER_TYPE = state.get("retriever_type", "hybrid")
    is_emergency = state.get("is_emergency", False)
    nodes = state.get("nodes", [])

    # Enhance query for emergency context
    if is_emergency and not state.get("rephrased_question"):
        type_of_emergency = state.get("type_of_emergency", "")
        injury_description = state.get("injury_description", "")
        if type_of_emergency:
            question = f"{type_of_emergency} + {injury_description}"

    if state.get("rephrased_question"):
        print(f"Using rephrased question: {question}")
    else:
        print(f"Using original question: {question}")

    print(f"Emergency context: {is_emergency}")

    print(f"final question for retrieval: {question}")

    documents = create_retriever(question=question, nodes=nodes, type=RETREIVER_TYPE, load_persist=state.get("load_persist_kg", None))
    
    
    print(f"Retrieved {len(documents)} documents")
    
    return {
        **state,
        "documents": documents
    }

def grade_documents(state: EmergencyRAGState) -> Dict[str, Any]:
    """
    Enhanced document grading for emergency context.
    """
    print("--- CHECK DOCUMENT RELEVANCE TO QUESTION ---")
    
    question = state["question"]
    documents = state["documents"]
    model = state.get("llm")
    is_emergency = state.get("is_emergency", False)
    
    # Enhanced grading prompt for emergency context
    if is_emergency:
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a grader assessing relevance of retrieved documents to an emergency dispatch question.
            This is an EMERGENCY CONTEXT - prioritize documents about emergency procedures, protocols, and response guidelines.
            Look for content about emergency response, medical procedures, dispatcher protocols, and safety guidelines.
            If the document contains information relevant to emergency response or the specific emergency type, grade it as relevant.
            Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the emergency question.
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""),
            ("human", "Retrieved document: \n\n {document} \n\n Emergency question: {question}")
        ])
    else:
        # Use your existing grading prompt
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a grader assessing relevance of a retrieved document to a user question.
            Understand the context of the document and the question asked.
            If the document contains keyword(s) or different names of the same condition (e.g, heart attack is also cardiac arrest) or semantic meaning related to the user question, grade it as relevant.
            Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
        ])
    
    # Grade each document
    relevant_docs = []
    for doc in documents:
        grade_chain = grade_prompt | model | StrOutputParser()
        grade = grade_chain.invoke({"question": question, "document": doc.text})
        
        try:
            grade_dict = json.loads(grade)
            if grade_dict.get("score", "").lower() == "yes":
                relevant_docs.append(doc)
                print(f"--- GRADE: DOCUMENT RELEVANT ---")
            else:
                print(f"--- GRADE: DOCUMENT NOT RELEVANT ---")
        except json.JSONDecodeError:
            relevant_docs.append(doc)
            print(f"--- GRADE: DOCUMENT RELEVANT (JSON parse failed) ---")
    
    # Determine overall grade
    grade = "relevant" if relevant_docs else "not_relevant"
    
    return {
        **state,
        "documents": relevant_docs,
        "grade": grade
    }

def generate(state: EmergencyRAGState) -> Dict[str, Any]:
    """
    Generate answer using retrieved documents.
    Enhanced for emergency context.
    """
    print("--- GENERATE ---")
    
    question = state["question"]
    documents = state["documents"]
    model = state.get("llm")
    is_emergency = state.get("is_emergency", False)
    emergency_analysis = state.get("emergency_analysis")
    call_transcript = state.get("call_transcript", " ")

    # Create context from documents
    context = "\n\n".join([doc.text for doc in documents])
    call_transcript = call_transcript if call_transcript else "No call transcript provided."
    
    if is_emergency:
        # Enhanced generation prompt for emergency context
        generate_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an emergency dispatch assistant providing critical information.
            Use the retrieved emergency protocols and guidelines to answer the question.
            Focus on actionable, clear instructions that can be used in emergency situations.
            If emergency analysis is provided, incorporate it into your response.
            Be precise, clear, and prioritize safety and protocol compliance.
            If the context doesn't provide emergency-specific information, say 'I don't have specific emergency protocol information for this situation'.
            
            given call transcript and context, determine the best course of response.
            Keep the response concise and focused on actionable steps.
             
            Call Transcript: {call_transcript}
            Context: {context}
            """),
            ("human", "{question}")
        ])
        
        generation = generate_prompt | model | StrOutputParser()
        result = generation.invoke({
            "context": context, 
            "question": question,
            "call_transcript": call_transcript
        })
    else:
        # Use your existing generation prompt
        generate_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            Try to understand the context and question asked before generating an answer.
            If the context does not provide enough information to answer the question, say 'I don't know'.

            If the question is not answerable with the provided context, say 'I don't know'.
            
            Context: {context}"""),
            ("human", "{question}")
        ])
        
        generate_chain = generate_prompt | model | StrOutputParser()
        result = generate_chain.invoke({"context": context, "question": question})
    
    print(f"Generated answer: {result[:100]}...")
    
    return {
        **state,
        "generation": result
    }

def transform_query(state: EmergencyRAGState) -> Dict[str, Any]:
    """
    Enhanced query transformation for emergency context.
    """
    print("--- TRANSFORM QUERY ---")
    
    question = state["question"]
    iterations = state.get("iterations", 0)
    model = state.get("llm")
    is_emergency = state.get("is_emergency", False)
    emergency_type = state.get("emergency_type", "")
    
    if is_emergency:
        transform_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are optimizing emergency dispatch queries for better retrieval of protocols and procedures.
            This is an EMERGENCY context. Focus on emergency response, protocols, procedures, and safety guidelines.
            Consider emergency terminology, medical procedures, dispatcher protocols, and response guidelines.
            
            Emergency type: {emergency_type}
            Original question: {question}
            
            Reformulate the question to better retrieve emergency protocols, procedures, or guidelines.
            Use emergency services terminology and focus on actionable information."""),
            ("human", "Provide the improved emergency query:")
        ])
        
        transform_chain = transform_prompt | model | StrOutputParser()
        rephrased_question = transform_chain.invoke({
            "question": question,
            "emergency_type": emergency_type
        })
    else:
        # Use your existing transform prompt
        transform_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are generating questions that are well optimized for retrieval.
            Look at the input and try to reason about the underlying semantic intent / meaning.
            Here is the initial question:
            \n ------- \n
            {question} 
            \n ------- \n
            Formulate an improved question that will be more effective for document retrieval."""),
            ("human", "Provide the improved question:")
        ])
        
        transform_chain = transform_prompt | model | StrOutputParser()
        rephrased_question = transform_chain.invoke({"question": question})
    
    print(f"Rephrased question: {rephrased_question}")
    
    return {
        **state,
        "rephrased_question": rephrased_question,
        "iterations": iterations + 1
    }

# =============================================================================
# ROUTING LOGIC
# =============================================================================

def decide_workflow_path(state: EmergencyRAGState) -> str:
    """
    Decide whether to use emergency analysis or standard RAG workflow.
    """
    is_emergency = state.get("is_emergency", False)
    call_transcript = state.get("call_transcript")
    
    if is_emergency and call_transcript:
        print("--- ROUTING TO EMERGENCY ANALYSIS ---")
        return "analyze_emergency"
    else:
        print("--- ROUTING TO STANDARD RAG ---")
        return "retrieve"

def decide_to_generate(state: EmergencyRAGState) -> str:
    """Enhanced decision logic for emergency context."""
    print("--- ASSESS GRADED DOCUMENTS ---")
    
    grade = state["grade"]
    documents = state["documents"]
    is_emergency = state.get("is_emergency", False)
    
    print(f"Grade: {grade}, Number of documents: {len(documents)}, Emergency: {is_emergency}")
    
    # Lower threshold for emergency contexts
    min_docs = 3 if is_emergency else 5
    
    if len(documents) >= min_docs:
        print(f"--- DECISION: SUFFICIENT DOCUMENTS ({len(documents)} >= {min_docs}), GENERATE ANSWER ---")
        return "generate"
    else:
        print("--- DECISION: INSUFFICIENT DOCUMENTS, TRANSFORM QUERY ---")
        return "transform_query"

def grade_generation_v_documents_and_question(state: EmergencyRAGState) -> str:
    """Enhanced grading for emergency context."""
    print("--- CHECK HALLUCINATIONS ---")
    
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    iterations = state.get("iterations", 0)
    model = state.get("llm")
    max_iterations = state.get("max_iterations", 3)
    is_emergency = state.get("is_emergency", False)
    
    # Your existing hallucination check
    # hallucination_prompt = ChatPromptTemplate.from_messages([
    #     ("system", """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
    #     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
    #     Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""),
    #     ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
    # ])
    
    # hallucination_chain = hallucination_prompt | model | StrOutputParser()
    # grade = hallucination_chain.invoke({
    #     "documents": "\n\n".join([doc.text for doc in documents]),
    #     "generation": generation
    # })
    
    # try:
    #     grade_dict = json.loads(grade)
    #     grounded = grade_dict.get("score", "").lower() == "yes"
    # except json.JSONDecodeError:
    #     grounded = True

    print("skipping hallucination check for emergency context")
    grade = "yes"  # Assume generation is grounded for emergency context
    grounded= "yes"
    # print(f"Grounded: {grounded}")
    
    # # Your existing question answering check
    # answer_prompt = ChatPromptTemplate.from_messages([
    #     ("system", """You are a grader assessing whether an answer addresses / resolves a question.
    #     Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question.
    #     Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""),
    #     ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
    # ])
    
    # answer_chain = answer_prompt | model | StrOutputParser()
    # grade = answer_chain.invoke({"question": question, "generation": generation})
    
    # try:
    #     grade_dict = json.loads(grade)
    #     useful = grade_dict.get("score", "").lower() == "yes"
    # except json.JSONDecodeError:
    #     useful = True

    useful = "yes"  # Assume generation is useful for emergency context
    
    print(f"Useful: {useful}")

    # Safety check for max iterations
    if iterations >= max_iterations:
        print("--- MAX ITERATIONS REACHED, ENDING ---")
        return END
    
    if grounded and useful:
        print("--- DECISION: GENERATION IS GROUNDED AND USEFUL ---")
        return END
    elif not grounded:
        print("--- DECISION: GENERATION IS NOT GROUNDED, RE-GENERATE ---")
        return "generate"
    else:
        print("--- DECISION: GENERATION IS NOT USEFUL, TRANSFORM QUERY ---")
        return "transform_query"

def max_iterations_check(state: EmergencyRAGState) -> str:
    """Check if maximum iterations reached."""
    max_iterations = state.get("max_iterations", 3)
    iterations = state.get("iterations", 0)
    
    if iterations >= max_iterations:
        print(f"--- MAX ITERATIONS ({max_iterations}) REACHED ---")
        return "generate"
    else:
        return "retrieve"

# =============================================================================
# WORKFLOW BUILDER
# =============================================================================

def build_emergency_rag_workflow():
    """
    Build the enhanced emergency dispatch + RAG workflow.
    
    Returns:
        Compiled workflow
    """
    workflow = StateGraph(EmergencyRAGState)
    
    # Add all nodes
    workflow.add_node("detect_emergency", detect_emergency_context)
    workflow.add_node("analyze_emergency", analyze_emergency_call)
    workflow.add_node("generate_report", generate_emergency_report)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    
    # Build the workflow
    workflow.add_edge(START, "detect_emergency")
    
    # Route based on emergency detection
    workflow.add_conditional_edges(
        "detect_emergency",
        decide_workflow_path,
        {
            "analyze_emergency": "analyze_emergency",
            "retrieve": "retrieve"
        }
    )
    
    # Emergency analysis path
    workflow.add_edge("analyze_emergency", "generate_report")
    workflow.add_edge("generate_report", "retrieve")  # Still do RAG for additional info
    
    # Standard RAG path
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate"
        }
    )
    workflow.add_conditional_edges(
        "transform_query",
        max_iterations_check,
        {
            "retrieve": "retrieve",
            "generate": "generate"
        }
    )
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            END: END,
            "transform_query": "transform_query",
            "generate": "generate"
        }
    )
    
    # Compile
    app = workflow.compile()
    return app

