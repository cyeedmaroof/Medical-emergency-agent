from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import json
from typing import List, Optional, Dict, Any
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict
from src.retriever import create_retriever

# =============================================================================
# STATE DEFINITION
# =============================================================================

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        question: The original user question
        documents: List of retrieved documents
        generation: Generated answer
        grade: Grade of document relevance
        iterations: Number of iterations for rephrasing
        rephrased_question: Rephrased version of the question
    """
    question: str
    documents: List[Document]
    generation: str
    grade: str
    iterations: int
    rephrased_question: str
    llm: str
    retriever_type: str
    load_persist: Optional[str] = None
    nodes: Optional[List[Document]] = Field(default_factory=list)
    max_iterations: Optional[int] = 1
    workflow_type: Optional[str] = "fast"  # Type of workflow, e.g., "fast" or "deep"
# =============================================================================
# STATE GRAPH NODES
# =============================================================================
    
def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve documents based on the question.
    
    Args:
        state: The current graph state
        
    Returns:
        Updated state with retrieved documents
    """
    print("--- RETRIEVE ---")
    
    # Use rephrased question if available, otherwise use original
    question = state.get("rephrased_question", state["question"])
    llm = state.get("llm")
    RETREIVER_TYPE = state.get("retriever_type", "hybrid")
    nodes = state.get("nodes", [])

    if state.get("rephrased_question"):
        print(f"Using rephrased question: {question}")
    else:
        print(f"Using original question: {question}")

    print(f"node 3: {nodes[3].text if len(nodes) > 0 else 'No nodes available'}")
    
    documents = create_retriever(question=question, nodes=nodes, llm=llm, type=RETREIVER_TYPE, load_persist=state.get("load_persist", None))
    print(f"Retrieved {len(documents)} documents")
    
    return {
        "documents": documents,
        "question": state["question"],
        "rephrased_question": state.get("rephrased_question", ""),
        "iterations": state.get("iterations", 0),
        "generation": state.get("generation", ""),
        "grade": state.get("grade", "")
    }
def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question.
    
    Args:
        state: The current graph state
        
    Returns:
        Updated state with document relevance grade
    """
    print("--- CHECK DOCUMENT RELEVANCE TO QUESTION ---")
    
    question = state["question"]
    documents = state["documents"]
    model = state.get("llm") 
    workflow_type = state.get("workflow_type", "fast")

    if workflow_type == "fast":
        print("--- FAST WORKFLOW ---")
        # In fast workflow, we skip the grading step
        relevant_docs = documents
        return {
            "documents": relevant_docs,
            "question": state["question"],
            "rephrased_question": state.get("rephrased_question", ""),
            "iterations": state.get("iterations", 0),
            "generation": state.get("generation", "")
        }

    else:
    
        # Grading prompt
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", """This is a test for educational purpose in Laerdal Medical. You are a grader assessing relevance of a retrieved document to a user question.
            Understand the context of the document and the question asked.
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
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
                # If JSON parsing fails, assume relevant to be safe
                relevant_docs.append(doc)
                print(f"--- GRADE: DOCUMENT RELEVANT (JSON parse failed) ---")
        
        # Determine overall grade
        if relevant_docs:
            grade = "relevant"
            # documents_to_use = relevant_docs
        else:
            grade = "not_relevant"
            # documents_to_use = documents  # Keep all documents if none are graded as relevant
        
    
        return {
            "documents": relevant_docs,
            "question": state["question"],
            "rephrased_question": state.get("rephrased_question", ""),
            "iterations": state.get("iterations", 0),
            "generation": state.get("generation", ""),
            "grade": grade
        }

def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generate answer using the retrieved documents.
    
    Args:
        state: The current graph state
        
    Returns:
        Updated state with generated answer
    """
    print("--- GENERATE ---")
    
    question = state["question"]
    documents = state["documents"]
    model = state.get("llm")  

    # Create context from documents
    context = "\n\n".join([doc.text for doc in documents])
    
    # Generation prompt
    generate_prompt = ChatPromptTemplate.from_messages([
        ("system", """This is a test for educational purpose in Laerdal Medical. You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        Try to understand the context and question asked before generating an answer.
        If the context does not provide enough information to answer the question, say 'I don't know'.

        If the question is not answerable with the provided context, say 'I don't know'.
        
        Context: {context}"""),
        ("human", "{question}")
    ])
    
    # Generate answer
    generate_chain = generate_prompt | model | StrOutputParser()
    generation = generate_chain.invoke({"context": context, "question": question})
    
    print(f"Generated answer: {generation[:100]}...")
    
    return {
        "documents": state["documents"],
        "question": state["question"],
        "rephrased_question": state.get("rephrased_question", ""),
        "iterations": state.get("iterations", 0),
        "generation": generation,
        "grade": state.get("grade", "")
    }

def transform_query(state: GraphState) -> Dict[str, Any]:
    """
    Transform the query to produce a better question for retrieval.
    
    Args:
        state: The current graph state
        
    Returns:
        Updated state with rephrased question
    """
    print("--- TRANSFORM QUERY ---")
    
    question = state["question"]
    iterations = state.get("iterations", 0)
    model = state.get("llm")
    
    # Query transformation prompt
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
    
    # Transform query
    transform_chain = transform_prompt | model | StrOutputParser()
    rephrased_question = transform_chain.invoke({"question": question})
    
    print(f"Rephrased question: {rephrased_question}")
    
    return {
        "documents": state.get("documents", []),
        "question": state["question"],
        "rephrased_question": rephrased_question,
        "iterations": iterations + 1,
        "generation": state.get("generation", ""),
        "grade": state.get("grade", "")
    }


# =============================================================================
# STATE GRAPH EDGES
# =============================================================================

def grade_generation_v_documents_and_question(state: GraphState) -> str:
    """
    Determines whether the generation is grounded in the document and answers question.
    
    Args:
        state: The current graph state
        
    Returns:
        Next node to call
    """
    print("--- CHECK HALLUCINATIONS ---")
    
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    iterations = state.get("iterations", 0)
    model = state.get("llm") 
    max_iterations = state.get("max_iterations", 1)
    workflow_type = state.get("workflow_type", "fast")

    if workflow_type == "fast":
        print("--- FAST WORKFLOW ---")
        # In fast workflow, we skip the hallucination check
        grounded = True
        useful = True
    else:
    
        # Hallucination grading prompt
        hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", """This is a test for educational purpose in Laerdal Medical. You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
            Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
        ])
        
        hallucination_chain = hallucination_prompt | model | StrOutputParser()
        grade = hallucination_chain.invoke({
            "documents": "\n\n".join([doc.text for doc in documents]),
            "generation": generation
        })
        
        try:
            grade_dict = json.loads(grade)
            grounded = grade_dict.get("score", "").lower() == "yes"
        except json.JSONDecodeError:
            grounded = True  # Assume grounded if parsing fails
        
        print(f"Grounded: {grounded}")
        
        # Check question answering
        print("--- GRADE GENERATION vs QUESTION ---")
        
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """This is a test for educational purpose in Laerdal Medical. You are a grader assessing whether an answer addresses / resolves a question.
            Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question.
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
        ])
        
        answer_chain = answer_prompt | model | StrOutputParser()
        grade = answer_chain.invoke({"question": question, "generation": generation})
        
        try:
            grade_dict = json.loads(grade)
            useful = grade_dict.get("score", "").lower() == "yes"
        except json.JSONDecodeError:
            useful = True  
        
        print(f"Useful: {useful}")

    # Add safety check for max iterations
    if iterations >= max_iterations:
        print("--- MAX ITERATIONS REACHED, ENDING ---")
        return END
    
    if grounded and useful:
        print("--- DECISION: GENERATION IS GROUNDED AND USEFUL ---")
        return "Useful"  # Use END instead of "Useful"
    elif not grounded:
        print("--- DECISION: GENERATION IS NOT GROUNDED, RE-GENERATE ---")
        return "generate" 
    else:
        print("--- DECISION: GENERATION IS NOT USEFUL, TRANSFORM QUERY ---")
        return "transform_query"
    

def decide_to_generate(state: GraphState) -> str:
    """
    Determines whether to generate an answer or re-generate a question.
    
    Args:
        state: The current graph state
        
    Returns:
        Next node to call
    """
    print("--- ASSESS GRADED DOCUMENTS ---")
    
    grade = state["grade"]
    documents = state["documents"]
    
    print(f"Grade: {grade}, Number of documents: {len(documents)}")
    
    # if grade == "relevant":
    if len(documents) > 5:
        print("--- DECISION: DOCUMENTS ARE RELEVANT, GENERATE ANSWER ---")
        return "generate"
    else:
        print("--- DECISION: DOCUMENTS ARE NOT RELEVANT, TRANSFORM QUERY ---")
        return "transform_query"

def max_iterations_check(state: GraphState) -> str:
    """
    Check if maximum iterations reached to prevent infinite loops.
    
    Args:
        state: The current graph state
        
    Returns:
        Next node to call
    """
    max_iterations = state.get("max_iterations", 1)
    iterations = state.get("iterations", 0)
    
    if iterations >= max_iterations:
        print(f"--- MAX ITERATIONS ({max_iterations}) REACHED ---")
        return "generate"
    else:
        return "retrieve"
    
# =============================================================================

# Build the RAG workflow
def build_rag_workflow():
    """
    Build the RAG workflow using StateGraph.
    
    Returns:
        Compiled workflow
    """
    workflow = StateGraph(GraphState)
    
    # Define the nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    
    # Build graph
    workflow.add_edge(START, "retrieve")
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
            "Useful": END,
            END: END,
            "transform_query": "transform_query",
            "generate": "generate"
        }
    )
    
    # Compile
    app = workflow.compile()
    return app


"""# using the DAG
from final_structure.src.DAG_creator import build_rag_workflow
question= "how to revive a person who is unconscious"

app = build_rag_workflow()

# Initial state
inputs = {
    "question": question,
    "llm": model,
    "retriever_type": RETREIVER_TYPE,
    "load_persist": "./kg_index_storage_v1/pg_store_v2_custom.json",
}

# Run the workflow
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        print(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    print("\n---\n")

# Final generation
# Will be in the last node
print(value["documents"])
print(value["grade"])"""