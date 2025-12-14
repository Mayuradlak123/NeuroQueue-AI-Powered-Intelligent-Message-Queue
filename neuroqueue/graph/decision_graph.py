from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
import torch

from neuroqueue.ai.models.intent_net import IntentAttentionNet
from neuroqueue.ai.models.risk_net import RiskTransformer
from neuroqueue.ai.priority_model import PriorityPredictor
from neuroqueue.etl.pipeline import etl_pipeline
from neuroqueue.ai.embeddings import embedder
from neuroqueue.storage.mongo import mongo_client
from neuroqueue.config.logger import logger
from neuroqueue.storage.chroma import chroma_client
from neuroqueue.ai.llm import llm_client

# --- Load Models ---
try:
    intent_model = IntentAttentionNet(feature_dim=394)
    intent_model.load_state_dict(torch.load("models/intent_net.pth"))
    intent_model.eval()

    risk_model = RiskTransformer()
    risk_model.load_state_dict(torch.load("models/risk_net.pth"))
    risk_model.eval()
    
    priority_model = PriorityPredictor()
    priority_model.load_state_dict(torch.load("models/priority_model.pth"))
    priority_model.eval()
except FileNotFoundError:
    pass

# --- State ---
class AgentState(TypedDict):
    message_id: str
    content: str
    metadata: dict
    # features
    embedding: torch.Tensor
    combined_features: torch.Tensor 
    
    # outputs
    intent: str
    priority: float
    failure_risk: float
    routing_decision: str
    explanation: str
    
    # New: LLM specific
    llm_reasoning: str

# --- Nodes ---

def etl_node(state: AgentState):
    """Runs ETL Pipeline: Text -> Features (394 dim)"""
    logger.info(f"Running ETL for: {state['message_id']}")
    combined_feats = etl_pipeline.process_message(state['content'])
    embedding = combined_feats[:384] 
    return {"combined_features": combined_feats, "embedding": embedding}

def intent_classification_node(state: AgentState):
    """(Legacy) Uses Attention-based Net"""
    intent, conf = intent_model.predict(state["combined_features"])
    return {"intent": intent} # Will be overwritten by LLM if successful

def priority_prediction_node(state: AgentState):
    """(Legacy) Uses MLP"""
    prio = priority_model.predict(state["embedding"])
    return {"priority": prio}

def risk_assessment_node(state: AgentState):
    """(Legacy) Uses Transformer"""
    dummy_seq = torch.rand(1, 5, 4) 
    risk = risk_model.predict(dummy_seq)
    return {"failure_risk": risk}

def llm_analysis_node(state: AgentState):
    """
    Uses Groq/LLM to understand full context, domain, and urgency.
    This overrides the simpler models if successful.
    """
    logger.info(f"Asking LLM to reason about: '{state['content']}'")
    analysis = llm_client.analyze_message(state['content'])
    
    updates = {}
    if analysis.get("intent"):
        updates["intent"] = analysis["intent"]
    
    # LLM usually gives better semantic priority
    if "priority" in analysis:
        updates["priority"] = float(analysis["priority"])
        
    if "risk" in analysis:
        updates["failure_risk"] = float(analysis["risk"])
        
    if "reasoning" in analysis:
        updates["llm_reasoning"] = analysis["reasoning"]
        
    logger.info(f"LLM Output: {updates}")
    return updates

def routing_reasoning_node(state: AgentState):
    # Default Rule: High Priority -> Critical Queue
    decision = "NORMAL_QUEUE"
    
    # Dynamic logic based on State (which may be populated by LLM)
    prio = state.get('priority', 0.0)
    risk = state.get('failure_risk', 0.0)
    intent = state.get('intent', 'general')
    
    if risk > 0.8:
        decision = "DLQ_INVESTIGATION"
    elif prio > 0.7:
        decision = "CRITICAL_QUEUE"
    elif "payment" in intent.lower() or "billing" in intent.lower():
        decision = "PAYMENT_QUEUE"
    
    # Capture reasoning
    reasoning = state.get('llm_reasoning', 'Rule-based logic')
    explanation = f"[SmartRoute] Decision:{decision} | Prio:{prio:.2f} | Reason: {reasoning}"
    
    # Audit Log (Mongo)
    mongo_client.log_audit(state['message_id'], {
        "content": state['content'],
        "intent": intent,
        "priority": prio,
        "risk": risk,
        "decision": decision,
        "explanation": explanation
    })

    # Vector Memory (ChromaDB)
    try:
        analysis_text = f"Intent: {intent} | Priority: {prio:.2f} | Risk: {risk:.2f} | Reason: {reasoning}"
        chroma_client.store_message_vector(
            message_id=state['message_id'],
            content=state['content'], 
            analysis=analysis_text
        )
    except Exception as e:
        logger.error(f"ChromaDB Write Failed: {e}")
    
    return {"routing_decision": decision, "explanation": explanation}

# --- Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("etl", etl_node)
workflow.add_node("intent", intent_classification_node)
workflow.add_node("priority", priority_prediction_node)
workflow.add_node("risk", risk_assessment_node)
workflow.add_node("llm", llm_analysis_node) # New Brain
workflow.add_node("decide", routing_reasoning_node)

workflow.set_entry_point("etl")

# 1. Run Legacy Models in parallel first
workflow.add_edge("etl", "intent")
workflow.add_edge("etl", "priority")
workflow.add_edge("etl", "risk")

# 2. Then run LLM (it can overwrite legacy outputs safely here)
workflow.add_edge("intent", "llm")
workflow.add_edge("priority", "llm")
workflow.add_edge("risk", "llm")

# 3. Finally Decide
workflow.add_edge("llm", "decide")

workflow.add_edge("decide", END)

decision_app = workflow.compile()
