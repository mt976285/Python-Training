import streamlit as st
import json
import re
import time
import functools
from datetime import datetime
from typing import Optional, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Load Knowledge Base
# =========================
with open("case_tasks.json") as f:
    KB = json.load(f)["tasks"]

# =========================
# Audit Log
# =========================
AUDIT_LOG = []

def audit(event_type: str, details: dict):
    AUDIT_LOG.append({
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "details": details
    })

# =========================
# Utils
# =========================
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())

def is_account_number_only(text: str) -> bool:
    return bool(re.fullmatch(r"\d{6,20}", text.strip()))

def get_top_tasks(limit=3):
    return sorted(
        KB.items(),
        key=lambda x: x[1].get("frequency_rank", 999)
    )[:limit]

# =========================
# RAG setup
# =========================
def build_task_index(kb):
    corpus = []
    task_names = []
    for task_name, task in kb.items():
        text = (
            task_name.replace("_"," ") + " " +
            task.get("description","") + " " +
            " ".join(task.get("aliases",[]))
        )
        corpus.append(text.lower())
        task_names.append(task_name)
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix, task_names

VECTORIZER, TASK_MATRIX, TASK_NAMES = build_task_index(KB)

def rag_discover_tasks(user_input, top_k=3):
    query_vec = VECTORIZER.transform([user_input.lower()])
    scores = cosine_similarity(query_vec, TASK_MATRIX)[0]
    ranked = sorted(zip(TASK_NAMES, scores), key=lambda x: x[1], reverse=True)
    return [(t,s) for t,s in ranked if s>0][:top_k]

# =========================
# MCP Wrapper with retry + timeout
# =========================
def mcp_with_retry(max_retries=3, timeout_seconds=2):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_retries:
                try:
                    start = time.time()
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start
                    if elapsed > timeout_seconds:
                        raise TimeoutError("MCP call timed out")
                    return result
                except Exception as e:
                    attempts += 1
                    if attempts >= max_retries:
                        raise RuntimeError(f"MCP call failed after {max_retries} attempts: {str(e)}")
                    time.sleep(1)
        return wrapper
    return decorator

# =========================
# MCP Tool Stubs
# =========================
@mcp_with_retry()
def mcp_get_account_details(account_number: str):
    return {"account_number": account_number, "account_type": "joint", "status":"active", "balance":12000}

@mcp_with_retry()
def mcp_close_account(account_number, closure_reason, docs):
    return {"status":"SUCCESS","reference_id":"CLS-2026-9911"}

@mcp_with_retry()
def mcp_increase_limit(account_number, requested_limit):
    return {"status":"SUCCESS","new_limit":requested_limit}

# =========================
# Task resolution
# =========================
def resolve_task_with_confidence(user_input):
    ui = normalize(user_input)
    best = {"task": None, "score": 0.0}
    for task_name, task in KB.items():
        canonical = normalize(task_name.replace("_"," "))
        aliases = [normalize(a) for a in task.get("aliases",[])]
        if ui == canonical:
            return task_name, 1.0
        if ui in aliases:
            return task_name, 0.9
        for a in aliases:
            if a in ui or ui in a:
                best = max(best, {"task": task_name, "score":0.75}, key=lambda x: x["score"])
        if set(ui.split()) & set(canonical.split()):
            best = max(best, {"task": task_name, "score":0.6}, key=lambda x:x["score"])
    return best["task"], best["score"]

def resolve_task_hybrid(user_input):
    task, score = resolve_task_with_confidence(user_input)
    if score >= 0.85:
        return task, score
    rag_results = rag_discover_tasks(user_input)
    if rag_results:
        best_task, rag_score = rag_results[0]
        combined_score = max(score, min(0.75, rag_score))
        return best_task, combined_score
    return None, 0.0

# =========================
# Conditional logic
# =========================
def evaluate_condition(condition, collected):
    if "AND" in condition:
        result = all(evaluate_condition(c,collected) for c in condition["AND"])
        if result: audit("RULE_TRIGGERED", {"rule": condition})
        return result
    if "OR" in condition:
        result = any(evaluate_condition(c,collected) for c in condition["OR"])
        if result: audit("RULE_TRIGGERED", {"rule": condition})
        return result
    field = condition["field"]
    operator = condition["operator"]
    expected = condition["value"]
    actual = collected.get(field)
    if actual is None: return False
    if operator=="IN": return actual.lower() in [v.lower() for v in expected]
    if operator=="EQUALS": return actual.lower() == expected.lower()
    return False

def get_conditional_fields(task_name, collected):
    fields = {}
    for rule in KB[task_name].get("conditional_fields", []):
        if evaluate_condition(rule["when"], collected):
            fields.update(rule["then"])
    return fields

def get_next_question(task_name, collected):
    base = KB[task_name]["required_fields"]
    conditional = get_conditional_fields(task_name, collected)
    all_fields = {**base, **conditional}
    for field, meta in all_fields.items():
        if field not in collected:
            return field, meta
    return None, None

def validate_input(meta, value):
    if meta["type"]=="enum":
        if value.lower() not in [v.lower() for v in meta["values"]]:
            return False, f"Allowed values: {', '.join(meta['values'])}"
    if meta["type"]=="number":
        if not str(value).isdigit():
            return False, "Enter numeric value"
    return True, None

def prefill_from_mcp(task_name, collected, account_details):
    required = KB[task_name]["required_fields"]
    for field in required:
        if field not in collected and field in account_details:
            collected[field] = str(account_details[field])

# =========================
# Execute MCP per task
# =========================
def execute_task(task_name, data):
    audit("MCP_INVOKED", {"task":task_name})
    if task_name=="CLOSE_ACCOUNT":
        docs = {}
        if "death_certificate_number" in data:
            docs["death_certificate_number"] = data["death_certificate_number"]
        result = mcp_close_account(data["account_number"], data["closure_reason"], docs)
    elif task_name=="INCREASE_LIMIT":
        result = mcp_increase_limit(data["account_number"], int(data["requested_limit"]))
    else:
        raise RuntimeError("No MCP tool mapped for task")
    audit("MCP_RESULT", result)
    return result

# =========================
# In-memory case store
# =========================
CASE_STORE: Dict[str, Dict] = {}
CASE_COUNTER = 1

def start_case(user_input: str, user_id: str = "anonymous") -> Dict[str, Any]:
    global CASE_COUNTER
    case_id = f"CASE-{CASE_COUNTER}"
    CASE_COUNTER += 1

    state = {"task": None, "collected_fields": {}, "user_id": user_id}
    audit("USER_INPUT", {"user_input": user_input, "user_id": user_id})

    # Account number first
    if is_account_number_only(user_input):
        state["collected_fields"]["account_number"] = user_input
        audit("ACCOUNT_NUMBER_ENTERED", {"account_number": user_input})

        account_details = mcp_get_account_details(user_input)
        audit("MCP_INVOKED", {"task":"get_account_details"})
        state["account_details"] = account_details

    # Task resolution
    task, score = resolve_task_hybrid(user_input)
    if score < 0.6:
        CASE_STORE[case_id] = state
        return {"case_id": case_id, "status":"awaiting_task_selection",
                "message":"Please specify the task from supported tasks.", "available_tasks": list(KB.keys())}
    state["task"] = task
    audit("TASK_SELECTED", {"task":task, "user_id":user_id})

    prefill_from_mcp(task, state["collected_fields"], state.get("account_details", {}))

    CASE_STORE[case_id] = state
    next_field, meta = get_next_question(task, state["collected_fields"])
    return {"case_id": case_id, "task": task, "collected_fields": state["collected_fields"], "next_field": next_field, "prompt": meta["prompt"] if meta else None}

def answer_field(case_id: str, field: str, value: Any) -> Dict[str, Any]:
    case = CASE_STORE.get(case_id)
    if not case:
        return {"error": "Case not found"}
    case["collected_fields"][field] = value
    audit("FIELD_COLLECTED", {"case_id":case_id, "field":field, "value":value})
    next_field, meta = get_next_question(case["task"], case["collected_fields"])
    if next_field is None:
        mcp_result = execute_task(case["task"], case["collected_fields"])
        audit("CASE_COMPLETED", {"case_id":case_id})
        return {"status":"completed", "case":case, "mcp_result": mcp_result}
    else:
        return {"status":"awaiting_field", "next_field":next_field, "prompt":meta["prompt"]}

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Case Management AI", layout="wide")
st.title("Case Management AI — UI")

col1, col2 = st.columns([2,1])

with col1:
    st.header("Start a Case")
    user_input = st.text_input("Enter user input (task or account number)")
    user_id = st.text_input("User ID (optional)", value="anonymous")
    if st.button("Start Case"):
        result = start_case(user_input, user_id)
        st.session_state["last_case"] = result
        st.experimental_rerun()

    if "last_case" in st.session_state:
        res = st.session_state["last_case"]
        st.subheader("Case Created")
        st.json(res)

    # If awaiting field, show input widget
    if "last_case" in st.session_state and res.get("status") != "awaiting_task_selection":
        next_field = res.get("next_field")
        if next_field:
            st.subheader(f"Next: {next_field}")
            answer = st.text_input("Answer")
            if st.button("Submit Answer"):
                out = answer_field(res["case_id"], next_field, answer)
                st.session_state["last_case"] = out.get("case", res)
                st.write(out)

with col2:
    st.header("Admin / Debug")
    st.subheader("Cases")
    st.write(CASE_STORE)
    st.subheader("Audit Log")
    st.write(AUDIT_LOG)
    st.subheader("Top Tasks")
    st.write([t[0] for t in get_top_tasks(5)])
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import re
import time
import functools
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Load Knowledge Base
# =========================
with open("case_tasks.json") as f:
    KB = json.load(f)["tasks"]

# =========================
# Audit Log
# =========================
AUDIT_LOG = []

def audit(event_type: str, details: dict):
    AUDIT_LOG.append({
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "details": details
    })

# =========================
# Utils
# =========================
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())

def is_account_number_only(text: str) -> bool:
    return bool(re.fullmatch(r"\d{6,20}", text.strip()))

def get_top_tasks(limit=3):
    return sorted(
        KB.items(),
        key=lambda x: x[1].get("frequency_rank", 999)
    )[:limit]

# =========================
# RAG setup
# =========================
def build_task_index(kb):
    corpus = []
    task_names = []
    for task_name, task in kb.items():
        text = (
            task_name.replace("_"," ") + " " +
            task.get("description","") + " " +
            " ".join(task.get("aliases",[]))
        )
        corpus.append(text.lower())
        task_names.append(task_name)
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix, task_names

VECTORIZER, TASK_MATRIX, TASK_NAMES = build_task_index(KB)

def rag_discover_tasks(user_input, top_k=3):
    query_vec = VECTORIZER.transform([user_input.lower()])
    scores = cosine_similarity(query_vec, TASK_MATRIX)[0]
    ranked = sorted(zip(TASK_NAMES, scores), key=lambda x: x[1], reverse=True)
    return [(t,s) for t,s in ranked if s>0][:top_k]

# =========================
# MCP Wrapper with retry + timeout
# =========================
def mcp_with_retry(max_retries=3, timeout_seconds=2):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_retries:
                try:
                    start = time.time()
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start
                    if elapsed > timeout_seconds:
                        raise TimeoutError("MCP call timed out")
                    return result
                except Exception as e:
                    attempts += 1
                    if attempts >= max_retries:
                        raise RuntimeError(f"MCP call failed after {max_retries} attempts: {str(e)}")
                    time.sleep(1)
        return wrapper
    return decorator

# =========================
# MCP Tool Stubs
# =========================
@mcp_with_retry()
def mcp_get_account_details(account_number: str):
    return {"account_number": account_number, "account_type": "joint", "status":"active", "balance":12000}

@mcp_with_retry()
def mcp_close_account(account_number, closure_reason, docs):
    return {"status":"SUCCESS","reference_id":"CLS-2026-9911"}

@mcp_with_retry()
def mcp_increase_limit(account_number, requested_limit):
    return {"status":"SUCCESS","new_limit":requested_limit}

# =========================
# Task resolution
# =========================
def resolve_task_with_confidence(user_input):
    ui = normalize(user_input)
    best = {"task": None, "score": 0.0}
    for task_name, task in KB.items():
        canonical = normalize(task_name.replace("_"," "))
        aliases = [normalize(a) for a in task.get("aliases",[])]
        if ui == canonical:
            return task_name, 1.0
        if ui in aliases:
            return task_name, 0.9
        for a in aliases:
            if a in ui or ui in a:
                best = max(best, {"task": task_name, "score":0.75}, key=lambda x: x["score"])
        if set(ui.split()) & set(canonical.split()):
            best = max(best, {"task": task_name, "score":0.6}, key=lambda x:x["score"])
    return best["task"], best["score"]

def resolve_task_hybrid(user_input):
    task, score = resolve_task_with_confidence(user_input)
    if score >= 0.85:
        return task, score
    rag_results = rag_discover_tasks(user_input)
    if rag_results:
        best_task, rag_score = rag_results[0]
        combined_score = max(score, min(0.75, rag_score))
        return best_task, combined_score
    return None, 0.0

# =========================
# Conditional logic
# =========================
def evaluate_condition(condition, collected):
    if "AND" in condition:
        result = all(evaluate_condition(c,collected) for c in condition["AND"])
        if result: audit("RULE_TRIGGERED", {"rule": condition})
        return result
    if "OR" in condition:
        result = any(evaluate_condition(c,collected) for c in condition["OR"])
        if result: audit("RULE_TRIGGERED", {"rule": condition})
        return result
    field = condition["field"]
    operator = condition["operator"]
    expected = condition["value"]
    actual = collected.get(field)
    if actual is None: return False
    if operator=="IN": return actual.lower() in [v.lower() for v in expected]
    if operator=="EQUALS": return actual.lower() == expected.lower()
    return False

def get_conditional_fields(task_name, collected):
    fields = {}
    for rule in KB[task_name].get("conditional_fields", []):
        if evaluate_condition(rule["when"], collected):
            fields.update(rule["then"])
    return fields

def get_next_question(task_name, collected):
    base = KB[task_name]["required_fields"]
    conditional = get_conditional_fields(task_name, collected)
    all_fields = {**base, **conditional}
    for field, meta in all_fields.items():
        if field not in collected:
            return field, meta
    return None, None

def validate_input(meta, value):
    if meta["type"]=="enum":
        if value.lower() not in [v.lower() for v in meta["values"]]:
            return False, f"Allowed values: {', '.join(meta['values'])}"
    if meta["type"]=="number":
        if not str(value).isdigit():
            return False, "Enter numeric value"
    return True, None

def prefill_from_mcp(task_name, collected, account_details):
    required = KB[task_name]["required_fields"]
    for field in required:
        if field not in collected and field in account_details:
            collected[field] = str(account_details[field])

# =========================
# Execute MCP per task
# =========================
def execute_task(task_name, data):
    audit("MCP_INVOKED", {"task":task_name})
    if task_name=="CLOSE_ACCOUNT":
        docs = {}
        if "death_certificate_number" in data:
            docs["death_certificate_number"] = data["death_certificate_number"]
        result = mcp_close_account(data["account_number"], data["closure_reason"], docs)
    elif task_name=="INCREASE_LIMIT":
        result = mcp_increase_limit(data["account_number"], int(data["requested_limit"]))
    else:
        raise RuntimeError("No MCP tool mapped for task")
    audit("MCP_RESULT", result)
    return result

# =========================
# FastAPI App
# =========================
app = FastAPI(title="Case Management Agent API")

# Request Models
class StartCaseRequest(BaseModel):
    user_input: str
    user_id: Optional[str] = "anonymous"

class AnswerFieldRequest(BaseModel):
    case_id: str
    field: str
    value: Any

# In-memory case store
CASE_STORE: Dict[str, Dict] = {}
CASE_COUNTER = 1

@app.post("/start_case")
def start_case(req: StartCaseRequest):
    global CASE_COUNTER
    case_id = f"CASE-{CASE_COUNTER}"
    CASE_COUNTER += 1

    state = {"task": None, "collected_fields": {}, "user_id": req.user_id}
    audit("USER_INPUT", {"user_input": req.user_input, "user_id": req.user_id})

    # Account number first
    if is_account_number_only(req.user_input):
        state["collected_fields"]["account_number"] = req.user_input
        audit("ACCOUNT_NUMBER_ENTERED", {"account_number": req.user_input})

        account_details = mcp_get_account_details(req.user_input)
        audit("MCP_INVOKED", {"task":"get_account_details"})
        state["account_details"] = account_details

    # Task resolution
    task, score = resolve_task_hybrid(req.user_input)
    if score < 0.6:
        return {"case_id": case_id, "status":"awaiting_task_selection",
                "message":"Please specify the task from supported tasks."}
    state["task"] = task
    audit("TASK_SELECTED", {"task":task, "user_id":req.user_id})

    prefill_from_mcp(task, state["collected_fields"], state.get("account_details", {}))

    CASE_STORE[case_id] = state
    return {"case_id": case_id, "task": task, "collected_fields": state["collected_fields"]}

@app.post("/answer_field")
def answer_field(req: AnswerFieldRequest):
    case = CASE_STORE.get(req.case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    case["collected_fields"][req.field] = req.value
    audit("FIELD_COLLECTED", {"case_id":req.case_id, "field":req.field, "value":req.value})
    # Next field
    next_field, meta = get_next_question(case["task"], case["collected_fields"])
    if next_field is None:
        # All fields collected → execute MCP
        mcp_result = execute_task(case["task"], case["collected_fields"])
        audit("CASE_COMPLETED", {"case_id":req.case_id})
        return {"status":"completed", "case":case, "mcp_result": mcp_result}
    else:
        return {"status":"awaiting_field", "next_field":next_field, "prompt":meta["prompt"]}

@app.get("/audit_log")
def get_audit_log():
    return AUDIT_LOG

@app.get("/cases")
def get_cases():
    return CASE_STORE
