from typing import Optional

import json

import core, mcp

# ------------------------------------------------------------
# STEP 1: ASK LLM TO EXTRACT THE NAME
# ------------------------------------------------------------
def extract_name(question: str) -> Optional[str]:
    """
    Use the LLM to extract the customer name from the natural language question.
    Deterministic execution: LLM gives a parameter, not a plan.
    """
    system = (
        "You extract the CUSTOMER NAME from user questions about orders.\n"
        "If no name is clearly mentioned, respond with just: null\n"
        "Otherwise, return only the name string (e.g. Emre Akkus)."
    )
    name = core.ollama_chat(system,question).strip()
    if name.lower() == "null" or not name:
        return None
    return name

# ------------------------------------------------------------
# STEP 3: HYBRID PIPELINE
# ------------------------------------------------------------
def hybrid_rag_mcp(question: str, top_k):
    print(f"\n🧠 Question: {question}")

    # Extract entity using the LLM (no planning, just interpretation)
    name = extract_name(question)
    print(f"📍 Extracted name: {name}")

    structured = ""
    if not name:
        print("⚠️ No customer name found in the question.")
    else:
        structured = mcp.call("sql.order_lookup", {"name": name, "limit": 5})

    unstructured = rag_search(question, top_k)

    # Combine both contexts for final reasoning
    system = "You are a helpful assistant combining structured and unstructured information."
    prompt = f"""
Structured data (from MCP):
{json.dumps(structured, indent=2)}

Unstructured context:
{unstructured}

Question:
{question}

Answer concisely using both sources.
"""
    answer = core.ollama_chat(system, prompt)
    print("\n💬 Final Answer:\n", answer)

    return answer

def rag_search(question, top_k):
    hits = core.retrieve(question, top_k)
    ctx = core.build_context(hits)
    return ctx