# core/langgraph_runner.py

import os
import json
from typing import TypedDict, List, Any

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import SQLChatMessageHistory

from langgraph.graph import StateGraph, END

# -------------------------------------------------------
# Environment
# -------------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in environment.")

# -------------------------------------------------------
# Models
# -------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY,
)

embedder = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=GOOGLE_API_KEY,
)

# -------------------------------------------------------
# Vectorstore
# -------------------------------------------------------
vectorstore = FAISS.load_local(
    folder_path="faiss_index",
    embeddings=embedder,
    allow_dangerous_deserialization=True,
)

# MMR retrieval
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 24},
)

# -------------------------------------------------------
# Domain Guardrail Prompt
# -------------------------------------------------------
DOMAIN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Determine whether the user's question belongs to the Rogers Customer Support FAQ domain.\n"
            "Valid topics include: billing, payments, account management, mobility, SIM/eSIM, roaming, "
            "TV service, internet/Wi-Fi, technical troubleshooting, device issues, moving services, "
            "and customer support contact info.\n\n"
            "If the question fits these topics, answer ONLY: in-domain\n"
            "Otherwise answer ONLY: out-of-domain"
        ),
        ("user", "{query}")
    ]
)


def is_out_of_domain(query: str) -> bool:
    resp = llm.invoke(DOMAIN_PROMPT.format_messages(query=query))
    decision = (resp.content or "").strip().lower()
    return decision == "out-of-domain"


# -------------------------------------------------------
# RAG Prompts
# -------------------------------------------------------
QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer ONLY using the provided context. "
            "If answer is not contained in the context, reply EXACTLY:\n"
            "\"I could not find this information in the provided materials.\""
        ),
        MessagesPlaceholder("history"),
        (
            "user",
            "Context:\n{context}\n\n"
            "Question: {query}\n\n"
            "Answer clearly:"
        ),
    ]
)

RAG_CHAIN = QUESTION_PROMPT | llm | StrOutputParser()

REFLECTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Generate exactly TWO helpful follow-up questions based strictly on the answer and the context. "
            "Avoid unrelated topics. Return ONLY a JSON array of strings or objects."
        ),
        (
            "user",
            "Original Question:\n{query}\n\n"
            "Answer:\n{answer}\n\n"
            "Context:\n{context}"
        ),
    ]
)

REASONING_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Return ONLY 'sufficient' or 'insufficient'."),
        ("user", "Question:\n{query}\n\nContext:\n{context}")
    ]
)


def should_retrieve_again(query: str, context: str) -> bool:
    resp = llm.invoke(REASONING_PROMPT.format_messages(query=query, context=context))
    decision = (resp.content or "").strip().lower()
    return decision == "insufficient"


# -------------------------------------------------------
# Graph State
# -------------------------------------------------------
class GraphState(TypedDict, total=False):
    query: str
    context: str
    answer: str
    source_documents: List[Any]
    suggested_questions: List[str]
    needs_more: bool
    history: List[Any]


# -------------------------------------------------------
# Domain Check Node
# -------------------------------------------------------
def domain_check_node(state: GraphState) -> GraphState:
    query = state["query"]

    if is_out_of_domain(query):
        return {
            **state,
            "answer": (
                "I could not find this information in the provided materials. "
                "This assistant only answers topics related to Rogers billing, "
                "internet, TV service, mobility, device support, and technical troubleshooting."
            ),
            "context": "",
            "source_documents": [],
            "suggested_questions": [],
            "needs_more": False,
        }

    return state


# -------------------------------------------------------
# Retrieval
# -------------------------------------------------------
def retrieve_facts(state: GraphState) -> GraphState:
    query = state["query"]

    # MMR (no threshold filtering)
    results = vectorstore.similarity_search_with_score(query, k=8)
    docs = [doc for doc, score in results]

    context = "\n".join(doc.page_content for doc in docs)

    return {
        **state,
        "context": context,
        "source_documents": docs,
    }


def reason_node(state: GraphState) -> GraphState:
    if not state.get("context"):
        return {**state, "needs_more": False}

    needs_more = should_retrieve_again(state["query"], state["context"])
    return {**state, "needs_more": needs_more}


def retrieve_again(state: GraphState) -> GraphState:
    query = state["query"]
    exp = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 12, "fetch_k": 32},
    )
    extra = exp.invoke(query)

    combined_docs = list({doc.page_content: doc for doc in state["source_documents"] + extra}.values())
    new_context = "\n".join(doc.page_content for doc in combined_docs)

    return {
        **state,
        "context": new_context,
        "source_documents": combined_docs,
    }


# -------------------------------------------------------
# Answer Generation
# -------------------------------------------------------
def generate_answer(state: GraphState) -> GraphState:
    answer = RAG_CHAIN.invoke(
        {
            "query": state["query"],
            "context": state.get("context", ""),
            "history": state.get("history", []),
        }
    )
    return {**state, "answer": answer}


# -------------------------------------------------------
# Follow-up Suggestions (Robust, Crash-proof)
# -------------------------------------------------------
def reflect_and_suggest(state: GraphState) -> GraphState:
    """
    Generate exactly two follow-up questions.
    GUARANTEED: Only return strings → React never crashes.
    """
    if not state.get("context"):
        return {**state, "suggested_questions": []}

    response = llm.invoke(
        REFLECTION_PROMPT.format_messages(
            query=state["query"],
            answer=state.get("answer", ""),
            context=state.get("context", "")
        )
    )
    raw = (response.content or "").strip()

    # Remove code fences
    for token in ["```json", "```JSON", "```", "json", "JSON"]:
        raw = raw.replace(token, "").strip()

    # 1. Try JSON parsing
    try:
        parsed = json.loads(raw)

        # Case A: list of strings
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            q = parsed

        # Case B: list of objects → extract fields
        elif isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
            q = [
                x.get("question")
                or x.get("q")
                or x.get("follow_up")
                or x.get("text")
                or str(x)
                for x in parsed
            ]

        # Case C: single object
        elif isinstance(parsed, dict):
            q = [
                parsed.get("question")
                or parsed.get("q")
                or parsed.get("follow_up")
                or parsed.get("text")
                or str(parsed)
            ]

        else:
            q = [str(parsed)]

    except Exception:
        # 2. Fallback: split lines
        q = [line.strip(" -•") for line in raw.split("\n") if line.strip()]

    # Guarantee strings
    q = [x if isinstance(x, str) else str(x) for x in q]

    # Guarantee two questions
    if len(q) < 2:
        q += ["Can you explain more?", "What else should I know?"]

    q = q[:2]

    return {**state, "suggested_questions": q}


# -------------------------------------------------------
# Graph Construction
# -------------------------------------------------------
builder = StateGraph(GraphState)

builder.add_node("domain_check", domain_check_node)
builder.add_node("retrieve", retrieve_facts)
builder.add_node("reason", reason_node)
builder.add_node("retrieve_again", retrieve_again)
builder.add_node("respond", generate_answer)
builder.add_node("reflect", reflect_and_suggest)

builder.set_entry_point("domain_check")

# FIX: Exit early if out-of-domain
builder.add_conditional_edges(
    "domain_check",
    lambda s: "END" if s.get("answer") else "retrieve",
    {"retrieve": "retrieve", "END": END},
)

builder.add_edge("retrieve", "reason")

builder.add_conditional_edges(
    "reason",
    lambda s: "retrieve_again" if s.get("needs_more") else "respond",
    {"retrieve_again": "retrieve_again", "respond": "respond"},
)

builder.add_edge("retrieve_again", "respond")
builder.add_edge("respond", "reflect")
builder.add_edge("reflect", END)

graph = builder.compile()


# -------------------------------------------------------
# Memory Wrapper
# -------------------------------------------------------
def get_session_history(session_id: str) -> SQLChatMessageHistory:
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///memory.db",
    )


runnable_with_history = RunnableWithMessageHistory(
    graph,
    get_session_history,
    input_messages_key="query",
    history_messages_key="history",
    output_messages_key="answer",
)
