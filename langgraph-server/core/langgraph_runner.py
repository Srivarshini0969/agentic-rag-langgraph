# =======================================================
# core/langgraph_runner.py
# Agentic RAG using LangGraph + Gemini + FAISS
# =======================================================

import os
from typing import TypedDict, List, Any

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import SQLChatMessageHistory

from langgraph.graph import StateGraph, END


# =======================================================
# ENVIRONMENT
# =======================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in environment.")


# =======================================================
# GEMINI MODEL
# =======================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY,
)


# =======================================================
# HELPER: GEMINI CONTENT -> STRING
# =======================================================

def content_to_text(content: Any) -> str:
    """
    Converts Gemini output into a normal Python string.
    """

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:

            if isinstance(item, dict):
                text = item.get("text")

                if text is not None:
                    parts.append(str(text))
                else:
                    parts.append(str(item))

            else:
                parts.append(str(item))

        return " ".join(parts)

    return str(content)


# =======================================================
# EMBEDDINGS
# =======================================================

embedder = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=768,
    google_api_key=GOOGLE_API_KEY,
)


# =======================================================
# LOAD FAISS VECTORSTORE
# =======================================================

vectorstore = FAISS.load_local(
    folder_path="faiss_index",
    embeddings=embedder,
    allow_dangerous_deserialization=True,
)


# =======================================================
# RETRIEVER
# =======================================================

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 8,
        "fetch_k": 24,
    },
)


# =======================================================
# DOMAIN CHECK PROMPT
# =======================================================

DOMAIN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Determine whether the user's question belongs to the Rogers Customer
Support FAQ domain.

Valid topics include:
- billing
- payments
- account management
- mobility
- SIM/eSIM
- roaming
- TV service
- internet/Wi-Fi
- technical troubleshooting
- device issues
- moving services
- customer support contact information

If the question fits these topics, answer ONLY:

in-domain

Otherwise answer ONLY:

out-of-domain
""",
        ),
        ("user", "{query}"),
    ]
)


# =======================================================
# DOMAIN CHECK
# =======================================================

def is_out_of_domain(query: str) -> bool:

    response = llm.invoke(
        DOMAIN_PROMPT.format_messages(query=query)
    )

    decision = content_to_text(response.content).strip().lower()

    return decision == "out-of-domain"


# =======================================================
# RAG QUESTION PROMPT
# =======================================================

QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a Rogers Customer Support assistant.

Answer ONLY using the provided context.

If the answer is not contained in the context, reply EXACTLY:

I could not find this information in the provided materials.

Do not invent information.
Do not use outside knowledge.
""",
        ),

        MessagesPlaceholder(variable_name="history"),

        (
            "user",
            """
Context:
{context}

Question:
{query}

Answer clearly:
""",
        ),
    ]
)


# =======================================================
# RAG CHAIN
# =======================================================

RAG_CHAIN = QUESTION_PROMPT | llm | StrOutputParser()


# =======================================================
# REASONING PROMPT
# =======================================================

REASONING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Decide whether the provided context contains enough information
to answer the user's question.

Return ONLY one of:

sufficient

or

insufficient
""",
        ),
        (
            "user",
            """
Question:
{query}

Context:
{context}
""",
        ),
    ]
)


# =======================================================
# REASONING
# =======================================================

def should_retrieve_again(
    query: str,
    context: str,
) -> bool:

    response = llm.invoke(
        REASONING_PROMPT.format_messages(
            query=query,
            context=context,
        )
    )

    decision = content_to_text(response.content).strip().lower()

    return decision == "insufficient"


# =======================================================
# GRAPH STATE
# =======================================================

class GraphState(TypedDict, total=False):

    query: str
    context: str
    answer: str

    source_documents: List[Any]

    suggested_questions: List[str]

    needs_more: bool

    history: List[BaseMessage]


# =======================================================
# DOMAIN CHECK NODE
# =======================================================

def domain_check_node(
    state: GraphState,
) -> GraphState:

    query = state["query"]

    if is_out_of_domain(query):

        return {
            **state,

            "answer": (
                "I could not find this information in the provided materials. "
                "This assistant only answers topics related to Rogers billing, "
                "internet, TV service, mobility, device support, and technical "
                "troubleshooting."
            ),

            "context": "",

            "source_documents": [],

            "suggested_questions": [],

            "needs_more": False,
        }

    return state


# =======================================================
# FIRST RETRIEVAL
# =======================================================

def retrieve_facts(
    state: GraphState,
) -> GraphState:

    query = state["query"]

    results = vectorstore.similarity_search_with_score(
        query,
        k=8,
    )

    docs = [
        doc
        for doc, score in results
    ]

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return {
        **state,
        "context": context,
        "source_documents": docs,
    }


# =======================================================
# REASONING NODE
# =======================================================

def reason_node(
    state: GraphState,
) -> GraphState:

    if not state.get("context"):

        return {
            **state,
            "needs_more": False,
        }

    needs_more = should_retrieve_again(
        state["query"],
        state["context"],
    )

    return {
        **state,
        "needs_more": needs_more,
    }


# =======================================================
# SECOND RETRIEVAL
# =======================================================

def retrieve_again(
    state: GraphState,
) -> GraphState:

    query = state["query"]

    expanded_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 12,
            "fetch_k": 32,
        },
    )

    extra_docs = expanded_retriever.invoke(query)

    previous_docs = state.get(
        "source_documents",
        [],
    )

    all_docs = previous_docs + extra_docs

    unique_docs = {}

    for doc in all_docs:
        unique_docs[doc.page_content] = doc

    combined_docs = list(
        unique_docs.values()
    )

    new_context = "\n\n".join(
        doc.page_content
        for doc in combined_docs
    )

    return {
        **state,
        "context": new_context,
        "source_documents": combined_docs,
    }


# =======================================================
# ANSWER GENERATION
# =======================================================

def generate_answer(
    state: GraphState,
) -> GraphState:

    answer = RAG_CHAIN.invoke(
        {
            "query": state["query"],
            "context": state.get("context", ""),
            "history": state.get("history", []),
        }
    )

    answer = content_to_text(answer).strip()

    return {
        **state,
        "answer": answer,
    }


# =======================================================
# FOLLOW-UP QUESTIONS
# =======================================================
# These are predefined so we do NOT make another Gemini
# request just to generate follow-up questions.
# =======================================================

def reflect_and_suggest(
    state: GraphState,
) -> GraphState:

    if not state.get("context"):

        return {
            **state,
            "suggested_questions": [],
        }

    return {
        **state,
        "suggested_questions": [
            "Can you explain this in more detail?",
            "What else should I know about this?",
        ],
    }


# =======================================================
# LANGGRAPH CONSTRUCTION
# =======================================================

builder = StateGraph(GraphState)


builder.add_node(
    "domain_check",
    domain_check_node,
)

builder.add_node(
    "retrieve",
    retrieve_facts,
)

builder.add_node(
    "reason",
    reason_node,
)

builder.add_node(
    "retrieve_again",
    retrieve_again,
)

builder.add_node(
    "respond",
    generate_answer,
)

builder.add_node(
    "reflect",
    reflect_and_suggest,
)


# =======================================================
# ENTRY POINT
# =======================================================

builder.set_entry_point(
    "domain_check"
)


# =======================================================
# DOMAIN CHECK -> RETRIEVAL OR END
# =======================================================

builder.add_conditional_edges(
    "domain_check",

    lambda state:
        "END"
        if state.get("answer")
        else "retrieve",

    {
        "retrieve": "retrieve",
        "END": END,
    },
)


# =======================================================
# RETRIEVAL -> REASONING
# =======================================================

builder.add_edge(
    "retrieve",
    "reason",
)


# =======================================================
# REASONING -> SECOND RETRIEVAL OR ANSWER
# =======================================================

builder.add_conditional_edges(
    "reason",

    lambda state:
        "retrieve_again"
        if state.get("needs_more")
        else "respond",

    {
        "retrieve_again": "retrieve_again",
        "respond": "respond",
    },
)


# =======================================================
# SECOND RETRIEVAL -> ANSWER
# =======================================================

builder.add_edge(
    "retrieve_again",
    "respond",
)


# =======================================================
# ANSWER -> REFLECTION
# =======================================================

builder.add_edge(
    "respond",
    "reflect",
)


# =======================================================
# REFLECTION -> END
# =======================================================

builder.add_edge(
    "reflect",
    END,
)


# =======================================================
# COMPILE GRAPH
# =======================================================

graph = builder.compile()


# =======================================================
# SESSION HISTORY
# =======================================================

def get_session_history(
    session_id: str,
) -> SQLChatMessageHistory:

    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///memory.db",
    )


# =======================================================
# MAIN GRAPH RUNNER
# =======================================================

def run_agentic_rag(
    query: str,
    session_id: str = "default",
) -> GraphState:

    # Load previous conversation
    history_store = get_session_history(
        session_id
    )

    history = history_store.messages

    # Run LangGraph
    result = graph.invoke(
        {
            "query": query,
            "history": history,
        }
    )

    # Store current user question
    history_store.add_user_message(query)

    # Store generated answer
    answer = result.get("answer", "")

    if answer:
        history_store.add_ai_message(answer)

    return result