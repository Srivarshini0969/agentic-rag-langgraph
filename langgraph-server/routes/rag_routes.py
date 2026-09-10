# routes/rag_routes.py

import os
from flask import Blueprint, request, jsonify
from dotenv import load_dotenv

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import AIMessage

from core.langgraph_runner import runnable_with_history

load_dotenv()

rag_api = Blueprint("rag_api", __name__)


@rag_api.route("/agentic-rag", methods=["POST"])
def rag_handler():
    data = request.get_json() or {}

    # FIX: strip() instead of trim()
    query = (data.get("query") or "").strip()
    session_id = data.get("session_id", "default-session")

    if not query:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    try:
        result = runnable_with_history.invoke(
            {"query": query},
            config={"configurable": {"session_id": session_id}},
        )

        return jsonify(
            {
                "query": result.get("query"),
                "facts": result.get("context", ""),
                "answer": result.get("answer", ""),
                "suggestedQuestions": result.get("suggested_questions", []),
            }
        )

    except Exception as e:
        print("[Agentic RAG Error]", str(e))
        return jsonify({"error": "Agentic RAG processing failed."}), 500


@rag_api.route("/reset-session", methods=["DELETE"])
def reset_session():
    session_id = request.args.get("session_id", "default-session")

    try:
        history = SQLChatMessageHistory(
            session_id=session_id,
            connection="sqlite:///memory.db",
        )
        history.clear()
        return jsonify({"message": f"Session '{session_id}' reset successfully."})
    except Exception as e:
        print("[Session Reset Error]", str(e))
        return jsonify({"error": "Failed to reset session."}), 500
