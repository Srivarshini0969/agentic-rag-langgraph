from flask import Blueprint, request, jsonify

from core.langgraph_runner import run_agentic_rag

rag_api = Blueprint("rag_api", __name__)


@rag_api.route("/agentic-rag", methods=["POST"])
def agentic_rag():
    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "error": "Request body must be valid JSON."
            }), 400

        query = data.get("query")
        session_id = data.get("session_id", "default")

        if not query:
            return jsonify({
                "error": "Missing 'query' in request body."
            }), 400

        if not isinstance(query, str):
            query = str(query)

        if not isinstance(session_id, str):
            session_id = str(session_id)

        # Run Agentic RAG
        result = run_agentic_rag(
            query=query,
            session_id=session_id
        )

        # Convert retrieved documents into simple strings
        facts = []

        for doc in result.get("source_documents", []):
            if hasattr(doc, "page_content"):
                facts.append(doc.page_content)

        return jsonify({
            "answer": result.get("answer", ""),
            "facts": facts,
            "suggested_questions": result.get(
                "suggested_questions",
                []
            ),
            "session_id": session_id
        }), 200

    except Exception as e:
        print("\n==============================")
        print("❌ AGENTIC RAG ERROR")
        print("==============================")
        print(type(e).__name__)
        print(str(e))
        print("==============================\n")

        return jsonify({
            "error": "Agentic RAG processing failed.",
            "details": str(e),
            "type": type(e).__name__
        }), 500