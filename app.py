from flask import Flask, request, jsonify
from flask_cors import CORS
from bedrock import bedrock_answer

from search import FaissTfidfSearch

# ✅ Create app FIRST (before using @app.get / @app.post decorators)
app = Flask(__name__, static_folder="public", static_url_path="")
CORS(app)

# ✅ Serve the frontend
@app.get("/")
def home():
    return app.send_static_file("index.html")

# ✅ Build the FAISS index on startup
engine = FaissTfidfSearch()
engine.fit()  # reads data/menu.json and data/faqs.json


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/search")
def search():
    body = request.get_json(force=True) or {}
    query = (body.get("query") or "").strip()
    k = int(body.get("k") or 5)
    filter_type = body.get("type")  # "menu" or "faq" or None

    if not query:
        return jsonify({"error": "query is required"}), 400

    results = engine.search(query=query, k=k, filter_type=filter_type)
    return jsonify({"query": query, "k": k, "results": results})


@app.post("/ask")
def ask():
    """
    Returns best FAQ + best menu items for a natural question.
    """
    body = request.get_json(force=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400

    faq = engine.search(query, k=3, filter_type="faq")
    menu = engine.search(query, k=5, filter_type="menu")

    # Light “smart” filtering (optional but nice)
    qlow = query.lower()
    if "vegetarian" in qlow or "veg" in qlow:
        menu = [m for m in menu if "vegetarian" in (m["meta"].get("tags") or [])]
    if "gluten" in qlow:
        menu = [m for m in menu if "gluten-free" in (m["meta"].get("tags") or [])]

    return jsonify({
        "query": query,
        "top_faq": faq[:1],
        "menu_suggestions": menu[:5]
    })
@app.post("/ask_llm")
def ask_llm():
    try:
        body = request.get_json(silent=True) or {}
        query = (body.get("query") or "").strip()
        if not query:
            return jsonify({"error": "query is required"}), 400

        faq = engine.search(query, k=3, filter_type="faq")
        menu = engine.search(query, k=5, filter_type="menu")

        faq_text = "\n".join([f"FAQ: {x.get('meta',{}).get('question','')} -> {x.get('meta',{}).get('answer','')}" for x in faq])
        menu_text = "\n".join([f"Menu: {m.get('meta',{}).get('name','')} (${m.get('meta',{}).get('price','')}): {m.get('meta',{}).get('description','')}" for m in menu])

        context = f"{faq_text}\n\n{menu_text}"
        answer = bedrock_answer(query, context)

        return jsonify({"query": query, "answer": answer, "faq": faq, "menu": menu})

    except Exception as e:
        print("ERROR in /ask_llm:", repr(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Run Flask
    app.run(port=5000, debug=True)

