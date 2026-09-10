# app.py
from flask import Flask
from flask_cors import CORS
from routes.rag_routes import rag_api  # note the routes/ package

app = Flask(__name__)
CORS(app)

app.register_blueprint(rag_api)

if __name__ == "__main__":
    app.run(debug=True)
