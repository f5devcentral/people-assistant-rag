# People Assistant

**People Assistant** is a demo application showcasing a Retrieval-Augmented Generation (RAG) architecture.  
It combines a **Python FastAPI backend** with **ChromaDB** for vector search and a **Node.js/Express frontend** serving a simple UI.

The app demonstrates:
- Querying a people database by job titles and names.
- Using a large language model (LLM) with context retrieved from ChromaDB.
- Basic **auth with JWT** (login flow in the UI + API tokens for cURL).
- A secure separation of concerns: frontend → Node.js proxy → FastAPI backend.

---

## 🚀 Architecture

[ Browser UI ]  –>  [ Express / Node.js proxy ]  –>  [ FastAPI backend ]
–>  [ ChromaDB vector store ]
–>  [ LLM via CalypsoAI ]

---

## ✨ Features
- Interactive web UI (chat-like assistant).
- Retrieval from `people.json` collection.
- Prevents unauthorized API access using JWTs.
- Simple login (admin/root credentials).
- Option to generate **non-expiring API tokens** for cURL requests.
- Status indicator (Idle/Working) in the UI.

---

## 🛠️ Setup

### 1. Clone repo
```bash
git clone https://gitlab.com/YOUR_GROUP/people-assistant.git
cd people-assistant
```

### 2. Backend (FastAPI)

Create virtual environment
```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
```
Install dependencies
```bash
pip install -r requirements.txt
```

Create a `.env` with your API keys so the backend can embed and scan content:
```bash
cd backend
cat <<'EOF' > .env
OPENAI_EMBED_API_KEY=sk-your-openai-embeddings-key
OPENAI_LLM_API_KEY=sk-your-openai-llm-key
CAI_API_KEY=your-calypso-key
EOF
```

`OPENAI_EMBED_API_KEY` powers the Chroma vector store: `VectorStore` calls OpenAI’s embedding endpoint for every document and query, so embeddings (and therefore vector lookups) fail without that key.  
`OPENAI_LLM_API_KEY` powers the chat generation call (`gpt-4o-mini` by default) before Calypso scanning.  
`OPENAI_API_KEY` remains supported as a fallback for either path if the dedicated key is not set.  
`VALIDATE_OPENAI_MODEL` (default `true`) makes a startup-time OpenAI models-list style validation on first request to ensure the configured `LLM_PROVIDER` is available for the LLM key.  
`CAI_API_KEY` is required by `calypso_client.send_text_to_calypso` to run CalypsoAI scans for the user prompt and model response as separate calls; without it, the backend blocks requests with “Missing CAI_API_KEY”.

Run backend
```bash
uvicorn main:app --reload --port 8000
```

### 3. Frontend (Node.js / Express)

Install dependencies
```bash
cd ../frontend
npm install
```
Run frontend
```bash
npm start
```
By default:
	•	Frontend: http://localhost:3000
	•	Backend: http://localhost:8000

---

## 🔑 Authentication
    - UI login: Username admin, Password root. Generates a short-lived JWT stored in localStorage.
	- API token (for curl): From the login page or index UI, click Generate API Token.
This creates a non-expiring JWT for testing.

Example: API token with curl
```bash
curl -X POST http://localhost:3000/ask \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -d '{"question": "Who are the people in the company?", "top_k": 5}'
```

---

## 🧩 Project Structure
    📂 people-assistant/
    ├── 🐍 backend/                # FastAPI + ChromaDB
    │   ├── main.py
    │   ├── vector_store.py
    │   ├── requirements.txt
    │   └── data/
    │       └── people.json
    ├── 🌐 frontend/               # Express + static UI
    │   ├── public/
    │   │   ├── index.html
    │   │   └── login.html
    │   ├── package.json
    │   └── server.js
    └── 📄 README.md

---

## 🔮 Roadmap
	•	Extend collections (finance, superheroes, etc.).
	•	Add role-based access controls (RBAC).
	•	Deploy on Render with environment configs.
	•	Add monitoring (New Relic integration).

⸻

📄 License

MIT — use freely for demos and learning.

---

👉 Do you want me to also generate the **requirements.txt** and **package.json** starter files** (so your repo is immediately runnable when you push), or do you already have those locally?
