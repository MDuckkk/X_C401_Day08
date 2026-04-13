# RAG IT Support Assistant

This repository contains two parts:

- `backend/`: FastAPI + LangGraph RAG API with Firestore persistence.
- `frontend/`: Vite + React chat UI and admin page.

## Prerequisites

- Python 3.11 or newer.
- Node.js 18 or newer.
- A Firebase project with Firestore enabled.
- An OpenAI API key.

## Project Structure

- `backend/main.py`: FastAPI entry point.
- `backend/rag.py`: RAG logic and retrieval.
- `backend/firestore_service.py`: Firestore persistence layer.
- `frontend/src/App.jsx`: Chat UI.
- `frontend/src/AdminPage.jsx`: Admin UI for documents.

## Backend Setup

1. Open a terminal in the project root.
2. Create and activate a Python environment.

```powershell
cd c:\Users\tuana\rag\rag\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install backend dependencies.

```powershell
pip install -r requirements.txt
```

4. Create a `backend/.env` file with your API key.

```env
OPENAI_API_KEY=your_openai_api_key
GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\firebase-service-account.json
```

5. Start the backend server.

```powershell
python main.py
```

The backend loads environment variables with `python-dotenv`, so keeping the `.env` file inside `backend/` is the simplest option.

## Firebase Setup For Backend

The backend uses the Firebase Admin SDK to store:

- document chunks
- chat history
- user memories

### 1. Create or select a Firebase project

Go to the Firebase console, create a project, and enable Firestore.

### 2. Create a service account key

1. Open Project settings in Firebase.
2. Go to the Service accounts tab.
3. Generate a new private key.
4. Save the JSON file somewhere secure on your machine.

### 3. Set the service account path

Set `GOOGLE_APPLICATION_CREDENTIALS` to the full path of that JSON file. The backend reads that variable in `backend/firestore_service.py`.

Example for Windows PowerShell:

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\secure\firebase-service-account.json"
```

If you want this to persist across terminal sessions, place it in `backend/.env` as shown above.

### 4. Required Firestore collections

The backend will create and use these collections automatically:

- `document_chunks`
- `chat_history`
- `user_memories`

## Frontend Setup

1. Open a second terminal in the project root.
2. Install frontend dependencies.

```powershell
cd c:/frontend
npm install
```

3. Start the Vite development server.

```powershell
npm run dev
```

4. Open the local URL shown by Vite, usually `http://localhost:5173`.

## Frontend Environment

There is no frontend `.env` file required in the current codebase.

The React app calls the backend directly at `http://127.0.0.1:8000`, so the backend must be running locally before chat and admin features will work.

If you later move the backend to a different host or port, update the `fetch` URLs in:

- `frontend/src/App.jsx`
- `frontend/src/AdminPage.jsx`

## Run The Full App

1. Start the backend from `backend/`.
2. Start the frontend from `frontend/`.
3. Open the frontend in your browser and try a chat message.

## Troubleshooting

- If the backend returns a 500 error about `OPENAI_API_KEY`, check `backend/.env` and restart the server.
- If Firestore access fails, confirm the service account JSON path is correct and that Firestore is enabled in the Firebase project.
- If the frontend cannot reach the backend, make sure the FastAPI server is running on `127.0.0.1:8000`.

## Notes

- The backend loads RAG data from Firestore on startup.
- The admin page lets you add, delete, and upload documents through the backend API.
