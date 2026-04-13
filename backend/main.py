import os
import json as json_module
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rag import get_rag_system, load_user_memory

app = FastAPI(title="IT Support RAG API")

@app.on_event("startup")
async def startup_event():
    print("Initializing RAG system and loading Firestore chunks on startup...")
    get_rag_system()

# Add CORS to allow frontend talking to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    user_id: str = "default"
    thread_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: list = []
    is_from_rag: bool = False

class AddDocumentRequest(BaseModel):
    source: str
    content: str
    department: str = ""

class DocumentResponse(BaseModel):
    source: str
    content: str
    chunk_index: int | None = None
    total_chunks: int | None = None
    department: str = ""

class DocumentsListResponse(BaseModel):
    documents: list

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY environment variable is not set. Please set it before running the server.")
        
    try:
        rag = get_rag_system()
        answer, sources, is_from_rag = rag.ask(
            request.query,
            user_id=request.user_id,
            thread_id=request.thread_id,
        )
        return ChatResponse(answer=answer, sources=sources, is_from_rag=is_from_rag)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """SSE streaming endpoint for chat responses."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY environment variable is not set.")

    async def event_generator():
        try:
            rag = get_rag_system()
            async for event in rag.ask_stream(
                request.query,
                user_id=request.user_id,
                thread_id=request.thread_id,
            ):
                yield f"data: {json_module.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json_module.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.get("/chat/history/{thread_id}")
async def get_chat_history(thread_id: str):
    try:
        rag = get_rag_system()
        history = rag.get_chat_history(thread_id)
        return {"thread_id": thread_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/memory/{user_id}")
async def get_user_memory(user_id: str):
    try:
        facts = load_user_memory(user_id)
        return {"user_id": user_id, "facts": facts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/add-document")
async def add_document(request: AddDocumentRequest):
    try:
        rag = get_rag_system()
        rag.add_document(request.source, request.content, department=request.department)
        return {"status": "success", "message": f"Document {request.source} added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/admin/delete-document/{source}")
async def delete_document(source: str):
    try:
        rag = get_rag_system()
        rag.delete_document(source)
        return {"status": "success", "message": f"Document {source} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/documents", response_model=DocumentsListResponse)
async def get_documents():
    try:
        rag = get_rag_system()
        documents = rag.get_all_documents()
        return DocumentsListResponse(documents=documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/upload-file")
async def upload_file(file: UploadFile = File(...), department: str = Form("")):
    try:
        rag = get_rag_system()
        content, source = await rag.parse_uploaded_file(file)
        rag.add_document(source, content, department=department)
        return {"status": "success", "message": f"File {file.filename} processed and added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
