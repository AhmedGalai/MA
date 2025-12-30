from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from rich import print

app = FastAPI(title="VisionOS Sample API")

class Option(BaseModel):
    id: str
    name: str

OPTIONS: List[Option] = [
    Option(id="opt-1", name="First Option"),
    Option(id="opt-2", name="Second Option"),
    Option(id="opt-3", name="Third Option"),
]

@app.get("/options", response_model=List[Option])
async def get_options():
    return OPTIONS

class SubmitRequest(BaseModel):
    username: str
    password: str
    notes: str
    volume: float
    count: int
    enabled: bool
    selectedOptionId: Optional[str] = None
    dateISO8601: str

class SubmitResponse(BaseModel):
    ok: bool
    message: str
    echo: SubmitRequest | None = None

@app.post("/submit", response_model=SubmitResponse)
async def submit(payload: SubmitRequest):
    msg = f"Received form for user={payload.username}, option={payload.selectedOptionId or 'none'}"
    print(payload)
    return SubmitResponse(ok=True, message=msg, echo=payload)

class UploadResponse(BaseModel):
    ok: bool
    filename: Optional[str] = None
    bytes: Optional[int] = None
    message: str

@app.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    note: str = Form(default="")
):
    data = await file.read()
    print(data)
    info = f"note={note}" if note else "no note"
    return UploadResponse(
        ok=True,
        filename=file.filename,
        bytes=len(data),
        message=f"Uploaded {file.filename} ({len(data)} bytes); {info}"
    )

@app.get("/")
async def root():
    return JSONResponse({"ok": True, "message": "VisionOS Sample API running"})
