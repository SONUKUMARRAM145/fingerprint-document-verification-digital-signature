from fastapi import FastAPI, UploadFile, Form, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import shutil
import json

from auth import authenticate_fingerprint
from crypto_utils import sign_document, verify_signature, hash_document

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

if not os.path.exists("uploads"):
    os.makedirs("uploads")

if not os.path.exists("backend_fingerprints"):
    os.makedirs("backend_fingerprints")

if not os.path.exists("database.json"):
    with open("database.json", "w") as f:
        json.dump([], f)

if not os.path.exists("original_documents.json"):
    with open("original_documents.json", "w") as f:
        json.dump([], f)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_fingerprint/", response_class=HTMLResponse)
async def upload_fingerprint(request: Request, file: UploadFile = File(...)):
    filepath = f"uploads/{file.filename}"
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    user_id, score = authenticate_fingerprint(filepath)
    os.remove(filepath)

    if score > 0.92:
        return templates.TemplateResponse("upload_document.html", {"request": request, "user_id": user_id})
    else:
        return templates.TemplateResponse("failure.html", {"request": request})

@app.post("/upload_document/", response_class=HTMLResponse)
async def upload_document(request: Request, file: UploadFile = File(...), user_id: str = Form(...)):
    filepath = f"uploads/{file.filename}"
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Step 1: Hash uploaded document
    uploaded_doc_hash = hash_document(filepath)

    # Step 2: Load original known document hashes
    with open("original_documents.json", "r") as f:
        original_docs = json.load(f)

    match_found = False
    for original in original_docs:
        if original["document_name"] == file.filename:
            # Compare stored hash with uploaded document hash
            if bytes.fromhex(original["document_hash"]) == uploaded_doc_hash:
                match_found = True
                break

    os.remove(filepath)  # Delete uploaded file after checking

    if match_found:
        # Step 3: Sign document hash if it matches
        signed_hash = sign_document(uploaded_doc_hash)

        # Step 4: Save to database
        with open("database.json", "r+") as f:
            data = json.load(f)
            data.append({
                "user_id": user_id,
                "document_name": file.filename,
                "document_hash": uploaded_doc_hash.hex(),
                "signature": signed_hash.hex()
            })
            f.seek(0)
            json.dump(data, f, indent=4)

        return templates.TemplateResponse("success.html", {"request": request})
    else:
        return templates.TemplateResponse("failure.html", {"request": request})
