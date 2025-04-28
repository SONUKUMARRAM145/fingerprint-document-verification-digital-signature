import hashlib
import json
import os

def compute_hash(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def add_original_document(file_path):
    if not os.path.exists("original_documents.json"):
        with open("original_documents.json", "w") as f:
            json.dump([], f, indent=4)

    with open("original_documents.json", "r+") as f:
        data = json.load(f)
        document_name = os.path.basename(file_path)
        document_hash = compute_hash(file_path)

        # Check if already exists
        for doc in data:
            if doc["document_name"] == document_name:
                print(f"Document '{document_name}' already exists!")
                return

        # Add new document
        data.append({
            "document_name": document_name,
            "document_hash": document_hash
        })
        f.seek(0)
        json.dump(data, f, indent=4)
        print(f"✅ Successfully added: {document_name}")

# Example usage:
file_path = input("Enter full path of document to add: ").strip().strip('"')

add_original_document(file_path)
