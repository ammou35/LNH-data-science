from pathlib import Path
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import json

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def download_documentation():
    DOCS_URL = "https://raw.githubusercontent.com/Zmalski/NHL-API-Reference/main/README.md"
    doc_path = DATA_DIR / "docs/nhl-api-reference.md"

    if not doc_path.exists():
        response = requests.get(DOCS_URL, timeout=30)
        response.raise_for_status()
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(response.text, encoding="utf-8")
    web_docs = doc_path.read_text(encoding="utf-8").split("# NHL Web API Documentation", 1)[1]
    web_docs = web_docs.split("# NHL Stats API Documentation", 1)[0]

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_text(web_docs)

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    vectors = embedder.encode(chunks, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])  # Dot product of unit vectors = cosine similarity.
    index.add(vectors)

    return chunks, embedder, index

def retrieve_sections(question, chunks, embedder, index):
    query_vector = embedder.encode([question], normalize_embeddings=True).astype("float32")
    scores, positions = index.search(query_vector, k=3)
    context = "\n\n".join(f"[{i}] {chunks[position]}" for i, position in enumerate(positions[0], 1))

    output_path = DATA_DIR / "docs/retrieved_sections.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    old_sections = json.loads(output_path.read_text()) if output_path.exists() else []
    old_sections.append({"question": question, "sections": context})
    output_path.write_text(json.dumps(old_sections, indent=2, ensure_ascii=False))

    return context