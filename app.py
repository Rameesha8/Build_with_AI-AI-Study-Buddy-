import io
import json
import os
import re
import textwrap
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
import pdfplumber
from PyPDF2 import PdfReader
from fastapi import Cookie, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from google import genai
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

load_dotenv()

APP_TITLE = "AI Study Buddy"
APP_VERSION = "1.0.0"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
DEFAULT_ANSWER_MODEL = "llama-3.3-70b-versatile" if LLM_PROVIDER == "groq" else "gemini-2.5-flash"
ANSWER_MODEL_NAME = os.getenv("ANSWER_MODEL_NAME", DEFAULT_ANSWER_MODEL)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "4"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

BASE_DIR = Path(__file__).parent
INDEX_DIR = Path(os.getenv("INDEX_DIR", str(BASE_DIR / "artifacts")))
INDEX_FILE = INDEX_DIR / "study_buddy.index"
METADATA_FILE = INDEX_DIR / "metadata.json"

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
gemini_api_key = os.getenv("GEMINI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
llm_client = None
if LLM_PROVIDER == "groq" and groq_api_key:
    llm_client = Groq(api_key=groq_api_key)
elif LLM_PROVIDER == "gemini" and gemini_api_key:
    llm_client = genai.Client(api_key=gemini_api_key)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Study Buddy</title>
  <style>
    :root {
      --bg: #f7f3ea;
      --ink: #172120;
      --muted: #5f6b67;
      --panel: rgba(255,255,255,0.84);
      --line: rgba(23,33,32,0.12);
      --accent: #0d7a68;
      --accent-dark: #085649;
      --danger: #a53f2d;
      --shadow: 0 20px 60px rgba(26,35,33,0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, 'Times New Roman', serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(13,122,104,0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(214,180,108,0.22), transparent 26%),
        linear-gradient(180deg, #fbf8f2 0%, var(--bg) 100%);
    }
    .shell { max-width: 1120px; margin: 0 auto; padding: 40px 20px 72px; }
    h1 { margin: 0; font-size: clamp(2.8rem, 6vw, 4.9rem); line-height: 0.95; }
    .eyebrow { margin: 0 0 12px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--accent-dark); font-size: 0.95rem; }
    .lede { max-width: 760px; color: var(--muted); font-size: 1.08rem; line-height: 1.7; margin: 18px 0 28px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 24px; padding: 22px; box-shadow: var(--shadow); }
    .panel h2 { margin-top: 0; font-size: 1.2rem; }
    form { display: grid; gap: 12px; }
    input[type='file'], textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      background: rgba(255,255,255,0.92);
      font: inherit;
      color: var(--ink);
    }
    textarea { resize: vertical; min-height: 120px; }
    button {
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      background: var(--accent);
      color: white;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: var(--accent-dark); }
    button.secondary { background: transparent; color: var(--accent-dark); border: 1px solid var(--accent-dark); }
    .status { color: var(--muted); }
    .status.error { color: var(--danger); }
    .output { white-space: pre-wrap; margin: 0; line-height: 1.75; font-family: Georgia, 'Times New Roman', serif; min-height: 120px; font-size: 1.02rem; }
    .matches { display: grid; gap: 12px; }
    .match { border: 1px solid var(--line); border-radius: 18px; padding: 14px; background: rgba(255,255,255,0.7); }
    .meta { color: var(--accent-dark); font-size: 0.92rem; margin-bottom: 6px; }
    @media (max-width: 640px) { .shell { padding: 24px 14px 40px; } }
  </style>
</head>
<body>
  <main class="shell">
    <p class="eyebrow">Workshop Demo</p>
    <h1>AI Study Buddy</h1>
    <p class="lede">Upload notes or a PDF, then ask questions grounded only in that material. This version keeps the frontend directly inside the FastAPI app.</p>
    <div class="grid">
      <section class="panel">
        <h2>1. Upload Study Material</h2>
        <form id="upload-form">
          <input id="file-input" name="file" type="file" accept=".pdf,.txt" required>
          <button type="submit">Upload</button>
        </form>
        <p id="upload-status" class="status">No document uploaded yet.</p>
      </section>
      <section class="panel">
        <h2>2. Ask a Question</h2>
        <p id="mode-status" class="status">Loading app mode...</p>
        <form id="ask-form">
          <textarea id="question-input" placeholder="What are the main causes of inflation?" required></textarea>
          <button type="submit">Ask</button>
        </form>
        <button id="reset-button" class="secondary" type="button">Reset Session</button>
      </section>
    </div>
    <div class="grid" style="margin-top: 18px;">
      <section class="panel">
        <h2>Answer</h2>
        <pre id="answer-output" class="output">Your answer will appear here.</pre>
      </section>
      <section class="panel">
        <h2>Retrieved Chunks</h2>
        <div id="matches-output" class="matches">
          <p class="status">Relevant chunks will appear here after you ask a question.</p>
        </div>
      </section>
    </div>
  </main>
  <script>
    const uploadForm = document.getElementById('upload-form');
    const askForm = document.getElementById('ask-form');
    const resetButton = document.getElementById('reset-button');
    const uploadStatus = document.getElementById('upload-status');
    const modeStatus = document.getElementById('mode-status');
    const answerOutput = document.getElementById('answer-output');
    const matchesOutput = document.getElementById('matches-output');
    let askEndpoint = '/ask';
    function renderMatches(matches) {
      if (!matches || !matches.length) {
        matchesOutput.innerHTML = '<p class="status">No chunks found.</p>';
        return;
      }
      matchesOutput.innerHTML = matches.map((item, index) => `
        <article class="match">
          <p class="meta">Match ${index + 1} | ${item.source} | score ${item.score}</p>
          <p>${item.chunk_preview}</p>
        </article>
      `).join('');
    }
    async function loadMode() {
      try {
        const response = await fetch('/knowledge-base');
        const data = await response.json();
        if (response.ok && data.loaded) {
          askEndpoint = '/ask-saved';
          modeStatus.textContent = `Saved knowledge base loaded with ${data.index_stats.documents} document(s).`;
        } else {
          askEndpoint = '/ask';
          modeStatus.textContent = 'Interactive upload mode active.';
        }
      } catch (error) {
        askEndpoint = '/ask';
        modeStatus.textContent = 'Interactive upload mode active.';
      }
    }
    uploadForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      const file = document.getElementById('file-input').files[0];
      if (!file) {
        uploadStatus.textContent = 'Choose a file first.';
        uploadStatus.className = 'status error';
        return;
      }
      const formData = new FormData();
      formData.append('file', file);
      uploadStatus.textContent = 'Uploading and indexing...';
      uploadStatus.className = 'status';
      try {
        const response = await fetch('/upload', { method: 'POST', body: formData });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Upload failed.');
        uploadStatus.textContent = `Uploaded ${data.filename}. Added ${data.chunks_added} chunks.`;
        askEndpoint = '/ask';
        modeStatus.textContent = 'Interactive upload mode active.';
      } catch (error) {
        uploadStatus.textContent = error.message;
        uploadStatus.className = 'status error';
      }
    });
    askForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      const question = document.getElementById('question-input').value.trim();
      if (!question) {
        answerOutput.textContent = 'Type a question first.';
        return;
      }
      answerOutput.textContent = 'Thinking...';
      matchesOutput.innerHTML = '<p class="status">Searching notes...</p>';
      try {
        const response = await fetch(askEndpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question, top_k: 4 })
        });
        const raw = await response.text();
        let data;
        try {
          data = JSON.parse(raw);
        } catch (parseError) {
          throw new Error(raw || 'The server returned an unreadable response.');
        }
        if (!response.ok) throw new Error(data.detail || 'Question failed.');
        answerOutput.textContent = data.answer;
        renderMatches(data.matches);
      } catch (error) {
        answerOutput.textContent = error.message;
        matchesOutput.innerHTML = '<p class="status">No chunks available.</p>';
      }
    });
    resetButton.addEventListener('click', async () => {
      const response = await fetch('/reset', { method: 'POST' });
      const data = await response.json();
      if (response.ok) {
        uploadStatus.textContent = 'Session cleared.';
        uploadStatus.className = 'status';
        answerOutput.textContent = 'Your answer will appear here.';
        matchesOutput.innerHTML = '<p class="status">Relevant chunks will appear here after you ask a question.</p>';
        loadMode();
      }
    });
    loadMode();
  </script>
</body>
</html>
"""

MOJIBAKE_REPLACEMENTS = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€\x9d": '"',
    "â€\"": "-",
    "â€“": "-",
    "â€”": "-",
    "â€¦": "...",
    "Â ": " ",
    "Â": "",
    "Î": "",
    "Õ": "",
}


def normalize_text(text: str) -> str:
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, good)

    # Remove most non-printable characters while preserving normal whitespace.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_answer_text(text: str) -> str:
    text = normalize_text(text)
    text = text.replace("**", "")
    text = text.replace("```", "")
    text = re.sub(r"^[ \t]*[-*][ \t]+", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_text_from_pdf(file_bytes: bytes) -> str:
    extracted_text: List[str] = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                extracted_text.append(page.extract_text() or "")
    except Exception:
        extracted_text = []
    merged = "\n".join(part.strip() for part in extracted_text if part and part.strip())
    if merged:
        return merged
    reader = PdfReader(io.BytesIO(file_bytes))
    fallback_pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(part.strip() for part in fallback_pages if part and part.strip())

def extract_text(filename: str, file_bytes: bytes) -> str:
    if filename.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_bytes)
    elif filename.lower().endswith('.txt'):
        text = file_bytes.decode('utf-8', errors='ignore')
    else:
        raise ValueError('Only PDF and TXT files are supported.')
    cleaned = "\n".join(line.strip() for line in normalize_text(text).splitlines() if line.strip())
    if not cleaned:
        raise ValueError('No readable text was extracted from the uploaded file.')
    return cleaned

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    normalized_text = " ".join(text.split())
    if len(normalized_text) <= chunk_size:
        return [normalized_text]
    chunks: List[str] = []
    start = 0
    while start < len(normalized_text):
        end = min(start + chunk_size, len(normalized_text))
        chunk = normalized_text[start:end]
        if end < len(normalized_text):
            last_space = chunk.rfind(' ')
            if last_space > int(chunk_size * 0.6):
                end = start + last_space
                chunk = normalized_text[start:end]
        chunks.append(chunk.strip())
        if end >= len(normalized_text):
            break
        start = max(end - overlap, 0)
    return [chunk for chunk in chunks if chunk]

@dataclass
class SearchResult:
    chunk: str
    source: str
    score: float

class StudyBuddyIndex:
    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.dimension = model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata: List[Dict[str, Any]] = []
    def _embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        return embeddings
    def add_document(self, source_name: str, text: str) -> Dict[str, Any]:
        chunks = chunk_text(text)
        vectors = self._embed(chunks)
        self.index.add(vectors)
        for i, chunk in enumerate(chunks):
            self.metadata.append({'source': source_name, 'chunk': chunk, 'chunk_id': i})
        return {'source': source_name, 'chunks_added': len(chunks), 'total_chunks_in_index': len(self.metadata)}
    def save(self, index_path: Path, metadata_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        metadata_path.write_text(json.dumps({'dimension': self.dimension, 'metadata': self.metadata}, indent=2), encoding='utf-8')
    def load(self, index_path: Path, metadata_path: Path) -> None:
        payload = json.loads(metadata_path.read_text(encoding='utf-8'))
        self.index = faiss.read_index(str(index_path))
        self.metadata = payload.get('metadata', [])
    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[SearchResult]:
        if not self.metadata:
            raise ValueError('No study material is available yet.')
        query_vector = self._embed([query])
        search_k = min(max(top_k * 3, top_k), len(self.metadata))
        scores, indices = self.index.search(query_vector, search_k)
        results = []
        seen_chunks = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = self.metadata[idx]
            chunk = item['chunk'].strip()
            if len(chunk) < 80:
                continue
            chunk_key = chunk[:180]
            if chunk_key in seen_chunks:
                continue
            seen_chunks.add(chunk_key)
            results.append(SearchResult(chunk=chunk, source=item['source'], score=float(score)))
            if len(results) >= top_k:
                break
        return results
    def clear(self) -> None:
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
    def stats(self) -> Dict[str, Any]:
        sources = sorted({item['source'] for item in self.metadata})
        return {'chunks': len(self.metadata), 'documents': len(sources), 'sources': sources, 'dimension': self.dimension}

class SessionStore:
    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.sessions: Dict[str, StudyBuddyIndex] = {}
    def get_or_create(self, session_id: str) -> StudyBuddyIndex:
        if session_id not in self.sessions:
            self.sessions[session_id] = StudyBuddyIndex(self.model)
        return self.sessions[session_id]
    def clear(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
    def count(self) -> int:
        return len(self.sessions)

def build_context(results: List[SearchResult]) -> str:
    sections = []
    for i, result in enumerate(results, start=1):
        sections.append(f"[Chunk {i} | Source: {result.source} | Score: {result.score:.4f}]\n{result.chunk}")
    return "\n\n".join(sections)

def answer_question(question: str, results: List[SearchResult]) -> Dict[str, Any]:
    context = build_context(results)
    if not llm_client:
        key_name = "GROQ_API_KEY" if LLM_PROVIDER == "groq" else "GEMINI_API_KEY"
        return {'answer': f'{LLM_PROVIDER.title()} is not configured yet. Retrieval worked, so use the retrieved chunks below or set {key_name} to enable answer generation.', 'context': context, 'model': None}
    system_prompt = (
        "You are an AI study buddy.\n"
        "Answer using only the provided study context.\n"
        "If the answer is not present in the notes, say so clearly.\n\n"
        "Write in clean plain text.\n"
        "Prefer short headings and bullet points.\n"
        "Do not use markdown symbols like **, *, or ```.\n"
        "Be concise and classroom-friendly."
    )
    user_prompt = (
        f"Question:\n{question}\n\nStudy Context:\n{context}"
    )
    if LLM_PROVIDER == "groq":
        response = llm_client.chat.completions.create(
            model=ANSWER_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content if response.choices else ''
        return {'answer': clean_answer_text(text or ''), 'context': context, 'model': ANSWER_MODEL_NAME}
    response = llm_client.models.generate_content(
        model=ANSWER_MODEL_NAME,
        contents=f"{system_prompt}\n\n{user_prompt}",
    )
    return {'answer': clean_answer_text(response.text or ''), 'context': context, 'model': ANSWER_MODEL_NAME}

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=10)

app = FastAPI(title=APP_TITLE, version=APP_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
session_store = SessionStore(embedding_model)
saved_index = StudyBuddyIndex(embedding_model)
if INDEX_FILE.exists() and METADATA_FILE.exists():
    saved_index.load(INDEX_FILE, METADATA_FILE)

def ensure_session(session_id: str | None) -> str:
    return session_id or str(uuid.uuid4())

@app.get('/', response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(HTML_PAGE)

@app.get('/health')
def health() -> Dict[str, Any]:
    return {
        'status': 'ok',
        'embedding_model': EMBEDDING_MODEL_NAME,
        'answer_generation_enabled': bool(llm_client),
        'llm_provider': LLM_PROVIDER,
        'answer_model': ANSWER_MODEL_NAME,
        'saved_index_loaded': bool(saved_index.metadata),
        'saved_index_stats': saved_index.stats(),
        'active_sessions': session_store.count(),
    }

@app.get('/knowledge-base')
def knowledge_base_status() -> Dict[str, Any]:
    return {'loaded': bool(saved_index.metadata), 'index_stats': saved_index.stats()}

@app.post('/upload')
async def upload_document(response: Response, file: UploadFile = File(...), study_buddy_session: str | None = Cookie(default=None)) -> Dict[str, Any]:
    session_id = ensure_session(study_buddy_session)
    response.set_cookie(key='study_buddy_session', value=session_id, httponly=True, samesite='lax')
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail=f'File too large. Maximum supported size is {MAX_FILE_SIZE_MB} MB.')
    text = extract_text(file.filename, file_bytes)
    index = session_store.get_or_create(session_id)
    result = index.add_document(file.filename, text)
    return {'message': 'Document uploaded and indexed successfully.', 'filename': file.filename, 'characters_extracted': len(text), 'index_stats': index.stats(), **result}

@app.post('/ask')
def ask_question(payload: AskRequest, response: Response, study_buddy_session: str | None = Cookie(default=None)) -> Dict[str, Any]:
    session_id = ensure_session(study_buddy_session)
    response.set_cookie(key='study_buddy_session', value=session_id, httponly=True, samesite='lax')
    try:
        index = session_store.get_or_create(session_id)
        results = index.search(payload.question.strip(), top_k=payload.top_k)
        answer_payload = answer_question(payload.question.strip(), results)
        return {
            'question': payload.question.strip(),
            'answer': answer_payload['answer'],
            'llm_model': answer_payload['model'],
            'matches': [
                {'source': item.source, 'score': round(item.score, 4), 'chunk_preview': textwrap.shorten(item.chunk, width=260, placeholder=' ...')}
                for item in results
            ],
            'context_used': answer_payload['context'],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Backend error: {exc}') from exc

@app.post('/ask-saved')
def ask_saved(payload: AskRequest) -> Dict[str, Any]:
    try:
        results = saved_index.search(payload.question.strip(), top_k=payload.top_k)
        answer_payload = answer_question(payload.question.strip(), results)
        return {
            'question': payload.question.strip(),
            'answer': answer_payload['answer'],
            'llm_model': answer_payload['model'],
            'matches': [
                {'source': item.source, 'score': round(item.score, 4), 'chunk_preview': textwrap.shorten(item.chunk, width=260, placeholder=' ...')}
                for item in results
            ],
            'context_used': answer_payload['context'],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Backend error: {exc}') from exc

@app.post('/reset')
def reset_session(response: Response, study_buddy_session: str | None = Cookie(default=None)) -> Dict[str, Any]:
    session_id = ensure_session(study_buddy_session)
    session_store.clear(session_id)
    response.set_cookie(key='study_buddy_session', value=session_id, httponly=True, samesite='lax')
    return {'message': 'Session cleared.'}
