
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Chunk(BaseModel):
    chunk_id: str
    text: str

class RequestBody(BaseModel):
    question: str
    chunks: List[Chunk]

def tokenize(text: str):
    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))

def split_sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def score(question_tokens, sentence):
    st = tokenize(sentence)
    if not st:
        return 0.0
    overlap = len(question_tokens & st)
    if overlap == 0:
        return 0.0
    return overlap / max(len(question_tokens), 1)

@app.get("/")
def health():
    return {"status":"ok"}

def process(req: RequestBody):
    if req is None or not req.question.strip():
        return {
            "answerable": False,
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.2
        }

    q_tokens = tokenize(req.question)
    best_sentence = None
    best_chunk = None
    best_score = 0.0

    for chunk in req.chunks:
        for sent in split_sentences(chunk.text):
            s = score(q_tokens, sent)
            if s > best_score:
                best_score = s
                best_sentence = sent
                best_chunk = chunk

    if best_chunk is None or best_score < 0.25:
        return {
            "answerable": False,
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.2
        }

    return {
        "answerable": True,
        "answer": best_sentence,
        "citations": [best_chunk.chunk_id],
        "confidence": round(min(0.35 + best_score, 0.95),2)
    }

@app.post("/")
def root(req: RequestBody):
    return process(req)

@app.post("/grounded-answer")
def grounded(req: RequestBody):
    return process(req)

@app.post("/answer")
def answer(req: RequestBody):
    return process(req)

@app.post("/api")
def api(req: RequestBody):
    return process(req)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
