from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
import os
import math

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class RequestBody(BaseModel):
    query_id: Optional[str] = None
    query: str
    candidates: List[str]

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/")
def rank_candidates(req: RequestBody):

    texts = [req.query] + req.candidates

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    embeddings = [item.embedding for item in response.data]

    query_embedding = embeddings[0]
    candidate_embeddings = embeddings[1:]

    similarities = []

    for index, embedding in enumerate(candidate_embeddings):
        score = cosine_similarity(query_embedding, embedding)
        similarities.append((index, score))

    similarities.sort(key=lambda x: x[1], reverse=True)

    ranking = [idx for idx, _ in similarities[:3]]

    return {"ranking": ranking}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
