"""Live DeepSRGM inference API using the project's saved best_model.pt checkpoint.

Run with: uvicorn ragasense.live_inference:app --host 127.0.0.1 --port 8001
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "ragasense_deepsrgm_results" / "training" / "best_model.pt"


class DeepSRGM(torch.nn.Module):
    """Checkpoint-compatible LSTM + temporal-attention DeepSRGM classifier."""
    def __init__(self, *, vocab_size, embedding_size, hidden_size, num_layers,
                 num_classes, drop_prob, **_):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.rnn = torch.nn.LSTM(embedding_size, hidden_size, num_layers,
                                 batch_first=True)
        self.w_omega = torch.nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.u_omega = torch.nn.Parameter(torch.empty(hidden_size, 1))
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = torch.nn.Linear(hidden_size // 2, num_classes)
        self.dropout = torch.nn.Dropout(drop_prob)

    def forward(self, tokens):
        outputs, _ = self.rnn(self.embedding(tokens))
        attention = torch.softmax(torch.tanh(outputs @ self.w_omega) @ self.u_omega, dim=1)
        context = torch.sum(outputs * attention, dim=1)
        return self.fc2(self.dropout(torch.relu(self.fc1(context))))


class LiveRequest(BaseModel):
    pitch_sequence: list[int] = Field(min_length=5000, max_length=5000)
    tonic: float = Field(gt=0)
    sample_rate: float = Field(gt=0)
    tradition: Literal["all", "Carnatic", "Hindustani"] = "all"


def load_model():
    if not CHECKPOINT.exists():
        raise RuntimeError(f"Checkpoint not found: {CHECKPOINT}")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = DeepSRGM(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint["class_mapping"], checkpoint["model_config"]


MODEL, CLASSES, CONFIG = load_model()
app = FastAPI(title="RagaSense live inference", version="1.0")
app.add_middleware(
    CORSMiddleware,
    # Development dashboard may use any free localhost port.
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):[0-9]+",
    allow_methods=["POST", "GET"],
    allow_headers=["content-type"],
)


@app.get("/api/health")
def health():
    return {"status": "ready", "sequence_length": CONFIG["input_length"], "checkpoint": CHECKPOINT.name}


@app.post("/api/analyze/live")
def analyze_live(payload: LiveRequest):
    if any(token < 0 or token >= CONFIG["vocab_size"] for token in payload.pitch_sequence):
        raise HTTPException(422, "pitch_sequence contains an out-of-vocabulary token")
    with torch.inference_mode():
        logits = MODEL(torch.tensor([payload.pitch_sequence], dtype=torch.long))[0]
        allowed = range(20) if payload.tradition == "all" else (
            range(0, 10) if payload.tradition == "Carnatic" else range(10, 20)
        )
        mask = torch.full_like(logits, float("-inf"))
        mask[list(allowed)] = logits[list(allowed)]
        probabilities = torch.softmax(mask, dim=0)
        values, indices = torch.topk(probabilities, k=5)
    top = [{"raga": CLASSES[int(index)], "probability": float(value)}
           for value, index in zip(values, indices)]
    prediction_tradition = "Carnatic" if int(indices[0]) < 10 else "Hindustani"
    return {"raga": top[0]["raga"], "tradition": prediction_tradition,
            "confidence": top[0]["probability"], "top_predictions": top}
