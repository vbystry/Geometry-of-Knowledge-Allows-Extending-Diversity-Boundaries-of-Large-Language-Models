# server.py
import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1a) Read your HF token
HF_TOKEN = os.getenv("HUGGINGFACE_API_key")

# 1b) Initialize FastAPI
app = FastAPI(title="Mistral-7B Inference Server")

# 1c) Load model & tokenizer exactly once at startup
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    use_auth_token=HF_TOKEN
)
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,                # assuming single GPU
    trust_remote_code=True   # if needed
)

# 1d) Define request/response schema
class GenRequest(BaseModel):
    prompt: str
    max_length: int = 512
    temperature: float = 0.7

class GenResponse(BaseModel):
    generated_text: str

# 1e) Expose a single /generate endpoint
@app.post("/generate", response_model=GenResponse)
def generate(req: GenRequest):
    out = generator(
        req.prompt,
        max_length=req.max_length,
        do_sample=True,
        temperature=req.temperature
    )
    return GenResponse(generated_text=out[0]["generated_text"])
