import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# 🔒 LOCAL MODEL PATH (offline)
MODEL_PATH = "/home/ubuntu/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a"

SYSTEM_PROMPT = (
    "You are a helpful AI assistant specialized in machine learning, "
    "deep learning, and reasoning tasks. Answer clearly and correctly."
)

# Load tokenizer & model (offline)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True
)

app = FastAPI(title="Local Mistral LLM API")

class Request(BaseModel):
    prompt: str
    max_tokens: int = 300
    temperature: float = 0.7

@app.post("/generate")
def generate(req: Request):
    # Build final prompt
    full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {req.prompt}\nAssistant:"

    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt echo
    response = decoded.split("Assistant:")[-1].strip()

    return {"response": response}
