import torch
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# 🔒 LOCAL MODEL PATH (offline)
MODEL_PATH = "/home/ubuntu/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a"

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Answer clearly, concisely, and based only on the given context."
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True
)

model.eval()  # 🔥 IMPORTANT

# Optional compile (PyTorch 2.x)
if torch.__version__.startswith("2"):
    model = torch.compile(model)

app = FastAPI(title="Optimized Local Mistral LLM API")

# 🔥 Serialize requests (VERY IMPORTANT)
semaphore = asyncio.Semaphore(1)

class Request(BaseModel):
    prompt: str
    max_tokens: int = 300

@app.post("/generate")
async def generate(req: Request):
    async with semaphore:
        # Hard limits (speed protection)
        max_new_tokens = min(req.max_tokens, 400)

        full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {req.prompt}\nAssistant:"

        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048  # 🔥 CRITICAL
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # 🔥 deterministic & faster
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True            # 🔥 speed-up
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = decoded.split("Assistant:")[-1].strip()

        return {"response": response}