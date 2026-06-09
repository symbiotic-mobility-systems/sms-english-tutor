from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import anthropic
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

@app.get("/")
def root():
    return {"status": "SMS English Tutor API is running"}

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    system = body.get("system", "You are a helpful assistant.")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=system,
        messages=messages
    )

    return {"content": response.content[0].text}
