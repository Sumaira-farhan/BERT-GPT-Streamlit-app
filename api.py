from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Load Models

sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

# Request Schemas

class TextRequest(BaseModel):
    text: str

class PromptRequest(BaseModel):
    prompt: str

# Home Route

@app.get("/")
def home():
    return {"message": "AI API Running"}

# Sentiment Route

@app.post("/predict")
def predict(data: TextRequest):

    result = sentiment_model(data.text)[0]

    return {
        "label": result["label"],
        "score": round(result["score"], 2)
    }

# Text Generation Route

@app.post("/generate")
def generate(data: PromptRequest):

    output = generator(
        data.prompt,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    return {
        "generated_text": output[0]["generated_text"]
    }
