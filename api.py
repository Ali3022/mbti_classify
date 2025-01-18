from fastapi import FastAPI
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

app = FastAPI()

# Add this test endpoint to ensure the server is running
@app.get("/")
def read_root():
    return {"message": "Server is running"}

# Load the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("./mbti_model")
model = RobertaForSequenceClassification.from_pretrained("./mbti_model")
model.eval()

# Prediction endpoint
@app.post("/predict/")
def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return {"predicted_personality": prediction}
