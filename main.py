from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import requests

app = FastAPI()

@app.get("/")
def home():
    return {"status": "working"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allows frontend (HTML/React/Vercel)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")

OPENROUTER_API_KEY = "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"

def get_llm_advice(risk, data):

    prompt = f"""
Patient data:
Age: {data['age']}
BMI: {data['bmi']}
Blood Pressure: {data['bp']}
Exercise: {data['exercise']}
Family History: {data['family']}

Risk score: {risk}%

Give:
1. Explanation
2. Preventive steps
"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer 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"
        },
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}]
        }
    )

    return response.json()["choices"][0]["message"]["content"]

@app.post("/predict")
def predict(data: dict):

    features = np.array([[
        data["age"],
        data["bmi"],
        data["bp"],
        data["exercise"],
        data["family"]
    ]])

    prob = model.predict_proba(features)[0][1]
    risk = round(prob * 100, 2)

    advice = get_llm_advice(risk, data)

    return {
        "risk": risk,
        "category": (
            "High" if risk > 70 else
            "Medium" if risk > 40 else
            "Low"
        ),
        "advice": advice
    }

def get_llm_advice(risk, data):

    prompt = f"""
You are a medical education assistant.

Patient data:
Age: {data['age']}
BMI: {data['bmi']}
Blood Pressure: {data['bp']}
Exercise: {data['exercise']}
Family History: {data['family']}

Risk score: {risk}%

Give:
1. Simple explanation of risk
2. Top 5 lifestyle improvements
3. Preventive screening advice

Keep it simple and non-technical.
"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer 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"
        },
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )

    return response.json()["choices"][0]["message"]["content"]