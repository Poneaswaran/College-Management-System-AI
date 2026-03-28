import requests
import json
import re

OLLAMA_URL = "http://172.27.112.1:11434/api/generate"

def clean_response(text):
    # remove markdown ```json ```
    text = re.sub(r"```json|```", "", text).strip()
    return text

def generate_answer(query, context):
    prompt = f"""
You are a construction expert AI.

STRICT RULES:
- Output ONLY valid JSON
- No markdown
- No explanation outside JSON
- DO NOT wrap output in ```json or markdown.
Write clear, grammatically correct English.
Avoid typos or broken words.
Context:
{context}

User:
{query}

Format:
{{
  "answer": "",
  "materials": [],
  "risks": [],
  "next_question": ""
}}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "phi3",
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 300   # prevents cut-off
            }
        }
    )

    raw = response.json()["response"]
    cleaned = clean_response(raw)

    try:
        return json.loads(cleaned)
    except:
        return {
            "error": "Invalid JSON",
            "raw": cleaned
        }