import os
import json
import base64
import requests
import re
from datetime import datetime

def analyze_material(image_path, api_key):
    """
    Sends the crack snapshot to DeepSeek's vision model to extract material and repair info.
    Expects a strict JSON response.
    """
    if not api_key:
        return {"error": "No API key provided.", "material_label": "unparsed"}
        
    if not os.path.exists(image_path):
        return {"error": "Image file not found.", "material_label": "unparsed"}
        
    with open(image_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")
        
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = """
You are an expert civil engineer and materials scientist analyzing a photo of a surface crack.
Analyze the provided image crop and return ONLY valid JSON with this exact shape (no markdown fences, no preamble, no other text):
{
  "material": "<concrete | plaster | brick | wood | tile | metal | stone | unknown>",
  "confidence": <0.0-1.0 float>,
  "surface_condition": "<short phrase, e.g. 'painted, moderately weathered'>",
  "repair_materials": ["<item>", "..."],
  "repair_tools": ["<item>", "..."],
  "notes": "<one short sentence>"
}
"""

    payload = {
        "model": "deepseek-flash",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt.strip()},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}
                    }
                ]
            }
        ]
    }
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data['choices'][0]['message']['content'].strip()
        
        # Defensive parse: strip markdown fences
        clean_text = raw_text
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()
        
        try:
            parsed = json.loads(clean_text)
            return {
                "material_label": parsed.get('material', 'unknown'),
                "material_confidence": float(parsed.get('confidence', 0.0)),
                "repair_notes": parsed.get('notes', ''),
                "repair_materials": parsed.get('repair_materials', []),
                "repair_tools": parsed.get('repair_tools', []),
                "surface_condition": parsed.get('surface_condition', ''),
                "raw_model_response": raw_text,
                "analyzed_at": datetime.now().isoformat(),
                "model_used": "deepseek-flash"
            }
        except json.JSONDecodeError:
            return {
                "material_label": "unparsed",
                "material_confidence": 0.0,
                "repair_notes": "Failed to parse JSON",
                "raw_model_response": raw_text,
                "analyzed_at": datetime.now().isoformat(),
                "model_used": "deepseek-flash",
                "error": "Malformed JSON from API"
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "error": f"API Request failed: {str(e)}",
            "material_label": "unparsed"
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "material_label": "unparsed"
        }
