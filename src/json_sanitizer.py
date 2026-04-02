import re
import json

def safe_json_array(text: str):
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        raise ValueError("No JSON array found in LLM output.")
    return json.loads(match.group(0))
