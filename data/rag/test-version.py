import json, time
from tqdm import tqdm
from openai import OpenAI

# Use Ollama as local model backend (through OpenAI-compatible API)
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

KB_FILE = "./data/rag/knowledge_base.jsonl"
ENRICHED_FILE = "./data/rag/matching_result.jsonl" 
MODEL = "phi3:mini"  # try "phi3" or "llama3" if you want even faster responses

# Load KB safely (or start empty)
try:
    with open(KB_FILE, "r", encoding="utf-8") as f:
        kb_docs = [json.loads(line) for line in f]
except FileNotFoundError:
    print("⚠️ knowledge_base.jsonl not found, starting with empty KB.")
    kb_docs = []

def enrich_text_with_model(text: str, model: str = MODEL) -> str:
    """Enrich a text description using a local model."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in schema evolution and the matching with semantic details and synonyms."},
                {"role": "user", "content": f"Retrieve the corresponding matching for this element: {text} with the database mimic iii schema with score of the confidence for that matching."}
            ],
            temperature=0.2,
            max_tokens=200
        )
        # ✅ FIXED: use .choices[0].message.content instead of indexing incorrectly
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Error enriching text: {e}")
        return text

with open(ENRICHED_FILE, "w", encoding="utf-8") as out:
    documents = ["patient id", "first name", "last name", "date of birth", "gender", "address", "phone number", "email", "emergency contact", "allergies", "current medications", "past medical history", "family medical history", "immunizations", "vital signs", "lab results", "imaging results", "diagnoses", "treatment plans", "progress notes", "discharge summaries"]
    for doc in tqdm(documents, desc="element not matched yet : wait for matching"):
        enriched_doc = {}
        content = doc
        enriched_doc["element not matched yet : wait for matching"] = enrich_text_with_model(content)
        out.write(json.dumps(enriched_doc) + "\n")
        time.sleep(0.05)  # small delay for stability

print(f"✅ Matching done! File saved as: {ENRICHED_FILE}")
