# import json
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# # Load a local open-source model, e.g., GPT4All or a small LLaMA variant
# # Example: Nous-Hermes-13b (can be swapped for a smaller model)
# model_name = "tiiuae/falcon-7b-instruct"  # choose a model available locally

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")  # GPU if available

# # Create a text-generation pipeline
# generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)

# # Load existing knowledge base
# kb_file = "./data/rag/knowledge_base.jsonl"
# with open(kb_file, "r", encoding="utf-8") as f:
#     kb_docs = [json.loads(line) for line in f]

# # Function to enrich one KB entry
# def enrich_kb_entry(entry):
#     # Generate prompt with context
#     prompt = f"""
#     You are a database expert. Enrich the following column/table metadata with:
#     - Synonyms
#     - Abbreviations
#     - Extended description
#     - Example values
#     Only return JSON with keys: synonyms, abbreviations, extended_description, examples.
    
#     Input: {entry['content']}
#     """
#     result = generator(prompt, max_length=256, do_sample=True, temperature=0.7)
#     try:
#         enrichment_text = result[0]["generated_text"]
#         # Extract JSON from text (simple heuristic)
#         start = enrichment_text.find("{")
#         end = enrichment_text.rfind("}") + 1
#         enrichment_json = json.loads(enrichment_text[start:end])
#         return enrichment_json
#     except Exception as e:
#         print(f"Error enriching {entry['id']}: {e}")
#         return {}

# # Loop over KB and enrich
# for entry in kb_docs:
#     enrichment = enrich_kb_entry(entry)
#     if enrichment:
#         entry.setdefault("metadata", {}).update(enrichment)

# # Save enriched KB
# with open("./data/rag/knowledge_base_enriched_2.jsonl", "w", encoding="utf-8") as f:
#     for entry in kb_docs:
#         f.write(json.dumps(entry) + "\n")

# print("Knowledge base enrichment done!")

import os
import json
from groq import Groq

# Initialize the Groq client
client = Groq(api_key="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")

# Load your knowledge base
kb_file = "./data/rag/knowledge_base_enriched.jsonl"
with open(kb_file, "r", encoding="utf-8") as f:
    kb_docs = [json.loads(line) for line in f]

# Define the model
MODEL_NAME = "llama-3.1-8b-instant"  # small, very fast, free-tier friendly

# Function to enrich a single entry
def enrich_kb_entry(entry):
    prompt = f"""
    You are a data model enrichment assistant. 
    Given this database column or table description:
    "{entry['content']}"
    
    Enrich it by providing:
    - synonyms
    - abbreviations
    - an extended_description
    - example_values
    
    Respond ONLY in JSON with this exact structure:
    {{
        "synonyms": [...],
        "abbreviations": [...],
        "extended_description": "...",
        "examples": [...]
    }}
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = response.choices[0].message["content"]
        data = json.loads(content)
        return data
    except Exception as e:
        print(f"Error enriching {entry.get('id', 'unknown')}: {e}")
        return {}

# Enrich all entries
for entry in kb_docs:
    enrichment = enrich_kb_entry(entry)
    if enrichment:
        entry.setdefault("metadata", {}).update(enrichment)

# Save the enriched version
output_file = "./data/rag/knowledge_base_enriched_groq.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for entry in kb_docs:
        f.write(json.dumps(entry) + "\n")

print("✅ Knowledge base enrichment done using Groq API!")
