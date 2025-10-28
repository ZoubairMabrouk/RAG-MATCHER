import json
import openai
from pathlib import Path
from time import sleep

# -----------------------------
# CONFIGURATION
# -----------------------------
openai.api_key = "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"

INPUT_KB_FILE = "./data/rag/knowledge_base.jsonl"        # Your original KB
OUTPUT_KB_FILE = "./data/rag/knowledge_base_enriched_2.jsonl"  # Output enriched KB
MODEL = "gpt-3.5-turbo"                                 # Use GPT-4 for better contextual enrichment
SLEEP_BETWEEN_CALLS = 1                          # Avoid rate limits

# -----------------------------
# GPT Prompt Template
# -----------------------------
def generate_prompt(kb_entry):
    table = kb_entry.get("table", "")
    column = kb_entry.get("column", "")
    description = kb_entry.get("metadata", {}).get("description", "")
    
    prompt = f"""
You are a database expert. Enrich the following database column description for semantic search:

Table: {table}
Column: {column}
Current description: "{description}"

Please provide:
1. A detailed human-readable content description.
2. Synonyms or abbreviations for this column.
3. Example values (up to 5 realistic examples).
4. Notes about relationships with other tables if applicable.

Return output in JSON format with keys:
- content
- metadata: synonyms (list), examples (list), description (string)
"""
    return prompt

# -----------------------------
# GPT-4 Enrichment Function
# -----------------------------
def enrich_entry(kb_entry):
    prompt = generate_prompt(kb_entry)
    
    try:
        response = openai.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        enriched_text = response.choices[0].message.content.strip()
        enriched_json = json.loads(enriched_text)
        # Merge enriched fields into original KB entry
        kb_entry["content"] = enriched_json.get("content", kb_entry.get("content"))
        kb_entry["metadata"].update(enriched_json.get("metadata", {}))
        return kb_entry
    except Exception as e:
        print(f"Error enriching {kb_entry.get('id')}: {e}")
        return kb_entry

# -----------------------------
# Main Loop
# -----------------------------
def enrich_kb(input_file, output_file):
    enriched_entries = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            kb_entry = json.loads(line)
            enriched_entry = enrich_entry(kb_entry)
            enriched_entries.append(enriched_entry)
            sleep(SLEEP_BETWEEN_CALLS)  # avoid hitting API rate limits

    # Save enriched KB
    with open(output_file, "w", encoding="utf-8") as f_out:
        for entry in enriched_entries:
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Enriched KB saved to: {output_file}")

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    enrich_kb(INPUT_KB_FILE, OUTPUT_KB_FILE)
