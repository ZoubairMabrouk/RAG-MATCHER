import json
import requests

from domain.entities.schema import SchemaMetadata
from infrastructure.di_container import DIContainer

# Ollama server endpoint
OLLAMA_URL = "http://rag_ollama:11434/completions"
db_url = "postgresql://test:test@rag_db:5432/test"


# Load U-Schema and relational schema
with open("uschema_testdata.json") as f:
    u_schema = json.load(f)

with open("mimic_dictionnary.json") as f:
    catalog = json.load(f)

container = DIContainer()
container.configure(db_url, "postgresql")
inspector = container.get_inspector()
current_schema: SchemaMetadata = inspector.introspect_schema()
# Construct the prompt for semantic matching with confidence scoring
prompt = f"""
You are a schema matching assistant.

Input:
1. Unified Schema (U-Schema):
{json.dumps(u_schema, indent=2)}

2. Relational Schema:
{current_schema}

3. Additional Catalog Information:
{json.dumps(catalog, indent=2)}

Task:
- For each entity and attribute in the U-Schema, find the best matching attribute in the relational schema based the additional catalog information.
- Provide a semantic confidence score (0 to 1) for each match.
- Output JSON in the following format:
[
  {{
    "uschema_entity": "EntityName",
    "uschema_attribute": "AttributeName",
    "relational_table": "TableName",
    "relational_column": "ColumnName",
    "confidence": 0.85
  }}
]
"""

# Payload for Ollama
payload = {
    "model": "llama3.1",  # replace with your deployed model
    "prompt": prompt,
    "temperature": 0
}

# Send request to Ollama server


from src.infrastructure.llm.llm_client import OpenAI,BaseLLMClient, LLMClient
llm_client = LLMClient(base_url=OLLAMA_URL, model="llama3.1")


response = llm_client
result = response.json()

# The model's output
llm_output = result.get("completion", "")
try:
    matches = json.loads(llm_output)
except json.JSONDecodeError:
    matches = [{"error": "Failed to parse LLM output", "raw_output": llm_output}]

# Print matches with confidence
print(json.dumps(matches, indent=2))
