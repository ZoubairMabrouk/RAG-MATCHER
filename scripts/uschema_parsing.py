#!/usr/bin/env python3
"""
Dynamic RAG Virtual Rename runner

What it does
------------
1) Loads U-Schema (JSON) from --uschema-file (or stdin).
2) Introspects the current relational schema dynamically from your DI container
   using the provided --db-url (or $DATABASE_URL) and --dialect.
3) Builds a semantic KB from the current schema and indexes it in a FAISS store.
4) Runs RAG-based matching:
   - entity -> existing table (virtual rename)
   - attribute -> existing column (virtual rename)
5) Computes a migration plan with DiffEngine (using the matcher) and prints SQL
   statements (no physical RENAME operations).
6) Emits a JSON mapping report to stdout (optional file via --out).

Usage
-----
python scripts/run_rag_virtual_rename.py \
  --uschema-file path/to/uschema.json \
  --db-url postgresql://user:pass@localhost:5432/db \
  --dialect postgresql \
  --index-type auto \
  --table-threshold 0.35 \
  --column-threshold 0.35 \
  --top-k 5 \
  --out artifacts/rag_mapping.json

Notes
-----
- For tiny schemas (few docs), start with low thresholds (0.25–0.40).
- The script **never** generates physical RENAME statements; it relies on
  virtual mapping when computing the plan.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

# project imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.infrastructure.di_container import DIContainer
from src.domain.entities.schema import (
    USchema, USchemaEntity, USchemaAttribute, DataType,
    SchemaMetadata
)
from src.domain.entities.rules import NamingConvention
from src.domain.entities.evolution import ChangeType
from src.domain.services.diff_engine import DiffEngine
from src.domain.services.migration_builder import MigrationBuilder

from src.infrastructure.rag.embedding_service import EmbeddingService, LocalEmbeddingProvider
from src.infrastructure.rag.vector_store import RAGVectorStore
from src.infrastructure.rag.rag_schema_matcher import RAGSchemaMatcher
from src.infrastructure.llm.llm_client import OpenAILLMClient, LLMClient, AnthropicLLMClient,BaseLLMClient


# -------------------- logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("run_rag_virtual_rename")


# -------------------- helpers --------------------
_STR2DT = {
    "string": DataType.STRING,
    "integer": DataType.INTEGER,
    "int": DataType.INTEGER,
    "decimal": DataType.DECIMAL,
    "float": DataType.DECIMAL,
    "boolean": DataType.BOOLEAN,
    "bool": DataType.BOOLEAN,
    "timestamp": DataType.TIMESTAMP,
    "datetime": DataType.TIMESTAMP,
    "date": DataType.DATE,
    "json": DataType.JSON,
    "uuid": DataType.UUID,
}

def _to_datatype(val: str) -> DataType:
    if isinstance(val, DataType):
        return val
    key = str(val).strip().lower()
    return _STR2DT.get(key, DataType.STRING)

def load_uschema(uschema_json: Dict[str, Any]) -> USchema:
    entities = []

    for e in uschema_json.get("uSchemaModel", {}).get("entities", []):
        et = e.get("EntityType", {})
        name = et.get("name", "").lower()  # normalize
        attributes = []

        for variation in et.get("variations", []):
            sv = variation.get("StructuralVariation", {})

            for prop_block in sv.get("properties", []):
                attr_list = prop_block.get("Attribute") or []
                if isinstance(attr_list, dict):
                    attr_list = [attr_list]
                for a in attr_list:
                    aname = a.get("name", "").lower()
                    atype = _to_datatype(a.get("type", "string"))
                    iskey = a.get("iskey", False)
                    if aname:
                        attributes.append(USchemaAttribute(
                            name=aname,
                            data_type=atype,
                            is_key=iskey
                        ))

            for agg_block in sv.get("aggregates", []):
                agg = agg_block.get("Aggregation")
                if agg:
                    attributes.append(USchemaAttribute(
                        name=f"{agg.get('name','').lower()}->{agg.get('target','').lower()}",
                        data_type=DataType.JSON,
                        is_key=False
                    ))

            for ref_block in sv.get("references", []):
                ref = ref_block.get("Reference")
                if ref:
                    attributes.append(USchemaAttribute(
                        name=f"{ref.get('name','').lower()}->{ref.get('target','').lower()}",
                        data_type=DataType.JSON,
                        is_key=False
                    ))

        entities.append(USchemaEntity(name=name, attributes=attributes))

    return USchema(entities=entities)

# -----------------------------

with open("./data/gpt/source_uschema.json", "r", encoding="utf-8") as f:
    uschema_json = json.load(f)
base_model = load_uschema(uschema_json)

# affichage du model de base
print("Base U-Schema Model:")
print(base_model)
            

