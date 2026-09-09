# """
# Exemple d'utilisation
# ----------------------
# python select_best_embedding_and_k.py \
#     --uschema uschema.json \
#     --target-schema mimic_schema.json \
#     --labels omop_mimic_data.xlsx \
#     --models sentence-transformers/LaBSE sentence-transformers/all-MiniLM-L6-v2 \
#              BAAI/bge-small-en-v1.5 intfloat/e5-small-v2 dmis-lab/biobert-base-cased-v1.2 \
#     --k-list 1 3 5 10 15 20 30 \
#     --out results.json

# Si vous n'avez ni uschema.json ni mimic_schema.json sous la main, lancez le
# script uniquement avec --labels : les deux schemas seront reconstruits
# automatiquement a partir du fichier etiquete (utile pour un premier test
# rapide de la methodologie).
# """

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


DEFAULT_MODELS = [
    "sentence-transformers/LaBSE",          
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-small-v2",
    "dmis-lab/biobert-base-cased-v1.2",
]

DEFAULT_K_LIST = [1, 3, 5, 10, 15, 20, 30]

RECALL_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9] 



@dataclass
class SchemaItem:
    """Un element indexable : un attribut du U-Schema, ou une colonne cible."""
    item_id: str          # ex: "person-person_id"  ou  "admissions-hadm_id"
    parent: str            # nom de l'entite / table
    name: str              # nom de l'attribut / colonne
    text: str              # texte concatene utilise pour l'embedding


def _clean(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\xa0", " ")).strip()


def load_uschema(path: Optional[str], labels_path: Optional[str]) -> List[SchemaItem]:
    """Charge le U-Schema (source, ex: OMOP) depuis un fichier JSON dedie,
    ou le reconstruit depuis le fichier xlsx etiquete si absent."""
    if path and Path(path).exists():
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        items = []
        for entity in raw:
            ent_name = entity["entity"]
            ent_desc = _clean(entity.get("description"))
            for attr in entity["attributes"]:
                name = attr["name"]
                desc = _clean(attr.get("description"))
                text = " ".join(filter(None, [name.replace("_", " "), ent_desc, desc]))
                items.append(SchemaItem(f"{ent_name}-{name}", ent_name, name, text))
        return items

    if not labels_path:
        raise ValueError("Ni --uschema ni --labels fournis : impossible de "
                          "construire le U-Schema.")
    return _uschema_from_labels(labels_path)


def load_target_schema(path: Optional[str], labels_path: Optional[str]) -> List[SchemaItem]:
    """Charge le schema cible (ex: mimic_schema.json) depuis un fichier JSON
    dedie, ou le reconstruit depuis le fichier xlsx etiquete si absent."""
    if path and Path(path).exists():
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        items = []
        for table in raw:
            tbl_name = table["table"]
            tbl_desc = _clean(table.get("description"))
            for col in table["columns"]:
                name = col["name"]
                desc = _clean(col.get("description"))
                text = " ".join(filter(None, [name.replace("_", " "), tbl_desc, desc]))
                items.append(SchemaItem(f"{tbl_name}-{name}", tbl_name, name, text))
        return items

    if not labels_path:
        raise ValueError("Ni --target-schema ni --labels fournis : impossible "
                          "de construire le schema cible.")
    return _target_schema_from_labels(labels_path)


def _iter_label_rows(labels_path: str):
    import openpyxl
    wb = openpyxl.load_workbook(labels_path, data_only=True)
    ws = wb.active
    for row in ws.iter_rows(min_row=2, values_only=True):
        omop, table_col, des1, des2, label, d1, d2, d3, d4 = row
        yield omop, table_col, label, d1, d2, d3, d4


def _uschema_from_labels(labels_path: str) -> List[SchemaItem]:
    seen: Dict[str, SchemaItem] = {}
    for omop, _table_col, _label, d1, d2, _d3, _d4 in _iter_label_rows(labels_path):
        if omop in seen:
            continue
        entity, attr = omop.split("-", 1)
        text = " ".join(filter(None, [attr.replace("_", " "), _clean(d1), _clean(d2)]))
        seen[omop] = SchemaItem(omop, entity, attr, text)
    return list(seen.values())


def _target_schema_from_labels(labels_path: str) -> List[SchemaItem]:
    seen: Dict[str, SchemaItem] = {}
    for _omop, table_col, _label, _d1, _d2, d3, d4 in _iter_label_rows(labels_path):
        if table_col in seen:
            continue
        table, col = table_col.split("-", 1)
        text = " ".join(filter(None, [col.replace("_", " "), _clean(d3), _clean(d4)]))
        seen[table_col] = SchemaItem(table_col, table, col, text)
    return list(seen.values())


def load_gold_labels(labels_path: str) -> Dict[str, set]:
    """qid (U-Schema item_id) -> set(cid) des vraies correspondances (label=1)."""
    gold: Dict[str, set] = defaultdict(set)
    for omop, table_col, label, *_ in _iter_label_rows(labels_path):
        if label == 1:
            gold[omop].add(table_col)
    return dict(gold)



class Embedder:
    """Enveloppe un modele d'embedding. .encode(list[str]) -> np.ndarray (n, d)."""

    def __init__(self, model_name: str, offline_fallback: bool = False):
        self.model_name = model_name
        self._st_model = None
        self._fallback_vectorizer = None
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(model_name)
        except Exception as e:  
            if not offline_fallback:
                raise RuntimeError(
                    f"Impossible de charger le modele '{model_name}' via "
                    f"sentence-transformers ({e}). Relancez avec "
                    f"--offline-fallback pour utiliser un embedding local de "
                    f"secours (TF-IDF caracteres), ou verifiez votre acces "
                    f"reseau / installez sentence-transformers."
                ) from e
            print(f"[WARN] '{model_name}' indisponible ({e}). "
                  f"Fallback local (TF-IDF caracteres) active pour ce modele.",
                  file=sys.stderr)

    def fit_fallback(self, all_texts: List[str]):
        """A appeler uniquement en mode fallback : fit du TF-IDF sur tout le
        corpus (U-Schema + schema cible) avant d'encoder quoi que ce soit."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._fallback_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self._fallback_vectorizer.fit(all_texts)

    @property
    def is_fallback(self) -> bool:
        return self._st_model is None

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._st_model is not None:
            emb = self._st_model.encode(
                texts, batch_size=64, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
            )
            return emb.astype(np.float32)
        if self._fallback_vectorizer is None:
            raise RuntimeError("fit_fallback() doit etre appele avant encode() "
                                "en mode fallback.")
        X = self._fallback_vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms



class SchemaIndex:
    """Index vectoriel simple sur les embeddings du schema cible."""

    def __init__(self, embeddings: np.ndarray, item_ids: List[str]):
        self.item_ids = item_ids
        self.embeddings = embeddings
        self._faiss_index = None
        try:
            import faiss
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)  
            index.add(embeddings)
            self._faiss_index = index
        except Exception:
            self._faiss_index = None  

    def search(self, query_embeddings: np.ndarray, k: int) -> np.ndarray:
        """Retourne les indices (dans item_ids) des k plus proches voisins,
        pour chaque ligne de query_embeddings. Shape (n_queries, k)."""
        k = min(k, len(self.item_ids))
        if self._faiss_index is not None:
            _scores, idx = self._faiss_index.search(query_embeddings, k)
            return idx
        sims = query_embeddings @ self.embeddings.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        return idx



def evaluate_model(
    embedder: Embedder,
    u_items: List[SchemaItem],
    t_items: List[SchemaItem],
    gold: Dict[str, set],
    k_list: List[int],
) -> Dict:
    if embedder.is_fallback:
        embedder.fit_fallback([it.text for it in u_items] + [it.text for it in t_items])

    q_emb = embedder.encode([it.text for it in u_items])
    c_emb = embedder.encode([it.text for it in t_items])

    index = SchemaIndex(c_emb, [it.item_id for it in t_items])
    max_k = max(k_list)
    neighbor_idx = index.search(q_emb, max_k)  

    eval_qids = [it.item_id for it in u_items if it.item_id in gold and gold[it.item_id]]
    n_eval = len(eval_qids)
    if n_eval == 0:
        raise ValueError("Aucune requete evaluable : verifiez le fichier de labels.")

    recall_at_k = {k: 0 for k in k_list}
    reciprocal_ranks = []

    id_to_row = {it.item_id: r for r, it in enumerate(u_items)}
    cand_ids = [it.item_id for it in t_items]

    for qid in eval_qids:
        row = id_to_row[qid]
        order = neighbor_idx[row]
        gold_set = gold[qid]
        first_hit_rank = None
        for rank_pos, ci in enumerate(order):
            if cand_ids[ci] in gold_set:
                first_hit_rank = rank_pos + 1
                break
        reciprocal_ranks.append(1.0 / first_hit_rank if first_hit_rank else 0.0)
        for k in k_list:
            if first_hit_rank is not None and first_hit_rank <= k:
                recall_at_k[k] += 1

    recall_at_k = {k: round(v / n_eval, 4) for k, v in recall_at_k.items()}
    mrr = round(float(np.mean(reciprocal_ranks)), 4)

    return {
        "model": embedder.model_name,
        "is_fallback": embedder.is_fallback,
        "n_eval_queries": n_eval,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
    }


def min_k_for_thresholds(recall_at_k: Dict[int, float], k_list: List[int]) -> Dict[float, Optional[int]]:
    out = {}
    for th in RECALL_THRESHOLDS:
        min_k = None
        for k in sorted(k_list):
            if recall_at_k[k] >= th:
                min_k = k
                break
        out[th] = min_k
    return out


def pick_best(all_results: List[Dict], k_list: List[int]) -> Dict:
    """Meilleur (modele, k) = plus haut rappel atteint avec le plus petit k
    parmi tous les couples (modele, k) testes."""
    best = None
    for res in all_results:
        for k in k_list:
            r = res["recall_at_k"][k]
            candidate = {"model": res["model"], "k": k, "recall": r,
                         "is_fallback": res["is_fallback"]}
            if best is None:
                best = candidate
                continue
            if (candidate["recall"] > best["recall"] or
                    (candidate["recall"] == best["recall"] and candidate["k"] < best["k"])):
                best = candidate
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Compare plusieurs modeles d'embedding (dont LaBSE) et "
                    "plusieurs valeurs de k pour le matching U-Schema -> "
                    "schema cible, en s'appuyant sur des labels verifies.")
    parser.add_argument("--uschema", type=str, default=None,
                        help="JSON du U-Schema (source). Reconstruit depuis "
                             "--labels si absent.")
    parser.add_argument("--target-schema", type=str, default=None,
                        help="JSON du schema cible (ex: mimic_schema.json). "
                             "Reconstruit depuis --labels si absent.")
    parser.add_argument("--labels", type=str, required=True,
                        help="Fichier xlsx etiquete (verite terrain).")
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS,
                        help="Modeles d'embedding a comparer (ids sentence-transformers/HF).")
    parser.add_argument("--k-list", type=int, nargs="+", default=DEFAULT_K_LIST,
                        help="Valeurs de k a tester.")
    parser.add_argument("--offline-fallback", action="store_true",
                        help="Utiliser un embedding local (TF-IDF caracteres) "
                             "si un modele ne peut pas etre telecharge.")
    parser.add_argument("--out", type=str, default="embedding_k_selection_results.json",
                        help="Fichier de sortie JSON.")
    args = parser.parse_args()

    print("== Chargement et indexation du U-Schema (source) ==")
    u_items = load_uschema(args.uschema, args.labels)
    print(f"  -> {len(u_items)} attributs charges.")

    print("== Chargement et indexation du schema cible ==")
    t_items = load_target_schema(args.target_schema, args.labels)
    print(f"  -> {len(t_items)} colonnes chargees.")

    print("== Chargement des labels (verite terrain) ==")
    gold = load_gold_labels(args.labels)
    n_gold = sum(1 for v in gold.values() if v)
    print(f"  -> {n_gold} attributs avec au moins une correspondance verifiee.")

    all_results = []
    for model_name in args.models:
        print(f"\n== Modele : {model_name} ==")
        try:
            embedder = Embedder(model_name, offline_fallback=args.offline_fallback)
        except RuntimeError as e:
            print(f"[SKIP] {e}", file=sys.stderr)
            continue

        res = evaluate_model(embedder, u_items, t_items, gold, args.k_list)
        res["min_k_for_recall_threshold"] = min_k_for_thresholds(res["recall_at_k"], args.k_list)
        all_results.append(res)

        print(f"  Recall@k : { {k: res['recall_at_k'][k] for k in args.k_list} }")
        print(f"  MRR      : {res['mrr']}")

    if not all_results:
        print("Aucun modele n'a pu etre evalue.", file=sys.stderr)
        sys.exit(1)

    best = pick_best(all_results, args.k_list)

    print("\n" + "=" * 70)
    print("RESUME")
    print("=" * 70)
    header = f"{'modele':40s} " + " ".join(f"k={k:<5d}" for k in args.k_list) + "   MRR"
    print(header)
    for res in all_results:
        row = f"{res['model']:40s} " + " ".join(
            f"{res['recall_at_k'][k]:<6.3f}" for k in args.k_list
        ) + f"   {res['mrr']:.3f}"
        print(row)

    print("\nMeilleur couple (modele, k) : "
          f"model={best['model']} | k={best['k']} | recall={best['recall']}"
          + (" [fallback local]" if best["is_fallback"] else ""))

    output = {
        "k_list": args.k_list,
        "n_gold_queries": n_gold,
        "results": all_results,
        "best_model_k": best,
    }
    for res in output["results"]:
        res["recall_at_k"] = {str(k): v for k, v in res["recall_at_k"].items()}
        res["min_k_for_recall_threshold"] = {str(t): v for t, v in res["min_k_for_recall_threshold"].items()}

    Path(args.out).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultats complets ecrits dans : {args.out}")


if __name__ == "__main__":
    main()
