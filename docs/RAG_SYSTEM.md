# Système RAG pour le Matching de Schémas NoSQL → MIMIC-III

## Vue d'ensemble

Ce système implémente un pipeline RAG (Retrieval-Augmented Generation) complet pour le matching automatique de schémas NoSQL vers le schéma MIMIC-III, en respectant les contraintes de confidentialité et d'exécution avant le moteur de règles existant.

## Architecture

### Composants Principaux

1. **Base de Connaissances** (`knowledge_base_builder.py`)
   - Construction du corpus à partir du DDL MIMIC-III
   - Enrichissement avec dictionnaire de données
   - Ontologies médicales et techniques
   - Profils de colonnes agrégés (non sensibles)

2. **Service d'Embeddings** (`embedding_service.py`)
   - Bi-encodeur (Sentence-Transformers) pour retrieval rapide
   - Cross-encodeur pour reranking précis
   - Génération d'embeddings normalisés

3. **Stockage Vectoriel** (`vector_store.py`)
   - Index FAISS avec support IVF+PQ
   - Filtrage métadonnées avancé
   - Persistance et chargement

4. **Système de Retrieval** (`retriever.py`)
   - Filtres pré/post-retrieval
   - Cohérence FK/joins
   - Validation des unités et types

5. **Orchestrateur LLM** (`llm_orchestrator.py`)
   - Prompts structurés avec schéma JSON
   - Validation stricte des réponses
   - Gestion des erreurs et fallbacks

6. **Système de Scoring** (`scoring_system.py`)
   - Scoring hybride multi-signaux
   - Calibration avec données d'entraînement
   - Seuils adaptatifs

7. **Orchestrateur Principal** (`rag_orchestrator.py`)
   - Coordination de tous les composants
   - API simplifiée pour les cas d'usage courants
   - Gestion des erreurs et monitoring

## Utilisation

### Installation et Configuration

```bash
# Installer les dépendances
poetry install

# Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos clés API LLM
```

### Initialisation du Système

```bash
# Construire la base de connaissances et initialiser le système
python scripts/setup_rag_system.py
```

### Utilisation Programmée

```python
from src.infrastructure.rag.rag_orchestrator import RAGService
from src.domain.entities.rag_schema import SourceField, FieldType

# Initialiser le service (après setup)
rag_service = RAGService(orchestrator)

# Matching d'un champ unique
source_field = SourceField(
    path="patient.heart_rate",
    name_tokens=["heart", "rate"],
    inferred_type=FieldType.INTEGER,
    units="bpm",
    hints=["vital signs", "cardiac"]
)

result = rag_service._orchestrator.match_single_field(source_field)
print(f"Decision: {result.decision.action}")
print(f"Target: {result.decision.selected_target}")
```

### API REST

```bash
# Matching d'un champ
curl -X POST http://localhost:8000/api/v1/rag/match/single \
  -H "Content-Type: application/json" \
  -d '{
    "path": "patient.heart_rate",
    "name_tokens": ["heart", "rate"],
    "inferred_type": "integer",
    "units": "bpm",
    "hints": ["vital signs", "cardiac"]
  }'

# Matching par lot
curl -X POST http://localhost:8000/api/v1/rag/match/batch \
  -H "Content-Type: application/json" \
  -d '{
    "fields": [
      {
        "path": "patient.id",
        "name_tokens": ["patient", "id"],
        "inferred_type": "id"
      },
      {
        "path": "admission.date",
        "name_tokens": ["admission", "date"],
        "inferred_type": "datetime"
      }
    ]
  }'

# Recherche sémantique
curl -X POST http://localhost:8000/api/v1/rag/search/semantic \
  -H "Content-Type: application/json" \
  -d '{
    "hints": ["heart rate", "cardiac", "vital signs"],
    "top_k": 10
  }'
```

## Configuration

### Poids de Scoring

```python
from src.domain.entities.rag_schema import ScoringWeights

weights = ScoringWeights(
    bi_encoder=0.45,      # Similarité bi-encodeur
    cross_encoder=0.35,   # Reranking cross-encodeur
    llm_confidence=0.15,  # Confiance LLM
    constraints=0.05      # Validation contraintes
)
```

### Seuils de Décision

```python
from src.domain.entities.rag_schema import ScoringThresholds

thresholds = ScoringThresholds(
    accept_high=0.78,     # Seuil haut pour ACCEPT
    review_low=0.55,      # Seuil bas pour REVIEW
    reject_below=0.55     # En dessous = REJECT
)
```

## Format de Sortie

### Résultat de Matching

```json
{
  "source_field": "patient.heart_rate",
  "candidates": [
    {
      "target": "CHARTEVENTS.VALUENUM",
      "confidence_model": 0.82,
      "confidence_llm": 0.78,
      "rationale": "Heart rate measurement in vital signs"
    }
  ],
  "decision": {
    "action": "ACCEPT",
    "selected_target": "CHARTEVENTS.VALUENUM",
    "final_confidence": 0.80,
    "guardrails": ["type:integer", "units:bpm", "constraints_validated"]
  },
  "processing_time_ms": 245.6,
  "model_version": "1.0"
}
```

### Actions Possibles

- **ACCEPT** : Correspondance acceptée avec confiance élevée
- **REVIEW** : Nécessite une revue humaine
- **REJECT** : Correspondance rejetée, retour au moteur de règles

## Évaluation

### Métriques Disponibles

- **Accuracy** : Top-1, Top-5, Exact Match
- **Precision/Recall** : Précision et rappel par décision
- **Coverage** : Taux de couverture par type de champ
- **Efficiency** : Temps de traitement, nombre de candidats
- **Calibration** : Expected Calibration Error (ECE)

### Dataset d'Évaluation

```python
from src.infrastructure.rag.evaluation_metrics import EvaluationDataset

# Créer un dataset synthétique
dataset = EvaluationDataset()
dataset.create_synthetic_dataset(num_cases=100)

# Évaluer les résultats
metrics = EvaluationMetrics()
results = metrics.evaluate_results(matching_results, dataset.ground_truth)
```

## Sécurité et Conformité

### Confidentialité

- **Zéro PHI** : Aucune donnée personnelle dans le corpus
- **Profils agrégés** : Seulement statistiques anonymisées
- **Données synthétiques** : Exemples générés artificiellement

### Traçabilité

- **Audit logs** : Journal des décisions et prompts
- **Versioning** : Traçabilité des modèles et seuils
- **Déterminisme** : Temperature=0, schéma JSON imposé

### Fallback

- **SLA** : Cache des embeddings, timeouts LLM
- **Pipeline existant** : Retour au BERT+cosine si LLM indispo
- **Mode dégradé** : Fonctionnement sans LLM si nécessaire

## Déploiement Incrémental

### Phase 0 : Baseline
- Garder BERT cosine en parallèle (A/B testing)
- Métriques de comparaison

### Phase 1 : Sous-ensemble
- Activer RAG sur ADMISSIONS, PATIENTS, DIAGNOSES_ICD
- Validation sur cas simples

### Phase 2 : Reranking
- Ajouter cross-encodeur + guardrails FK/units
- Amélioration de la précision

### Phase 3 : LLM
- Introduire LLM + sortie JSON stricte
- Revue humaine pour cas REVIEW

### Phase 4 : Calibration
- Calibration des seuils sur golden set
- Élargissement à LABEVENTS, CHARTEVENTS, PRESCRIPTIONS

### Phase 5 : Apprentissage Actif
- Feedback loop avec annotateurs
- Amélioration continue

## Monitoring et Maintenance

### Métriques de Production

- Taux d'ACCEPT/REVIEW/REJECT
- Temps de traitement moyen
- Taux d'erreur LLM
- Utilisation des guardrails

### Maintenance

- Mise à jour du corpus (nouveaux schémas)
- Retraining des modèles d'embedding
- Ajustement des seuils
- Calibration périodique

## Limitations et Améliorations Futures

### Limitations Actuelles

- Corpus limité aux schémas MIMIC-III
- Pas de support multi-langues
- Dépendance aux modèles d'embedding

### Améliorations Futures

- **RAG Hybride** : Documents colonne + modèles de jointure
- **Graph-RAG** : Navigation par graphe relationnel
- **Multi-langues** : Support des schémas internationaux
- **Typage Probabiliste** : Auto-détection unités/formats
- **LLM-as-Judge** : Validation second passage

## Support et Documentation

- **Exemples** : `examples/rag_matching_example.py`
- **Tests** : `tests/unit/test_rag_*.py`
- **API Docs** : `http://localhost:8000/docs` (Swagger)
- **Logs** : Configuration logging dans `setup_rag_system.py`

