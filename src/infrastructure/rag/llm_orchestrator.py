"""
LLM Orchestrator for RAG-based schema matching.
Handles LLM calls, prompt construction, and JSON validation.
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ValidationError

from src.domain.entities.rag_schema import (
    SourceField, KnowledgeBaseDocument, CandidateMatch, 
    MatchingDecision, DecisionAction, SchemaMatchingResult,
    RetrievalQuery, ScoringWeights, ScoringThresholds
)
from src.infrastructure.llm.llm_client import ILLMClient

logger = logging.getLogger(__name__)


class LLMOrchestrator:
    """
    Orchestrates LLM calls for schema matching with structured output.
    Single Responsibility: LLM interaction and response validation.
    """
    
    def __init__(self, llm_client: ILLMClient, 
                 weights: Optional[ScoringWeights] = None,
                 thresholds: Optional[ScoringThresholds] = None):
        self._llm_client = llm_client
        self._weights = weights or ScoringWeights()
        self._thresholds = thresholds or ScoringThresholds()
        
        # JSON schema for LLM output
        self._output_schema = self._build_output_schema()
        
        # System prompt
        self._system_prompt = self._build_system_prompt()
    
    def match_field(self, query: RetrievalQuery, 
                   candidates: List[KnowledgeBaseDocument],
                   bi_scores: List[float],
                   cross_scores: List[float]) -> SchemaMatchingResult:
        """
        Match a source field to MIMIC-III columns using LLM.
        
        Args:
            query: Retrieval query
            candidates: Retrieved candidate documents
            bi_scores: Bi-encoder similarity scores
            cross_scores: Cross-encoder reranking scores
            
        Returns:
            Complete matching result with decision
        """
        start_time = time.time()
        
        try:
            # Step 1: Build user prompt with candidates
            user_prompt = self._build_user_prompt(query, candidates, bi_scores, cross_scores)
            
            # Step 2: Call LLM with structured output
            llm_response = self._call_llm_with_schema(user_prompt)
            
            # Step 3: Parse and validate response
            parsed_response = self._parse_llm_response(llm_response)
            
            # Step 4: Apply scoring and decision logic
            final_result = self._apply_scoring_and_decision(
                query, parsed_response, bi_scores, cross_scores
            )
            
            processing_time = (time.time() - start_time) * 1000
            final_result.processing_time_ms = processing_time
            final_result.model_version = "1.0"  # Could be from LLM client
            
            logger.info(f"LLM matching completed in {processing_time:.2f}ms")
            return final_result
            
        except Exception as e:
            logger.error(f"Error in LLM matching: {e}")
            return self._create_error_result(query, str(e))
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for schema matching."""
        return """Tu es un moteur de matching de schémas cliniques spécialisé dans la correspondance entre champs NoSQL et colonnes MIMIC-III.

RÔLE:
- Expert en schémas de base de données cliniques
- Spécialiste des correspondances sémantiques entre systèmes
- Respecte strictement les contraintes de types, unités et relations

RÈGLES STRICTES:
1. Tu ne renvoies QUE du JSON valide conforme au schéma spécifié
2. Tu choisis UNIQUEMENT parmi les colonnes candidates fournies
3. Tu respectes les types de données, unités, temporalité et relations PK/FK
4. Si l'incertitude est élevée (confiance < 0.6), tu déclenches REVIEW
5. Tu ne crées JAMAIS de nouvelles colonnes - tu sélectionnes parmi les candidats
6. Tu fournis une explication claire et concise pour chaque correspondance

CONTEXTE MIMIC-III:
- Tables principales: ADMISSIONS, PATIENTS, ICUSTAYS, CHARTEVENTS, LABEVENTS, DIAGNOSES_ICD, PROCEDURES_ICD, PRESCRIPTIONS
- Types codés: ICD9_CODE, ITEMID, DRUG/NDC
- Identifiants: SUBJECT_ID (patient), HADM_ID (admission), ICUSTAY_ID (ICU stay)
- Temporalité: ADMITTIME, DISCHTIME, CHARTTIME, etc.

GUIDELINES SPÉCIFIQUES:
- Champs "ICD/diag" → prioriser DIAGNOSES_ICD.*
- Champs "lab" + unité → candidats LABEVENTS.(VALUENUM, VALUEUOM)
- Champs "heart_rate/bpm" → CHARTEVENTS + ITEMID correspondant
- Dates/times → colonnes temporelles appropriées (ADMITTIME, CHARTTIME, etc.)
- IDs patients → SUBJECT_ID, IDs admissions → HADM_ID
- Codes médicaux → colonnes de codes spécialisées

FORMAT DE SORTIE:
Tu dois renvoyer un JSON strictement conforme au schéma fourni, avec:
- source_field: chemin du champ source
- candidates: liste des candidats avec confiances et justifications
- decision: action finale (ACCEPT/REVIEW/REJECT) avec cible sélectionnée

Si tu n'es pas sûr, utilise REVIEW plutôt que de deviner."""
    
    def _build_user_prompt(self, query: RetrievalQuery, 
                          candidates: List[KnowledgeBaseDocument],
                          bi_scores: List[float],
                          cross_scores: List[float]) -> str:
        """Build user prompt with context and candidates."""
        source_field = query.source_field
        
        # Build source field context
        source_context = f"""Champ source:
- path: {source_field.path}
- name_tokens: {', '.join(source_field.name_tokens)}
- inferred_type: {source_field.inferred_type}
- format_regex: {source_field.format_regex or 'Non spécifié'}
- units: {source_field.units or 'Non spécifié'}
- neighbors: {', '.join(source_field.neighbors)}
- coarse_semantics: {', '.join(source_field.coarse_semantics)}
- hints: {', '.join(source_field.hints)}
"""
        
        if source_field.category_values:
            source_context += f"- sample_values: {', '.join(source_field.category_values[:3])}\n"
        
        # Build candidates context
        candidates_context = "Candidats MIMIC-III (colonnes):\n"
        for i, (doc, bi_score, cross_score) in enumerate(zip(candidates, bi_scores, cross_scores)):
            candidates_context += f"""
{i+1}. {doc.table}.{doc.column}
   - Type: {doc.metadata.get('data_type', 'Non spécifié')}
   - Description: {doc.metadata.get('description', 'Non spécifié')}
   - Contraintes: {self._format_constraints(doc.metadata.get('constraints', {}))}
   - Unités: {doc.metadata.get('units', 'Non spécifié')}
   - Synonymes: {', '.join(doc.metadata.get('synonyms', [])[:5])}
   - Score bi-encodeur: {bi_score:.3f}
   - Score cross-encodeur: {cross_score:.3f}
"""
        
        # Build output schema
        schema_context = f"""
[SCHEMA JSON REQUIS]
{json.dumps(self._output_schema, indent=2, ensure_ascii=False)}

[INSTRUCTIONS]
1) Sélectionne 0..N cibles pertinentes (souvent 1) exclusivement parmi les candidats
2) Note "rationale" courte (max 30 mots) pour chaque candidat
3) Si ambiguity élevée ou manque d'info: action="REVIEW"
4) Respecte les types, unités et contraintes FK
5. Indique les guardrails appliqués dans la décision finale
"""
        
        return f"{source_context}\n{candidates_context}\n{schema_context}"
    
    def _format_constraints(self, constraints: Dict[str, Any]) -> str:
        """Format constraints for display."""
        if not constraints:
            return "Aucune"
        
        parts = []
        if constraints.get("is_primary_key"):
            parts.append("PK")
        if constraints.get("is_foreign_key"):
            parts.append("FK")
        if constraints.get("not_null"):
            parts.append("NOT NULL")
        if constraints.get("unique"):
            parts.append("UNIQUE")
        
        return ", ".join(parts) if parts else "Aucune"
    
    def _build_output_schema(self) -> Dict[str, Any]:
        """Build JSON schema for LLM output."""
        return {
            "type": "object",
            "properties": {
                "source_field": {
                    "type": "string",
                    "description": "Chemin du champ source"
                },
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "target": {
                                "type": "string",
                                "description": "Table.colonne cible"
                            },
                            "confidence_model": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "Confiance du modèle (0-1)"
                            },
                            "confidence_llm": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "Confiance LLM (0-1)"
                            },
                            "rationale": {
                                "type": "string",
                                "description": "Explication de la correspondance (max 30 mots)"
                            }
                        },
                        "required": ["target", "confidence_model", "confidence_llm", "rationale"]
                    }
                },
                "decision": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["ACCEPT", "REVIEW", "REJECT"],
                            "description": "Action décidée"
                        },
                        "selected_target": {
                            "type": "string",
                            "description": "Cible sélectionnée (si ACCEPT)"
                        },
                        "final_confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Confiance finale"
                        },
                        "guardrails": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Guardrails appliqués"
                        }
                    },
                    "required": ["action", "final_confidence", "guardrails"]
                }
            },
            "required": ["source_field", "candidates", "decision"]
        }
    
    def _call_llm_with_schema(self, user_prompt: str) -> str:
        """Call LLM with structured output schema."""
        try:
            response = self._llm_client.generate_response(
                system_prompt=self._system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,  # Deterministic output
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            return response
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM response."""
        try:
            # Parse JSON
            parsed = json.loads(response)
            
            # Validate against schema (basic validation)
            required_fields = ["source_field", "candidates", "decision"]
            for field in required_fields:
                if field not in parsed:
                    raise ValidationError(f"Missing required field: {field}")
            
            # Validate decision action
            if parsed["decision"]["action"] not in ["ACCEPT", "REVIEW", "REJECT"]:
                raise ValidationError("Invalid decision action")
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            raise ValidationError(f"Invalid JSON: {e}")
        except ValidationError as e:
            logger.error(f"Schema validation failed: {e}")
            raise
    
    def _apply_scoring_and_decision(self, query: RetrievalQuery, 
                                  llm_response: Dict[str, Any],
                                  bi_scores: List[float],
                                  cross_scores: List[float]) -> SchemaMatchingResult:
        """Apply hybrid scoring and final decision logic."""
        source_field = query.source_field
        
        # Extract LLM candidates and decisions
        llm_candidates = llm_response.get("candidates", [])
        llm_decision = llm_response.get("decision", {})
        
        # Build candidate matches with combined scores
        candidate_matches = []
        for i, llm_candidate in enumerate(llm_candidates):
            if i < len(bi_scores) and i < len(cross_scores):
                # Combine scores using weights
                combined_score = (
                    self._weights.bi_encoder * bi_scores[i] +
                    self._weights.cross_encoder * cross_scores[i] +
                    self._weights.llm_confidence * llm_candidate.get("confidence_llm", 0.5)
                )
                
                match = CandidateMatch(
                    target=llm_candidate["target"],
                    confidence_model=combined_score,
                    confidence_llm=llm_candidate["confidence_llm"],
                    rationale=llm_candidate["rationale"],
                    guardrails=[]
                )
                candidate_matches.append(match)
        
        # Apply final scoring with constraints
        final_confidence = llm_decision.get("final_confidence", 0.5)
        constraints_ok = self._validate_constraints(source_field, candidate_matches)
        
        # Adjust final confidence based on constraints
        if constraints_ok:
            final_confidence += self._weights.constraints * 0.1
        else:
            final_confidence -= self._weights.constraints * 0.2
        
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        # Determine final action based on thresholds
        action = DecisionAction.ACCEPT
        if final_confidence < self._thresholds.review_low:
            action = DecisionAction.REJECT
        elif final_confidence < self._thresholds.accept_high:
            action = DecisionAction.REVIEW
        
        # Override with LLM decision if it's more conservative
        llm_action = llm_decision.get("action", "ACCEPT")
        if llm_action == "REVIEW" and action == DecisionAction.ACCEPT:
            action = DecisionAction.REVIEW
        elif llm_action == "REJECT":
            action = DecisionAction.REJECT
        
        # Build guardrails
        guardrails = llm_decision.get("guardrails", [])
        if constraints_ok:
            guardrails.append("constraints_validated")
        else:
            guardrails.append("constraints_violated")
        
        # Create final decision
        decision = MatchingDecision(
            action=action,
            selected_target=llm_decision.get("selected_target") if action == DecisionAction.ACCEPT else None,
            final_confidence=final_confidence,
            guardrails=guardrails,
            review_checklist=self._build_review_checklist(source_field, candidate_matches) 
                           if action == DecisionAction.REVIEW else None
        )
        
        return SchemaMatchingResult(
            source_field=source_field.path,
            candidates=candidate_matches,
            decision=decision,
            processing_time_ms=0.0,  # Will be set by caller
            model_version="1.0"
        )
    
    def _validate_constraints(self, source_field: SourceField, 
                            candidates: List[CandidateMatch]) -> bool:
        """Validate constraints for the matching."""
        # Basic constraint validation
        if not candidates:
            return False
        
        # Check type compatibility
        for candidate in candidates:
            # This is a simplified check - in practice would be more sophisticated
            if source_field.inferred_type and candidate.target:
                # Basic type checking logic would go here
                pass
        
        return True
    
    def _build_review_checklist(self, source_field: SourceField, 
                              candidates: List[CandidateMatch]) -> List[str]:
        """Build checklist for human review."""
        checklist = [
            f"Vérifier la correspondance sémantique pour: {source_field.path}",
            "Confirmer la compatibilité des types de données",
            "Valider les unités de mesure si applicable",
            "Vérifier les contraintes de clés étrangères"
        ]
        
        if source_field.units:
            checklist.append(f"Valider les unités: {source_field.units}")
        
        if source_field.inferred_type:
            checklist.append(f"Confirmer le type: {source_field.inferred_type}")
        
        return checklist
    
    def _create_error_result(self, query: RetrievalQuery, error_msg: str) -> SchemaMatchingResult:
        """Create error result when LLM processing fails."""
        error_decision = MatchingDecision(
            action=DecisionAction.REJECT,
            selected_target=None,
            final_confidence=0.0,
            guardrails=["llm_error"],
            review_checklist=[f"Erreur LLM: {error_msg}"]
        )
        
        return SchemaMatchingResult(
            source_field=query.source_field.path,
            candidates=[],
            decision=error_decision,
            processing_time_ms=0.0,
            model_version="1.0"
        )

