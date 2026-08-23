from src.domain.services.relevance_policy import RelevancePolicy


def test_technical_pattern_rejects():
    policy = RelevancePolicy()
    result = policy.assess("document._metadata", semantic_evidence=0.9)
    assert result.relevant is False
    assert "pattern" in result.rationale.lower()


def test_high_semantic_evidence_is_relevant():
    policy = RelevancePolicy(min_semantic_evidence=0.3, ambiguous_band=0.1)
    result = policy.assess("patient.new_clinical_score", semantic_evidence=0.8)
    assert result.relevant is True


def test_low_semantic_evidence_is_irrelevant():
    policy = RelevancePolicy(min_semantic_evidence=0.3, ambiguous_band=0.1)
    result = policy.assess("patient.junk_field", semantic_evidence=0.05)
    assert result.relevant is False


def test_ambiguous_band_flags_ambiguous():
    policy = RelevancePolicy(min_semantic_evidence=0.3, ambiguous_band=0.1)
    result = policy.assess("patient.borderline_field", semantic_evidence=0.30)
    assert policy.is_ambiguous(result)


def test_llm_relevance_fn_overrides_semantic_evidence():
    policy = RelevancePolicy(llm_relevance_fn=lambda name: True)
    result = policy.assess("patient.weird_name", semantic_evidence=0.0)
    assert result.relevant is True
    assert "llm" in result.rationale.lower()


def test_llm_abstention_falls_back_to_semantic_evidence():
    policy = RelevancePolicy(llm_relevance_fn=lambda name: None, min_semantic_evidence=0.3, ambiguous_band=0.05)
    result = policy.assess("patient.new_field", semantic_evidence=0.9)
    assert result.relevant is True