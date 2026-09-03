from src.evaluation.dataset import Dataset, GoldPair, SourceRecord, TargetRecord
from src.evaluation.rank_metrics import compute_all_metrics, gold_ranks_from_predictions, hits_at_k, mean_rank, median_rank, mrr, recall_at_k
from src.evaluation.splits.splitters import (
    PairLevelSplitter,
    SchemaLevelSplitter,
    SourceAttributeSplitter,
    TableLevelSplitter,
)


def make_synthetic_dataset() -> Dataset:
    """3 sources, 4 targets, 5 gold (positive) pairs, plus negatives to fill out the grid."""
    sources = {
        "t1-a": SourceRecord("t1-a", "a", "t1", "ctx a"),
        "t1-b": SourceRecord("t1-b", "b", "t1", "ctx b"),
        "t2-c": SourceRecord("t2-c", "c", "t2", "ctx c"),
    }
    targets = {
        "x-1": TargetRecord("x-1", "1", "x", "ctx 1"),
        "x-2": TargetRecord("x-2", "2", "x", "ctx 2"),
        "y-3": TargetRecord("y-3", "3", "y", "ctx 3"),
        "y-4": TargetRecord("y-4", "4", "y", "ctx 4"),
    }
    pairs = [
        GoldPair("t1-a", "x-1", 1),
        GoldPair("t1-a", "x-2", 0),
        GoldPair("t1-a", "y-3", 0),
        GoldPair("t1-a", "y-4", 0),
        GoldPair("t1-b", "x-1", 0),
        GoldPair("t1-b", "x-2", 1),
        GoldPair("t1-b", "y-3", 0),
        GoldPair("t1-b", "y-4", 0),
        GoldPair("t2-c", "x-1", 0),
        GoldPair("t2-c", "x-2", 0),
        GoldPair("t2-c", "y-3", 1),
        GoldPair("t2-c", "y-4", 1),
    ]
    return Dataset(sources=sources, targets=targets, pairs=pairs)


# ---------------------------------------------------------------------------
# Splitters
# ---------------------------------------------------------------------------

def test_pair_level_split_covers_all_pairs():
    dataset = make_synthetic_dataset()
    split = PairLevelSplitter().split(dataset, seed=1, train=0.5, validation=0.25, test=0.25)
    total = len(split.train_pairs) + len(split.validation_pairs) + len(split.test_pairs)
    assert total == len(dataset.pairs)


def test_source_attribute_split_no_source_leakage():
    dataset = make_synthetic_dataset()
    split = SourceAttributeSplitter().split(dataset, seed=1, train=0.34, validation=0.33, test=0.33)
    assert split.train_sources.isdisjoint(split.test_sources)
    assert split.train_sources.isdisjoint(split.validation_sources)
    assert split.validation_sources.isdisjoint(split.test_sources)


def test_table_level_split_no_table_leakage():
    dataset = make_synthetic_dataset()
    split = TableLevelSplitter().split(dataset, seed=1, train=0.5, validation=0.0, test=0.5)
    train_tables = {dataset.sources[s].source_table for s in split.train_sources}
    test_tables = {dataset.sources[s].source_table for s in split.test_sources}
    assert train_tables.isdisjoint(test_tables)


def test_schema_level_split_blocked_without_family_mapping():
    dataset = make_synthetic_dataset()
    import pytest
    with pytest.raises(ValueError, match="BLOCKED_BY_DATASET_METADATA"):
        SchemaLevelSplitter().split(dataset)


def test_schema_level_split_works_with_explicit_family_mapping():
    dataset = make_synthetic_dataset()
    family_of = {"t1": "family_A", "t2": "family_B"}
    split = SchemaLevelSplitter().split(dataset, family_of=family_of, seed=1, train=0.5, validation=0.0, test=0.5)
    total = len(split.train_pairs) + len(split.validation_pairs) + len(split.test_pairs)
    assert total == len(dataset.pairs)


# ---------------------------------------------------------------------------
# Rank metrics (manually verified against the synthetic predictions below)
# ---------------------------------------------------------------------------

def make_synthetic_predictions():
    """
    3 sources. Gold targets: t1-a -> x-1, t1-b -> x-2, t2-c -> {y-3, y-4}.
    Ranked retrieval lists (rank 1 = best):
      t1-a: [y-4, x-1, x-2]      -> gold x-1 at rank 2
      t1-b: [x-2, y-3, x-1]      -> gold x-2 at rank 1
      t2-c: [x-1, x-2, y-3]      -> gold y-3 at rank 3, y-4 NOT retrieved at all
    """
    return [
        {"source_id": "t1-a", "target_id": "y-4", "rank": 1, "similarity": 0.9, "is_gold": False},
        {"source_id": "t1-a", "target_id": "x-1", "rank": 2, "similarity": 0.8, "is_gold": True},
        {"source_id": "t1-a", "target_id": "x-2", "rank": 3, "similarity": 0.7, "is_gold": False},

        {"source_id": "t1-b", "target_id": "x-2", "rank": 1, "similarity": 0.95, "is_gold": True},
        {"source_id": "t1-b", "target_id": "y-3", "rank": 2, "similarity": 0.6, "is_gold": False},
        {"source_id": "t1-b", "target_id": "x-1", "rank": 3, "similarity": 0.5, "is_gold": False},

        {"source_id": "t2-c", "target_id": "x-1", "rank": 1, "similarity": 0.4, "is_gold": False},
        {"source_id": "t2-c", "target_id": "x-2", "rank": 2, "similarity": 0.3, "is_gold": False},
        {"source_id": "t2-c", "target_id": "y-3", "rank": 3, "similarity": 0.2, "is_gold": True},
        # y-4 (also gold for t2-c) never retrieved -> not in this list at all
    ]


def test_gold_ranks_uses_min_rank_when_multiple_gold_in_topk():
    preds = make_synthetic_predictions()
    ranks = {r.source_id: r.gold_rank for r in gold_ranks_from_predictions(preds)}
    assert ranks["t1-a"] == 2
    assert ranks["t1-b"] == 1
    assert ranks["t2-c"] == 3  # y-3 found at rank 3; y-4 not retrieved, min() of found gold ranks


def test_hits_at_k_manual():
    preds = make_synthetic_predictions()
    ranks = gold_ranks_from_predictions(preds)
    # ranks: t1-a=2, t1-b=1, t2-c=3 -- all 3 have a rank (>=1 gold retrieved each)
    assert hits_at_k(ranks, 1) == 1 / 3       # only t1-b
    assert hits_at_k(ranks, 2) == 2 / 3       # t1-a, t1-b
    assert hits_at_k(ranks, 3) == 3 / 3       # all three


def test_mrr_manual():
    preds = make_synthetic_predictions()
    ranks = gold_ranks_from_predictions(preds)
    expected = (1 / 2 + 1 / 1 + 1 / 3) / 3
    assert abs(mrr(ranks) - expected) < 1e-9


def test_mean_and_median_rank_manual():
    preds = make_synthetic_predictions()
    ranks = gold_ranks_from_predictions(preds)
    assert mean_rank(ranks) == (2 + 1 + 3) / 3
    assert median_rank(ranks) == 2


def test_recall_at_k_counts_relationships_not_queries():
    preds = make_synthetic_predictions()
    # gold relationships: (t1-a,x-1) rank2, (t1-b,x-2) rank1, (t2-c,y-3) rank3 -- (t2-c,y-4) not in list at all
    # total gold relationships PRESENT in predictions = 3 (y-4 isn't in the predictions list to count against)
    assert recall_at_k(preds, 1) == 1 / 3
    assert recall_at_k(preds, 2) == 2 / 3
    assert recall_at_k(preds, 3) == 3 / 3


def test_missing_gold_rank_is_null_not_k_plus_1():
    preds = [
        {"source_id": "s_no_gold_found", "target_id": "z", "rank": 1, "similarity": 0.5, "is_gold": False},
    ]
    ranks = gold_ranks_from_predictions(preds)
    assert ranks[0].gold_rank is None
    assert hits_at_k(ranks, 1) is None  # no valid queries -> None, not silently 0


def test_compute_all_metrics_reports_missing_count():
    preds = make_synthetic_predictions()
    metrics = compute_all_metrics(preds, ks=[1, 2, 3])
    assert metrics["num_queries"] == 3
    assert metrics["num_queries_gold_not_retrieved"] == 0
    assert metrics["hits@1"] == 1 / 3