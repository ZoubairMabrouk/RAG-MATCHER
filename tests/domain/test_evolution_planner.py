from src.domain.entities.evolution import SchemaChange
from src.domain.entities.experimental import DecisionResult, DecisionType
from src.domain.entities.schema import ChangeType, DataType, USchemaAttribute
from src.domain.services.evolution_planner import EvolutionPlanner


def make_evolve_decision(source_element):
    return DecisionResult(
        source_element=source_element,
        selected_candidate=None,
        semantic_decision=None,
        mde_validation=None,
        decision_type=DecisionType.EVOLVE,
        confidence=0.8,
        rationale="new meaningful concept",
    )


def test_add_column_when_target_table_known():
    planner = EvolutionPlanner()
    decision = make_evolve_decision("patient.new_clinical_score")
    attr = USchemaAttribute(name="new_clinical_score", data_type=DataType.DECIMAL)

    change = planner.plan(decision, attr, target_table="patient")

    assert isinstance(change, SchemaChange)
    assert change.change_type == ChangeType.ADD_COLUMN
    assert change.target_table == "patient"
    assert change.target_column == "new_clinical_score"
    assert "DECIMAL" in change.definition
    assert change.source_element == "patient.new_clinical_score"


def test_create_table_when_no_target_table():
    planner = EvolutionPlanner()
    decision = make_evolve_decision("wearable_device.battery_level")
    attr = USchemaAttribute(name="battery_level", data_type=DataType.INTEGER)

    change = planner.plan(decision, attr, target_table=None)

    assert change.change_type == ChangeType.CREATE_TABLE
    assert change.target_table == "wearable_devices"
    assert change.target_column == "battery_level"


def test_non_evolve_decision_returns_none():
    planner = EvolutionPlanner()
    decision = DecisionResult(
        source_element="x",
        selected_candidate=None,
        semantic_decision=None,
        mde_validation=None,
        decision_type=DecisionType.MATCH,
        confidence=0.9,
    )
    attr = USchemaAttribute(name="x", data_type=DataType.STRING)
    assert planner.plan(decision, attr, target_table="t") is None


def test_planned_change_is_accepted_by_migration_builder():
    from src.domain.services.migration_builder import MigrationBuilder

    planner = EvolutionPlanner()
    decision = make_evolve_decision("patient.new_clinical_score")
    attr = USchemaAttribute(name="new_clinical_score", data_type=DataType.DECIMAL)
    change = planner.plan(decision, attr, target_table="patient")

    builder = MigrationBuilder("postgresql")
    sql_statements = builder.build_migration([change])

    assert len(sql_statements) == 1
    assert "ALTER TABLE patient ADD COLUMN" in sql_statements[0]
    assert "new_clinical_score" in sql_statements[0]