"""
EvolutionPlanner — STEP 7 (started here per the mission's exact execution
order, item 13, after the MATCH/EVOLVE/REVIEW/REJECT chain is functional).

Turns a DecisionResult with decision_type == EVOLVE into a SchemaChange,
reusing ChangeType/SchemaChange exactly as they exist (see
docs/experimental_architecture.md section 5: "extend SchemaChange rather
than creating a competing EvolutionOperation" -- respected here).

Scope for this phase (explicitly limited by the mission):
    - ADD_COLUMN (attribute belongs to an existing/known target table)
    - CREATE_TABLE (attribute's parent entity has no known target table yet)
Not yet implemented in this phase: ADD_FOREIGN_KEY / ADD_ASSOCIATION_TABLE
generation logic (the ChangeType values exist since STEP 7 prep in
schema.py, but EvolutionPlanner does not yet emit them -- see docstring on
`plan` below).
"""

from __future__ import annotations

from typing import Optional

from src.domain.entities.evolution import SchemaChange
from src.domain.entities.experimental import DecisionResult, DecisionType
from src.domain.entities.schema import ChangeType, DataType, USchemaAttribute

_DATATYPE_TO_SQL = {
    DataType.STRING: "VARCHAR(255)",
    DataType.INTEGER: "INTEGER",
    DataType.DECIMAL: "DECIMAL(10,2)",
    DataType.BOOLEAN: "BOOLEAN",
    DataType.TIMESTAMP: "TIMESTAMP",
    DataType.DATE: "DATE",
    DataType.JSON: "JSON",
    DataType.UUID: "UUID",
}


class EvolutionPlanner:
    """Single Responsibility: DecisionResult (EVOLVE) -> SchemaChange."""

    def plan(
        self,
        decision: DecisionResult,
        source_attribute: USchemaAttribute,
        target_table: Optional[str],
    ) -> Optional[SchemaChange]:
        """
        Args:
            decision: a DecisionResult with decision_type == DecisionType.EVOLVE.
                Anything else returns None (this planner only handles EVOLVE;
                MATCH/REVIEW/REJECT are handled elsewhere or not at all).
            source_attribute: the U-Schema attribute that triggered EVOLVE.
            target_table: the relational table the new column should be added
                to, if the parent entity already resolved to an existing
                table (virtual rename); None means the entity itself is new
                and a CREATE_TABLE change should be produced instead.
        """
        if decision.decision_type != DecisionType.EVOLVE:
            return None

        sql_type = _DATATYPE_TO_SQL.get(source_attribute.data_type, "VARCHAR(255)")
        column_name = source_attribute.name

        if target_table:
            return SchemaChange(
                change_type=ChangeType.ADD_COLUMN,
                target_table=target_table,
                target_column=column_name,
                definition=f"{column_name} {sql_type}",
                data_type=sql_type,
                primary_key=source_attribute.is_key,
                reason=f"EVOLVE decision for '{decision.source_element}': {decision.rationale}",
                rationale=decision.rationale,
                source_element=decision.source_element,
                safe=True,
                requires_data_migration=False,
                estimated_impact="low",
            )

        # No known target table: propose creating one. Column definition
        # only includes this single attribute -- a real multi-attribute
        # CREATE_TABLE would batch several EVOLVE decisions for the same
        # parent entity (left to the caller/pipeline to aggregate; out of
        # scope for a single-decision planner method).
        new_table_name = self._infer_new_table_name(decision.source_element)
        return SchemaChange(
            change_type=ChangeType.CREATE_TABLE,
            target_table=new_table_name,
            target_column=column_name,
            definition=f"{column_name} {sql_type}",
            data_type=sql_type,
            primary_key=source_attribute.is_key,
            reason=f"EVOLVE decision for '{decision.source_element}': no existing target table; "
                   f"proposing new table '{new_table_name}'.",
            rationale=decision.rationale,
            source_element=decision.source_element,
            safe=True,
            requires_data_migration=False,
            estimated_impact="medium",
        )

    @staticmethod
    def _infer_new_table_name(source_element: str) -> str:
        parent = source_element.rsplit(".", 1)[0] if "." in source_element else source_element
        table = parent.replace(".", "_").lower()
        return table if table.endswith("s") else f"{table}s"