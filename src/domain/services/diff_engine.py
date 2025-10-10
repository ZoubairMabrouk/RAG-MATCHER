from typing import List, Tuple, Dict, Set, Optional
from difflib import SequenceMatcher
import logging

from src.domain.entities.evolution import ChangeType, SchemaChange
from src.domain.entities.schema import USchema, SchemaMetadata, USchemaEntity, USchemaAttribute, Table
from src.domain.entities.rules import NamingConvention, DataType

logger = logging.getLogger(__name__)
class DiffEngine:
    """
    Computes symbolic differences between U-Schema and current RDB schema.
    Single Responsibility: Only handles diff computation.
    """
    
    def __init__(
        self,
        naming_convention: NamingConvention,
        rag_matcher=None,
        detect_virtual_renames: bool = True,
        emit_physical_renames: bool = False,   # <— on n'émet PAS de RENAME
        normalize_new_names: bool = True,
        table_threshold: float = 0.62,
        column_threshold: float = 0.68,
    ):
        self._naming = naming_convention
        self._rag_matcher = rag_matcher
        self._detect_virtual_renames = detect_virtual_renames
        self._emit_physical_renames = emit_physical_renames
        self._normalize_new_names = normalize_new_names
        self._table_threshold = table_threshold
        self._column_threshold = column_threshold

        # aliasing décidés pendant compute_diff
        self._entity_to_table_alias: Dict[str, str] = {}
        self._attr_to_column_alias: Dict[Tuple[str, str], str] = {}  # (table_effective, attr_name) -> existing_col

        # rapport pour EvolutionPlan.metadata
        self._normalization_report: Dict[str, Dict] = {
            "table_aliases": {},      # {entity_name: mapped_table_name}
            "column_aliases": {},     # {f"{table}.{attr}": mapped_column}
            "normalized_new": {       # suggestions de normalisation pour nouveaux objets
                "tables": {},         # {wanted: normalized}
                "columns": {}         # {f"{table}.{attr}": normalized}
            }
        }
        
        logger.info(f"[DiffEngine] Initialized with RAG matcher: {rag_matcher is not None}")
    def _name_similarity(self, a: str, b: str) -> float:
        def singularize(s: str) -> str:
            return s[:-1] if s.endswith('s') else s
        a_s, b_s = singularize(a), singularize(b)
        return max(
            SequenceMatcher(None, a, b).ratio(),
            SequenceMatcher(None, a_s, b_s).ratio()
        )

    def _jaccard(self, A: Set[str], B: Set[str]) -> float:
        if not A and not B:
            return 1.0
        inter = len(A & B)
        union = len(A | B) or 1
        return inter / union

    # ----------------------- Résolution d’alias virtuels ----------------------

    def _resolve_table_alias(
        self, entity: USchemaEntity, current_schema: SchemaMetadata
    ) -> Optional[str]:
        """Retourne le nom de la table EXISTANTE la plus probable (ou None)."""
        if self._rag_matcher:
            # Use RAG-based matching
            attr_names = [attr.name for attr in entity.attributes]
            result = self._rag_matcher.match_table(entity.name, attr_names)
            
            if result.target_name and result.confidence >= self._table_threshold:
                # alias virtuel retenu
                self._entity_to_table_alias[entity.name] = result.target_name
                self._normalization_report["table_aliases"][entity.name] = result.target_name
                logger.info(f"[DiffEngine] RAG table match: {entity.name} -> {result.target_name} (conf: {result.confidence:.3f})")
                return result.target_name
            else:
                logger.info(f"[DiffEngine] RAG table match failed: {entity.name} (conf: {result.confidence:.3f})")
                return None
        else:
            # Fallback to heuristic matching
            desired = self._entity_to_table_name(entity.name)
            want_cols = {a.name for a in entity.attributes}

            best_name, best_score = None, 0.0
            for tbl in current_schema.tables:
                have_cols = {c.name for c in tbl.columns}
                name_sim = self._name_similarity(desired, tbl.name)
                jac = self._jaccard(want_cols, have_cols)
                score = 0.4 * name_sim + 0.6 * jac
                if score > best_score:
                    best_name, best_score = tbl.name, score

            if best_score >= self._table_threshold:
                # alias virtuel retenu
                self._entity_to_table_alias[entity.name] = best_name
                self._normalization_report["table_aliases"][entity.name] = best_name
                logger.info(f"[DiffEngine] Heuristic table match: {entity.name} -> {best_name} (conf: {best_score:.3f})")
                return best_name
            return best_name

    def _resolve_column_aliases_for_table(
        self, entity: USchemaEntity, table: Table
    ) -> None:
        """Mappe les attributs aux colonnes existantes proches (alias virtuels)."""
        existing = {c.name for c in table.columns}
        for attr in entity.attributes:
            # si déjà présent, inutile
            if attr.name in existing:
                continue

            if self._rag_matcher:
                # Use RAG-based column matching
                result = self._rag_matcher.match_column(
                    table.name, 
                    attr.name, 
                    attr.data_type.value if hasattr(attr.data_type, 'value') else str(attr.data_type)
                )
                
                if result.target_name and result.confidence >= self._column_threshold:
                    # alias virtuel: on considère que l'attribut correspond à cette colonne
                    key = (table.name, attr.name)
                    self._attr_to_column_alias[key] = result.target_name
                    self._normalization_report["column_aliases"][f"{table.name}.{attr.name}"] = result.target_name
                    logger.info(f"[DiffEngine] RAG column match: {attr.name} -> {result.target_name} (conf: {result.confidence:.3f})")
            else:
                # Fallback to heuristic matching
                best_col, best_sim = None, 0.0
                for col in table.columns:
                    sim = self._name_similarity(attr.name, col.name)
                    if sim > best_sim:
                        best_col, best_sim = col.name, sim

                if best_sim >= self._column_threshold:
                    # alias virtuel: on considère que l'attribut correspond à cette colonne
                    key = (table.name, attr.name)
                    self._attr_to_column_alias[key] = best_col
                    self._normalization_report["column_aliases"][f"{table.name}.{attr.name}"] = best_col
                    logger.info(f"[DiffEngine] Heuristic column match: {attr.name} -> {best_col} (conf: {best_sim:.3f})")

    
    def compute_diff(self, uschema: USchema, current_schema: SchemaMetadata) -> List[SchemaChange]:
        """
        Compute differences between U-Schema and current database schema.
        Returns a list of required changes.
        """
        changes = []
        
        # Build lookup maps for efficient comparison
        existing_tables = {table.name for table in current_schema.tables}
        table_map = {table.name: table for table in current_schema.tables}
        
        for entity in uschema.entities:
            table_name = self._entity_to_table_name(entity.name)
            final_table_name = None
            if table_name not in existing_tables:
                # Need to create new table
                final_table_name = self._resolve_table_alias(entity, current_schema)
                if not final_table_name:
                    # No virtual mapping found, create new table with normalized name
                    normalized_name = table_name
                    if self._normalize_new_names:
                        normalized_name = self._naming.table_name(entity.name)
                        if normalized_name != table_name:
                            self._normalization_report["normalized_new"]["tables"][table_name] = normalized_name
                    final_table_name = normalized_name
                
                changes.extend(self._create_table_changes(entity, final_table_name))
            else:
                # Check for column differences
                changes.extend(self._compare_columns(entity, table_map[table_name]))
                
                # Check for relationship differences (foreign keys)
                changes.extend(self._compare_relationships(entity, table_map[table_name]))
        
        # Check for tables that exist but not in U-Schema (potential drops)
        uschema_tables = {self._entity_to_table_name(e.name) for e in uschema.entities}
        # for table_name in existing_tables:
        #     if table_name not in uschema_tables:
        #         changes.append(SchemaChange(
        #             change_type=ChangeType.DROP_TABLE,
        #             target_table=table_name,
        #             reason=f"Table '{table_name}' not present in U-Schema",
        #             safe=False,
        #             estimated_impact="high"
        #         ))
        
        return changes
    
    def _entity_to_table_name(self, entity_name: str) -> str:
        """Convert entity name to table name using naming convention."""
        return self._naming.table_name(entity_name)
    
    def _create_table_changes(self, entity: USchemaEntity, table_name: str) -> List[SchemaChange]:
        """Generate changes to create a new table."""
        changes = []
        
        # Create table
        column_defs = []
        for attr in entity.attributes:
            col_def = self._attribute_to_column_def(attr)
            column_defs.append(col_def)
        
        changes.append(SchemaChange(
            change_type=ChangeType.CREATE_TABLE,
            target_table=table_name,
            definition=", ".join(column_defs),
            reason=f"Entity '{entity.name}' requires new table",
            safe=True,
            estimated_impact="medium"
        ))
        
        return changes
    
    def _get_sql_type(self, data_type: DataType) -> str:
        """Convert U-Schema data type to SQL type."""
        type_map = {
            DataType.STRING: "VARCHAR(255)",
            DataType.INTEGER: "INTEGER",
            DataType.DECIMAL: "DECIMAL(10,2)",
            DataType.BOOLEAN: "BOOLEAN",
            DataType.TIMESTAMP: "TIMESTAMP",
            DataType.DATE: "DATE",
            DataType.JSON: "JSONB",
            DataType.UUID: "UUID"
        }
        
        return type_map.get(data_type, "TEXT")
    
    def _attribute_to_column_def(self, attr: USchemaAttribute) -> str:
        """Convert U-Schema attribute to SQL column definition."""
        sql_type = self._get_sql_type(attr.data_type)
        null_clause = "NOT NULL" if attr.required else ""
        
        return f"{attr.name} {sql_type} {null_clause}".strip()
    
    def _compare_columns(self, entity: USchemaEntity, table: Table) -> List[SchemaChange]:
        """Compare entity attributes with table columns."""
        changes = []
        
        existing_columns = {col.name for col in table.columns}
        required_columns = {attr.name for attr in entity.attributes}
        
        # Columns to add or modify
        for attr in entity.attributes:
            # Check for virtual column alias first
            key = (table.name, attr.name)
            mapped_column = self._attr_to_column_alias.get(key)
            
            if mapped_column and mapped_column in existing_columns:
                # Column exists via virtual alias - check if modification needed
                existing_col = next(col for col in table.columns if col.name == mapped_column)
                
                # Check if type modification is needed
                expected_type = self._get_sql_type(attr.data_type)
                if existing_col.data_type != expected_type:
                    changes.append(SchemaChange(
                        change_type=ChangeType.MODIFY_COLUMN,
                        target_table=table.name,
                        target_column=mapped_column,
                        definition=f"TYPE {expected_type}",
                        reason=f"Attribute '{attr.name}' mapped to '{mapped_column}' requires type change",
                        safe=False,
                        estimated_impact="medium"
                    ))
            elif attr.name not in existing_columns:
                # Add new column
                changes.append(SchemaChange(
                    change_type=ChangeType.ADD_COLUMN,
                    target_table=table.name,
                    target_column=attr.name,
                    definition=self._attribute_to_column_def(attr),
                    reason=f"Attribute '{attr.name}' not present in table '{table.name}'",
                    safe=True,
                    estimated_impact="low"
                ))
        
        return changes
    
    def _compare_relationships(self, entity: USchemaEntity, table: Table) -> List[SchemaChange]:
        """Compare entity relationships with foreign keys."""
        changes = []
        
        existing_fks = {fk.column: fk for fk in table.foreign_keys}
        
        for rel in entity.relationships:
            if rel.relationship_type == "belongsTo" and rel.foreign_key:
                if rel.foreign_key not in existing_fks:
                    ref_table = self._entity_to_table_name(rel.target_entity)
                    changes.append(SchemaChange(
                        change_type=ChangeType.ADD_CONSTRAINT,
                        target_table=table.name,
                        target_column=rel.foreign_key,
                        definition=f"FOREIGN KEY ({rel.foreign_key}) REFERENCES {ref_table}(id)",
                        reason=f"Relationship to '{rel.target_entity}' requires foreign key",
                        safe=True,
                        estimated_impact="low"
                    ))
        
        return changes
