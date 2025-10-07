from typing import Any, Dict

from src.domain.repositeries.interfaces import IUSchemaRepository

from src.domain.entities.schema import USchema, USchemaAttribute,DataType,USchemaEntity,USchemaRelationship

class USchemaRepository(IUSchemaRepository):
    """
    Repository for U-Schema data access.
    Single Responsibility: U-Schema parsing and validation.
    """
    
    def parse_uschema(self, json_data: Dict[str, Any]) -> USchema:
        """Parse U-Schema from JSON."""
        entities = []
        
        for entity_data in json_data.get("entities", []):
            attributes = [
                USchemaAttribute(
                    name=attr["name"],
                    data_type=DataType(attr["type"]),
                    required=attr.get("required", False),
                    description=attr.get("description")
                )
                for attr in entity_data.get("attributes", [])
            ]
            
            relationships = [
                USchemaRelationship(
                    relationship_type=rel["type"],
                    target_entity=rel["entity"],
                    foreign_key=rel.get("key")
                )
                for rel in entity_data.get("relationships", [])
            ]
            
            entities.append(USchemaEntity(
                name=entity_data["name"],
                attributes=attributes,
                relationships=relationships,
                description=entity_data.get("description")
            ))
        
        return USchema(
            entities=entities,
            version=json_data.get("version", "1.0"),
            metadata=json_data.get("metadata", {})
        )
    
    def validate_uschema(self, schema: USchema) -> bool:
        """Validate U-Schema structure."""
        if not schema.entities:
            return False
        
        for entity in schema.entities:
            if not entity.name or not entity.attributes:
                return False
        
        return True
