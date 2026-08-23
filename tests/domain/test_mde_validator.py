import pytest

from src.domain.entities.experimental import MDEStatus
from src.domain.entities.schema import DataType, USchemaAttribute
from src.domain.services.mde_validator import MDEValidator, StructuralContext


@pytest.fixture
def validator():
    return MDEValidator()


def attr(name, data_type, is_key=False, is_object=False, is_array=False, nesting_path=()):
    return USchemaAttribute(
        name=name,
        data_type=data_type,
        is_key=is_key,
        is_object=is_object,
        is_array=is_array,
        nesting_path=tuple(nesting_path) or (name,),
    )


# 1. integer -> integer : VALID
def test_integer_to_integer_valid(validator):
    result = validator.validate(attr("id", DataType.INTEGER), "INTEGER")
    assert result.status == MDEStatus.VALID
    assert result.type_score == 1.0


# 2. integer -> decimal : VALID_WITH_TRANSFORMATION
def test_integer_to_decimal_valid_with_transformation(validator):
    result = validator.validate(attr("amount", DataType.INTEGER), "DECIMAL(10,2)")
    assert result.status == MDEStatus.VALID_WITH_TRANSFORMATION


# 3. string -> date : INVALID by default policy
def test_string_to_date_invalid_by_default(validator):
    result = validator.validate(attr("label", DataType.STRING), "DATE")
    assert result.status == MDEStatus.INVALID
    assert any("type_incompatible" in v for v in result.violations)


def test_string_to_date_review_when_configured():
    lenient = MDEValidator(review_on_incompatible_type=True)
    result = lenient.validate(attr("label", DataType.STRING), "DATE")
    assert result.status == MDEStatus.REVIEW


# 4. object -> scalar : INVALID
def test_object_to_scalar_invalid(validator):
    source = attr("device", DataType.JSON, is_object=True, nesting_path=("patient", "device"))
    result = validator.validate(source, "INTEGER", StructuralContext(column_name="device_id"))
    assert result.status == MDEStatus.INVALID
    assert result.structural_score == 0.0
    assert any("object_to_scalar" in v for v in result.violations)


# 5. array -> scalar : INVALID
def test_array_to_scalar_invalid(validator):
    source = attr("tags", DataType.STRING, is_array=True)
    result = validator.validate(source, "VARCHAR(255)")
    assert result.status == MDEStatus.INVALID
    assert any("array_to_scalar" in v for v in result.violations)


# 6. compatible key -> compatible key : VALID
def test_key_to_key_valid(validator):
    source = attr("patient_id", DataType.UUID, is_key=True)
    ctx = StructuralContext(is_primary_key=True, column_name="patient_id")
    result = validator.validate(source, "UUID", ctx)
    assert result.status == MDEStatus.VALID
    assert result.key_score == 1.0


# 7. foreign-key mismatch : REVIEW (default policy) / INVALID (strict policy)
def test_foreign_key_mismatch_review(validator):
    source = attr("patient_id", DataType.UUID, is_key=True)
    ctx = StructuralContext(is_primary_key=False, is_foreign_key=False, column_name="notes")
    result = validator.validate(source, "UUID", ctx)
    assert result.status == MDEStatus.REVIEW
    assert result.key_score < 1.0


def test_foreign_key_mismatch_invalid_when_strict():
    strict = MDEValidator(review_on_key_mismatch=False)
    source = attr("patient_id", DataType.UUID, is_key=True)
    ctx = StructuralContext(is_primary_key=False, is_foreign_key=False, column_name="notes")
    result = strict.validate(source, "UUID", ctx)
    assert result.status == MDEStatus.INVALID


# 8. semantically compatible but structurally invalid -> NOT MATCH
def test_semantic_looking_match_but_structurally_invalid(validator):
    # 'patient.device' (object) superficially resembles 'device_id' by name,
    # but structurally it's an object being mapped onto a scalar column.
    source = attr("device", DataType.JSON, is_object=True, nesting_path=("patient", "device"))
    result = validator.validate(source, "INTEGER", StructuralContext(column_name="device_id", is_foreign_key=True))
    assert result.status != MDEStatus.VALID
    assert result.status == MDEStatus.INVALID


def test_nested_scalar_child_is_compatible(validator):
    # 'patient.device.id' (scalar leaf) CAN be compatible with 'device_id'
    source = attr("id", DataType.INTEGER, nesting_path=("patient", "device", "id"))
    ctx = StructuralContext(column_name="device_id", is_foreign_key=True)
    result = validator.validate(source, "INTEGER", ctx)
    assert result.status in (MDEStatus.VALID, MDEStatus.VALID_WITH_TRANSFORMATION)


def test_unknown_types_do_not_crash_and_are_neutral(validator):
    result = validator.validate(attr("x", DataType.STRING), "SOME_EXOTIC_TYPE")
    assert result.status in (MDEStatus.VALID_WITH_TRANSFORMATION, MDEStatus.REVIEW, MDEStatus.VALID)