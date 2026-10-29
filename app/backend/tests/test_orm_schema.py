from adapters.outbound.persistence.sqlalchemy.orm import Base

PHI_MARKERS = ("patient", "birth", "dob", "filename", "file_name")


def test_schema_has_the_four_aggregate_tables():
    assert set(Base.metadata.tables) == {"studies", "predictions", "reports", "chat_messages"}


def test_schema_has_no_phi_or_original_filename_columns():
    offending = [
        f"{table.name}.{column.name}"
        for table in Base.metadata.tables.values()
        for column in table.columns
        if any(marker in column.name.lower() for marker in PHI_MARKERS)
    ]

    assert not offending, f"PHI-like columns found: {offending}"


def test_dependent_tables_cascade_on_study_delete():
    for name in ("predictions", "reports", "chat_messages"):
        (fk,) = Base.metadata.tables[name].foreign_keys
        assert fk.column.table.name == "studies"
        assert fk.ondelete == "CASCADE"
