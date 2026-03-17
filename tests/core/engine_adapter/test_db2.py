# type: ignore
import typing as t
from unittest import mock

import pytest
from pytest_mock.plugin import MockerFixture
from sqlglot import expressions as exp
from sqlglot import parse_one

from sqlmesh.core.engine_adapter.db2_proper import DB2EngineAdapter
from sqlmesh.core.engine_adapter.shared import DataObject, DataObjectType
from sqlmesh.utils.errors import SQLMeshError
from tests.core.engine_adapter import to_sql_calls

pytestmark = [pytest.mark.engine, pytest.mark.db2]


@pytest.fixture
def adapter(make_mocked_engine_adapter: t.Callable) -> DB2EngineAdapter:
    return make_mocked_engine_adapter(DB2EngineAdapter)


def test_columns(adapter: DB2EngineAdapter):
    """Test column type mapping from DB2 system catalog."""
    adapter.cursor.fetchall.return_value = [
        ("id", "INTEGER", 4, 0),
        ("name", "VARCHAR", 100, 0),
        ("amount", "DECIMAL", 10, 2),
        ("created_at", "TIMESTAMP", 10, 6),
        ("data", "CLOB", 1048576, 0),
        ("binary_data", "BLOB", 1048576, 0),
        ("flag", "SMALLINT", 2, 0),
        ("big_num", "BIGINT", 8, 0),
        ("price", "DOUBLE", 8, 0),
        ("code", "CHAR", 10, 0),
    ]

    result = adapter.columns("test_schema.test_table")
    
    assert result == {
        "id": exp.DataType.build("INT", dialect=adapter.dialect),
        "name": exp.DataType.build("VARCHAR(100)", dialect=adapter.dialect),
        "amount": exp.DataType.build("DECIMAL(10,2)", dialect=adapter.dialect),
        "created_at": exp.DataType.build("TIMESTAMP", dialect=adapter.dialect),
        "data": exp.DataType.build("CLOB", dialect=adapter.dialect),
        "binary_data": exp.DataType.build("BLOB", dialect=adapter.dialect),
        "flag": exp.DataType.build("SMALLINT", dialect=adapter.dialect),
        "big_num": exp.DataType.build("BIGINT", dialect=adapter.dialect),
        "price": exp.DataType.build("DOUBLE", dialect=adapter.dialect),
        "code": exp.DataType.build("CHAR(10)", dialect=adapter.dialect),
    }


def test_table_exists_case_insensitive(adapter: DB2EngineAdapter, mocker: MockerFixture):
    """Test table existence check with case-insensitive matching."""
    # Mock fetchone to return actual case from DB2
    adapter.cursor.fetchone.return_value = ("TEST_SCHEMA", "TEST_TABLE")
    
    # Check with lowercase input
    result = adapter.table_exists("test_schema.test_table")
    
    assert result is True
    # Verify the query uses UPPER() for case-insensitive comparison
    executed_sql = to_sql_calls(adapter)[0]
    assert "UPPER" in executed_sql
    assert "SYSCAT.TABLES" in executed_sql


def test_table_exists_not_found(adapter: DB2EngineAdapter):
    """Test table existence check when table doesn't exist."""
    adapter.cursor.fetchone.return_value = None
    
    result = adapter.table_exists("test_schema.nonexistent_table")
    
    assert result is False


def test_create_index_without_if_not_exists(adapter: DB2EngineAdapter):
    """Test index creation without IF NOT EXISTS (DB2 doesn't support it)."""
    # Mock that index doesn't exist
    adapter.cursor.fetchone.return_value = (0,)
    
    adapter.create_index(
        table_name="test_schema.test_table",
        index_name="idx_test",
        columns=("col1", "col2"),
    )
    
    sql_calls = to_sql_calls(adapter)
    # Should have check query and create query
    assert len(sql_calls) >= 2
    # Create index should not have IF NOT EXISTS
    create_sql = [s for s in sql_calls if "CREATE INDEX" in s][0]
    assert "IF NOT EXISTS" not in create_sql


def test_create_index_already_exists(adapter: DB2EngineAdapter):
    """Test index creation when index already exists."""
    # Mock that index exists
    adapter.cursor.fetchone.return_value = (1,)
    
    adapter.create_index(
        table_name="test_schema.test_table",
        index_name="idx_test",
        columns=("col1",),
    )
    
    sql_calls = to_sql_calls(adapter)
    # Should only have check query, no create
    assert all("CREATE INDEX" not in s for s in sql_calls)


def test_create_table_primary_key_not_null(adapter: DB2EngineAdapter):
    """Test that primary key columns get NOT NULL constraint in DB2."""
    adapter.create_table(
        "test_table",
        {
            "id": exp.DataType.build("INT"),
            "name": exp.DataType.build("VARCHAR(100)"),
        },
        primary_key=("id",),
    )
    
    sql_calls = to_sql_calls(adapter)
    create_sql = sql_calls[0]
    
    # Primary key column should have NOT NULL
    assert "NOT NULL" in create_sql
    assert '"id" INT NOT NULL' in create_sql or "[id] INT NOT NULL" in create_sql


def test_create_table_as_select_with_data(adapter: DB2EngineAdapter, mocker: MockerFixture):
    """Test CTAS includes WITH DATA clause for DB2."""
    mocker.patch.object(adapter, "table_exists", return_value=False)
    
    adapter.ctas(
        table_name="test_table",
        query_or_df=parse_one("SELECT id, name FROM source_table"),
        exists=False,
    )
    
    sql_calls = to_sql_calls(adapter)
    create_sql = sql_calls[0]
    
    # Should include WITH DATA
    assert "WITH DATA" in create_sql
    # Should not have subquery alias
    assert '"_subquery"' not in create_sql and "[_subquery]" not in create_sql


def test_create_schema(adapter: DB2EngineAdapter):
    """Test schema creation."""
    # Mock that schema doesn't exist
    adapter.cursor.fetchone.return_value = None
    
    adapter.create_schema("test_schema", ignore_if_exists=True)
    
    sql_calls = to_sql_calls(adapter)
    # Should check existence and create
    assert any("SYSCAT.SCHEMATA" in s for s in sql_calls)
    assert any("CREATE SCHEMA" in s for s in sql_calls)


def test_drop_schema_cascade(adapter: DB2EngineAdapter):
    """Test schema drop with cascade."""
    # Mock schema exists
    adapter.cursor.fetchone.return_value = (1,)
    # Mock tables in schema
    adapter.cursor.fetchall.return_value = [("TABLE1",), ("TABLE2",)]
    
    adapter.drop_schema("test_schema", cascade=True)
    
    sql_calls = to_sql_calls(adapter)
    # Should drop tables first, then schema with RESTRICT
    assert any("DROP TABLE" in s for s in sql_calls)
    assert any("DROP SCHEMA" in s and "RESTRICT" in s for s in sql_calls)


def test_create_view_replace(adapter: DB2EngineAdapter, mocker: MockerFixture):
    """Test view creation with replace drops old view first."""
    mocker.patch.object(adapter, "drop_view")
    
    adapter.create_view(
        "test_view",
        parse_one("SELECT * FROM test_table"),
        replace=True,
    )
    
    # Should call drop_view before creating
    adapter.drop_view.assert_called_once()


def test_merge_with_simple_aliases(adapter: DB2EngineAdapter):
    """Test MERGE uses simple aliases (TARGET/SOURCE) instead of double underscores."""
    adapter.merge(
        target_table="target_table",
        source_table=parse_one("SELECT id, value FROM source_table"),
        target_columns_to_types={
            "id": exp.DataType.build("INT"),
            "value": exp.DataType.build("VARCHAR(100)"),
        },
        unique_key=[exp.to_identifier("id", quoted=True)],
    )
    
    sql_calls = to_sql_calls(adapter)
    merge_sql = sql_calls[0]
    
    # Should use TARGET and SOURCE aliases
    assert "TARGET" in merge_sql
    assert "SOURCE" in merge_sql
    # Should not use double underscore aliases
    assert "__MERGE_TARGET__" not in merge_sql
    assert "__MERGE_SOURCE__" not in merge_sql


def test_get_current_catalog(adapter: DB2EngineAdapter):
    """Test getting current catalog uses CURRENT SERVER."""
    adapter.cursor.fetchone.return_value = ("TESTDB",)
    
    result = adapter.get_current_catalog()
    
    assert result == "TESTDB"
    sql_calls = to_sql_calls(adapter)
    # Should query CURRENT SERVER FROM SYSIBM.SYSDUMMY1
    assert any("CURRENT SERVER" in s and "SYSIBM.SYSDUMMY1" in s for s in sql_calls)


def test_get_current_schema(adapter: DB2EngineAdapter):
    """Test getting current schema."""
    adapter.cursor.fetchone.return_value = ("TESTSCHEMA",)
    
    result = adapter._get_current_schema()
    
    assert result == "testschema"  # Returns lowercase
    sql_calls = to_sql_calls(adapter)
    # Should query CURRENT SCHEMA FROM SYSIBM.SYSDUMMY1
    assert any("CURRENT SCHEMA" in s and "SYSIBM.SYSDUMMY1" in s for s in sql_calls)


def test_server_version(adapter: DB2EngineAdapter, mocker: MockerFixture):
    """Test server version parsing."""
    fetchone_mock = mocker.patch.object(adapter, "fetchone")
    
    # Test DB2 11.5
    fetchone_mock.return_value = ("DB2 v11.5.0.0",)
    assert adapter.server_version == (11, 5)
    
    # Test DB2 12.1
    del adapter.server_version
    fetchone_mock.return_value = ("DB2 v12.1.0.0",)
    assert adapter.server_version == (12, 1)


def test_type_mapping_comprehensive(adapter: DB2EngineAdapter):
    """Test comprehensive type mapping including new types."""
    test_types = [
        ("DECFLOAT", 16, 0, "DOUBLE"),
        ("GRAPHIC", 50, 0, "CHAR(50)"),
        ("VARGRAPHIC", 100, 0, "VARCHAR(100)"),
        ("DBCLOB", 1048576, 0, "CLOB"),
        ("XML", 0, 0, "TEXT"),
        ("ROWID", 40, 0, "VARCHAR(40)"),
        ("BOOLEAN", 1, 0, "BOOLEAN"),
    ]
    
    for db2_type, length, scale, expected in test_types:
        result = adapter._db2_type_to_sqlglot(db2_type, length, scale)
        assert result.sql(dialect="db2") == expected


def test_staging_table_drop_before_create(adapter: DB2EngineAdapter, mocker: MockerFixture):
    """Test that staging tables are dropped before creation."""
    mocker.patch.object(adapter, "table_exists", return_value=False)
    
    # Create a staging table (contains sqlmesh__)
    adapter.ctas(
        table_name="test_schema.sqlmesh__test_table__abc123",
        query_or_df=parse_one("SELECT * FROM source"),
        exists=False,
    )
    
    sql_calls = to_sql_calls(adapter)
    # Should attempt to drop before creating
    assert any("DROP" in s for s in sql_calls)


def test_comments_on_table(adapter: DB2EngineAdapter):
    """Test table and column comments use COMMENT command."""
    adapter.create_table(
        "test_table",
        {
            "id": exp.DataType.build("INT"),
            "name": exp.DataType.build("VARCHAR(100)"),
        },
        table_description="Test table",
        column_descriptions={"id": "Primary key", "name": "User name"},
    )
    
    sql_calls = to_sql_calls(adapter)
    
    # Should have CREATE TABLE and COMMENT statements
    assert any("CREATE TABLE" in s for s in sql_calls)
    assert any("COMMENT ON TABLE" in s and "Test table" in s for s in sql_calls)
    assert any("COMMENT ON COLUMN" in s and "Primary key" in s for s in sql_calls)
    assert any("COMMENT ON COLUMN" in s and "User name" in s for s in sql_calls)


def test_catalog_support(adapter: DB2EngineAdapter):
    """Test that DB2 only supports single catalog."""
    from sqlmesh.core.engine_adapter.shared import CatalogSupport
    
    assert adapter.catalog_support == CatalogSupport.SINGLE_CATALOG_ONLY


def test_create_table_like(adapter: DB2EngineAdapter):
    """Test CREATE TABLE LIKE for DB2."""
    adapter._create_table_like(
        target_table_name="target_table",
        source_table_name="source_table",
        exists=True,
    )
    
    sql_calls = to_sql_calls(adapter)
    create_sql = sql_calls[0]
    
    # Should use LIKE clause
    assert "LIKE" in create_sql
    assert "source_table" in create_sql


def test_drop_view_staging(adapter: DB2EngineAdapter):
    """Test dropping staging views tries multiple formats."""
    # Mock that view doesn't exist in any format
    adapter.cursor.execute.side_effect = Exception("SQL0204N")
    
    # Should not raise error for staging view
    adapter.drop_view("test_schema.sqlmesh__test_view__abc", ignore_if_not_exists=True)
    
    # Should have tried multiple drop attempts
    assert adapter.cursor.execute.call_count > 1

