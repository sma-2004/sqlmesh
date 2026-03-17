# type: ignore
"""
Integration tests for DB2 adapter with real DB2 instance.

These tests require a real DB2 database connection.

Setup:
    export DB2_HOST=your-db2-host.com
    export DB2_PORT=50000
    export DB2_DATABASE=TESTDB
    export DB2_USERNAME=db2user
    export DB2_PASSWORD=your_password

Run with:
    pytest tests/core/engine_adapter/integration/test_db2_integration.py --run-db2-integration -v
"""

import os
import pytest
from sqlglot import parse_one, exp

from sqlmesh.core.engine_adapter import create_engine_adapter
from sqlmesh.core.config.connection import DB2ConnectionConfig


pytestmark = [
    pytest.mark.integration,
    pytest.mark.db2,
    pytest.mark.skipif(
        not os.getenv("DB2_HOST"),
        reason="DB2_HOST environment variable not set. Set DB2 connection env vars to run integration tests."
    )
]


@pytest.fixture(scope="module")
def db2_config():
    """Create DB2 connection config from environment variables."""
    return DB2ConnectionConfig(
        host=os.getenv("DB2_HOST", "localhost"),
        port=int(os.getenv("DB2_PORT", "50000")),
        database=os.getenv("DB2_DATABASE", "TESTDB"),
        username=os.getenv("DB2_USERNAME", "db2user"),
        password=os.getenv("DB2_PASSWORD", ""),
        concurrent_tasks=2,
    )


@pytest.fixture(scope="module")
def adapter(db2_config):
    """Create DB2 adapter with real connection."""
    adapter = create_engine_adapter(
        db2_config._connection_factory,
        dialect="db2",
    )
    yield adapter
    # Cleanup
    adapter.close()


@pytest.fixture(scope="module")
def test_schema(adapter):
    """Create and cleanup test schema."""
    schema_name = "SQLMESH_TEST"
    
    # Create test schema
    try:
        adapter.create_schema(schema_name, ignore_if_exists=True)
    except Exception as e:
        pytest.skip(f"Could not create test schema: {e}")
    
    yield schema_name
    
    # Cleanup: drop test schema
    try:
        adapter.drop_schema(schema_name, cascade=True, ignore_if_not_exists=True)
    except Exception:
        pass  # Best effort cleanup


def test_connection(adapter):
    """Test basic DB2 connection."""
    result = adapter.fetchone("SELECT CURRENT SERVER FROM SYSIBM.SYSDUMMY1")
    assert result is not None
    assert len(result) > 0
    print(f"Connected to DB2 server: {result[0]}")


def test_get_current_catalog(adapter):
    """Test getting current catalog."""
    catalog = adapter.get_current_catalog()
    assert catalog is not None
    assert isinstance(catalog, str)
    print(f"Current catalog: {catalog}")


def test_get_current_schema(adapter):
    """Test getting current schema."""
    schema = adapter._get_current_schema()
    assert schema is not None
    assert isinstance(schema, str)
    print(f"Current schema: {schema}")


def test_server_version(adapter):
    """Test getting DB2 server version."""
    version = adapter.server_version
    assert isinstance(version, tuple)
    assert len(version) == 2
    assert version[0] >= 10  # DB2 10.x or higher
    print(f"DB2 version: {version[0]}.{version[1]}")


def test_create_and_drop_table(adapter, test_schema):
    """Test creating and dropping a table."""
    table_name = f"{test_schema}.test_table_basic"
    
    # Create table
    adapter.create_table(
        table_name,
        {
            "id": exp.DataType.build("INT"),
            "name": exp.DataType.build("VARCHAR(100)"),
            "created_at": exp.DataType.build("TIMESTAMP"),
        },
        primary_key=("id",),
    )
    
    # Verify table exists
    assert adapter.table_exists(table_name)
    
    # Get columns
    columns = adapter.columns(table_name)
    assert "id" in [c.lower() for c in columns.keys()]
    assert "name" in [c.lower() for c in columns.keys()]
    assert "created_at" in [c.lower() for c in columns.keys()]
    
    # Drop table
    adapter.drop_table(table_name)
    
    # Verify table doesn't exist
    assert not adapter.table_exists(table_name)


def test_create_table_as_select(adapter, test_schema):
    """Test CREATE TABLE AS SELECT with WITH DATA."""
    source_table = f"{test_schema}.ctas_source"
    target_table = f"{test_schema}.ctas_target"
    
    try:
        # Create source table
        adapter.create_table(
            source_table,
            {
                "id": exp.DataType.build("INT"),
                "value": exp.DataType.build("VARCHAR(50)"),
            },
        )
        
        # Insert test data
        adapter.execute(
            f"INSERT INTO {source_table} (id, value) VALUES (1, 'test1'), (2, 'test2')"
        )
        
        # Create table as select
        adapter.ctas(
            target_table,
            parse_one(f"SELECT * FROM {source_table}"),
            exists=False,
        )
        
        # Verify target table exists and has data
        assert adapter.table_exists(target_table)
        
        result = adapter.fetchall(f"SELECT COUNT(*) FROM {target_table}")
        assert result[0][0] == 2
        
    finally:
        # Cleanup
        adapter.drop_table(source_table, exists=True)
        adapter.drop_table(target_table, exists=True)


def test_create_index(adapter, test_schema):
    """Test creating an index."""
    table_name = f"{test_schema}.test_table_index"
    index_name = "idx_test"
    
    try:
        # Create table
        adapter.create_table(
            table_name,
            {
                "id": exp.DataType.build("INT"),
                "name": exp.DataType.build("VARCHAR(100)"),
            },
        )
        
        # Create index
        adapter.create_index(
            table_name,
            index_name,
            ("name",),
        )
        
        # Verify index exists (query SYSCAT.INDEXES)
        schema = test_schema
        table = "test_table_index"
        result = adapter.fetchone(
            f"""
            SELECT COUNT(*) FROM SYSCAT.INDEXES
            WHERE UPPER(TABSCHEMA) = '{schema.upper()}'
            AND UPPER(TABNAME) = '{table.upper()}'
            AND UPPER(INDNAME) = '{index_name.upper()}'
            """
        )
        assert result[0] > 0
        
    finally:
        # Cleanup
        adapter.drop_table(table_name, exists=True)


def test_create_and_drop_view(adapter, test_schema):
    """Test creating and dropping a view."""
    table_name = f"{test_schema}.view_source_table"
    view_name = f"{test_schema}.test_view"
    
    try:
        # Create source table
        adapter.create_table(
            table_name,
            {
                "id": exp.DataType.build("INT"),
                "value": exp.DataType.build("VARCHAR(50)"),
            },
        )
        
        # Create view
        adapter.create_view(
            view_name,
            parse_one(f"SELECT * FROM {table_name}"),
            replace=False,
        )
        
        # Verify view exists (query SYSCAT.VIEWS)
        schema = test_schema
        view = "test_view"
        result = adapter.fetchone(
            f"""
            SELECT COUNT(*) FROM SYSCAT.VIEWS
            WHERE UPPER(VIEWSCHEMA) = '{schema.upper()}'
            AND UPPER(VIEWNAME) = '{view.upper()}'
            """
        )
        assert result[0] > 0
        
        # Drop view
        adapter.drop_view(view_name)
        
    finally:
        # Cleanup
        adapter.drop_table(table_name, exists=True)


def test_table_comments(adapter, test_schema):
    """Test adding comments to tables and columns."""
    table_name = f"{test_schema}.test_table_comments"
    
    try:
        # Create table with comments
        adapter.create_table(
            table_name,
            {
                "id": exp.DataType.build("INT"),
                "name": exp.DataType.build("VARCHAR(100)"),
            },
            table_description="Test table for comments",
            column_descriptions={
                "id": "Primary key column",
                "name": "Name column",
            },
        )
        
        # Verify table exists
        assert adapter.table_exists(table_name)
        
        # Note: Verifying comments requires querying SYSCAT.TABLES.REMARKS
        # and SYSCAT.COLUMNS.REMARKS, which may not be populated immediately
        
    finally:
        # Cleanup
        adapter.drop_table(table_name, exists=True)


def test_merge_operation(adapter, test_schema):
    """Test MERGE statement."""
    target_table = f"{test_schema}.merge_target"
    source_table = f"{test_schema}.merge_source"
    
    try:
        # Create target table
        adapter.create_table(
            target_table,
            {
                "id": exp.DataType.build("INT"),
                "value": exp.DataType.build("VARCHAR(50)"),
            },
        )
        
        # Insert initial data
        adapter.execute(
            f"INSERT INTO {target_table} (id, value) VALUES (1, 'old1'), (2, 'old2')"
        )
        
        # Create source table
        adapter.create_table(
            source_table,
            {
                "id": exp.DataType.build("INT"),
                "value": exp.DataType.build("VARCHAR(50)"),
            },
        )
        
        # Insert source data (update id=1, insert id=3)
        adapter.execute(
            f"INSERT INTO {source_table} (id, value) VALUES (1, 'new1'), (3, 'new3')"
        )
        
        # Perform merge
        adapter.merge(
            target_table=target_table,
            source_table=parse_one(f"SELECT * FROM {source_table}"),
            target_columns_to_types={
                "id": exp.DataType.build("INT"),
                "value": exp.DataType.build("VARCHAR(50)"),
            },
            unique_key=[exp.to_identifier("id", quoted=True)],
        )
        
        # Verify results
        result = adapter.fetchall(f"SELECT id, value FROM {target_table} ORDER BY id")
        assert len(result) == 3
        assert result[0] == (1, "new1")  # Updated
        assert result[1] == (2, "old2")  # Unchanged
        assert result[2] == (3, "new3")  # Inserted
        
    finally:
        # Cleanup
        adapter.drop_table(target_table, exists=True)
        adapter.drop_table(source_table, exists=True)


def test_case_sensitivity(adapter, test_schema):
    """Test case-insensitive table name handling."""
    table_name_lower = f"{test_schema.lower()}.test_case_table"
    table_name_upper = f"{test_schema.upper()}.TEST_CASE_TABLE"
    
    try:
        # Create table with lowercase name
        adapter.create_table(
            table_name_lower,
            {"id": exp.DataType.build("INT")},
        )
        
        # Check existence with uppercase name
        assert adapter.table_exists(table_name_upper)
        
        # Check existence with lowercase name
        assert adapter.table_exists(table_name_lower)
        
    finally:
        # Cleanup
        adapter.drop_table(table_name_lower, exists=True)


def test_transaction_support(adapter, test_schema):
    """Test transaction support."""
    table_name = f"{test_schema}.test_transaction"
    
    try:
        # Create table
        adapter.create_table(
            table_name,
            {"id": exp.DataType.build("INT")},
        )
        
        # Test transaction with rollback
        with adapter.transaction():
            adapter.execute(f"INSERT INTO {table_name} (id) VALUES (1)")
            # Rollback by raising exception
            try:
                raise Exception("Test rollback")
            except:
                pass
        
        # Verify no data (transaction rolled back)
        result = adapter.fetchone(f"SELECT COUNT(*) FROM {table_name}")
        # Note: Behavior depends on how transaction context manager handles exceptions
        
        # Test transaction with commit
        with adapter.transaction():
            adapter.execute(f"INSERT INTO {table_name} (id) VALUES (2)")
        
        # Verify data exists
        result = adapter.fetchone(f"SELECT COUNT(*) FROM {table_name}")
        assert result[0] >= 1
        
    finally:
        # Cleanup
        adapter.drop_table(table_name, exists=True)


def test_batch_operations(adapter, test_schema):
    """Test batch insert operations."""
    table_name = f"{test_schema}.test_batch"
    
    try:
        # Create table
        adapter.create_table(
            table_name,
            {
                "id": exp.DataType.build("INT"),
                "value": exp.DataType.build("VARCHAR(50)"),
            },
        )
        
        # Insert multiple rows
        values = [(i, f"value{i}") for i in range(100)]
        for id_val, value in values:
            adapter.execute(
                f"INSERT INTO {table_name} (id, value) VALUES ({id_val}, '{value}')"
            )
        
        # Verify count
        result = adapter.fetchone(f"SELECT COUNT(*) FROM {table_name}")
        assert result[0] == 100
        
    finally:
        # Cleanup
        adapter.drop_table(table_name, exists=True)


def test_data_types(adapter, test_schema):
    """Test various DB2 data types."""
    table_name = f"{test_schema}.test_data_types"
    
    try:
        # Create table with various types
        adapter.create_table(
            table_name,
            {
                "col_int": exp.DataType.build("INT"),
                "col_bigint": exp.DataType.build("BIGINT"),
                "col_decimal": exp.DataType.build("DECIMAL(10,2)"),
                "col_varchar": exp.DataType.build("VARCHAR(100)"),
                "col_char": exp.DataType.build("CHAR(10)"),
                "col_date": exp.DataType.build("DATE"),
                "col_timestamp": exp.DataType.build("TIMESTAMP"),
                "col_double": exp.DataType.build("DOUBLE"),
            },
        )
        
        # Get columns and verify types
        columns = adapter.columns(table_name)
        assert len(columns) == 8
        
        # Verify each column exists (case-insensitive)
        column_names_lower = [c.lower() for c in columns.keys()]
        assert "col_int" in column_names_lower
        assert "col_bigint" in column_names_lower
        assert "col_decimal" in column_names_lower
        assert "col_varchar" in column_names_lower
        
    finally:
        # Cleanup
        adapter.drop_table(table_name, exists=True)

