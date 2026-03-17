# type: ignore
"""
Integration tests for DB2 adapter with custom sqlglot DB2 dialect.

These tests verify that the custom sqlglot DB2 dialect (provided as .whl)
works correctly with the DB2 engine adapter.

Prerequisites:
    - Custom sqlglot with DB2 dialect installed
    - ibm_db and ibm_db_dbi packages installed
"""

import pytest
from sqlglot import parse_one, exp
from sqlglot.dialects import Dialects

from sqlmesh.core.engine_adapter.db2_proper import DB2EngineAdapter


pytestmark = [pytest.mark.engine, pytest.mark.db2, pytest.mark.integration]


def test_db2_dialect_available():
    """Verify that DB2 dialect is available in sqlglot."""
    # Check if db2 is in available dialects
    assert "db2" in [d.value for d in Dialects]
    

def test_db2_parse_select():
    """Test parsing basic SELECT with DB2 dialect."""
    sql = "SELECT * FROM SYSIBM.SYSDUMMY1"
    parsed = parse_one(sql, dialect="db2")
    
    assert isinstance(parsed, exp.Select)
    assert parsed.sql(dialect="db2") == sql


def test_db2_parse_current_server():
    """Test parsing CURRENT SERVER special register."""
    sql = "SELECT CURRENT SERVER FROM SYSIBM.SYSDUMMY1"
    parsed = parse_one(sql, dialect="db2")
    
    assert isinstance(parsed, exp.Select)
    generated = parsed.sql(dialect="db2")
    assert "CURRENT SERVER" in generated
    assert "SYSIBM.SYSDUMMY1" in generated


def test_db2_parse_create_table():
    """Test parsing CREATE TABLE with DB2 syntax."""
    sql = """
    CREATE TABLE test_table (
        id INTEGER NOT NULL,
        name VARCHAR(100),
        amount DECIMAL(10, 2),
        created_at TIMESTAMP,
        PRIMARY KEY (id)
    )
    """
    parsed = parse_one(sql, dialect="db2")
    
    assert isinstance(parsed, exp.Create)
    generated = parsed.sql(dialect="db2")
    assert "CREATE TABLE" in generated
    assert "INTEGER NOT NULL" in generated
    assert "VARCHAR(100)" in generated
    assert "DECIMAL(10, 2)" in generated


def test_db2_parse_create_table_as_select():
    """Test parsing CREATE TABLE AS SELECT with WITH DATA."""
    sql = "CREATE TABLE new_table AS (SELECT * FROM old_table) WITH DATA"
    parsed = parse_one(sql, dialect="db2")
    
    assert isinstance(parsed, exp.Create)
    generated = parsed.sql(dialect="db2")
    assert "CREATE TABLE" in generated
    assert "WITH DATA" in generated


def test_db2_parse_merge():
    """Test parsing MERGE statement."""
    sql = """
    MERGE INTO target AS t
    USING source AS s
    ON t.id = s.id
    WHEN MATCHED THEN UPDATE SET t.value = s.value
    WHEN NOT MATCHED THEN INSERT (id, value) VALUES (s.id, s.value)
    """
    parsed = parse_one(sql, dialect="db2")
    
    assert isinstance(parsed, exp.Merge)
    generated = parsed.sql(dialect="db2")
    assert "MERGE INTO" in generated


def test_db2_parse_comment():
    """Test parsing COMMENT ON statement."""
    sql = "COMMENT ON TABLE test_table IS 'Test table description'"
    parsed = parse_one(sql, dialect="db2")
    
    assert isinstance(parsed, exp.Comment)
    generated = parsed.sql(dialect="db2")
    assert "COMMENT ON TABLE" in generated


def test_db2_parse_create_index():
    """Test parsing CREATE INDEX."""
    sql = "CREATE INDEX idx_test ON test_table (col1, col2)"
    parsed = parse_one(sql, dialect="db2")
    
    assert isinstance(parsed, exp.Create)
    generated = parsed.sql(dialect="db2")
    assert "CREATE INDEX" in generated
    assert "idx_test" in generated


def test_db2_parse_drop_schema_restrict():
    """Test parsing DROP SCHEMA with RESTRICT."""
    sql = "DROP SCHEMA test_schema RESTRICT"
    parsed = parse_one(sql, dialect="db2")
    
    assert isinstance(parsed, exp.Drop)
    generated = parsed.sql(dialect="db2")
    assert "DROP SCHEMA" in generated
    assert "RESTRICT" in generated


def test_db2_data_types():
    """Test DB2-specific data types parsing."""
    test_types = [
        "INTEGER",
        "BIGINT",
        "SMALLINT",
        "DECIMAL(10, 2)",
        "DOUBLE",
        "REAL",
        "VARCHAR(100)",
        "CHAR(10)",
        "CLOB",
        "BLOB",
        "DATE",
        "TIME",
        "TIMESTAMP",
        "GRAPHIC(50)",
        "VARGRAPHIC(100)",
        "DBCLOB",
        "XML",
    ]
    
    for type_str in test_types:
        sql = f"CREATE TABLE test (col {type_str})"
        parsed = parse_one(sql, dialect="db2")
        assert isinstance(parsed, exp.Create)
        # Verify it can be generated back
        generated = parsed.sql(dialect="db2")
        assert "CREATE TABLE" in generated


def test_db2_adapter_uses_correct_dialect(make_mocked_engine_adapter):
    """Test that DB2 adapter uses db2 dialect."""
    adapter = make_mocked_engine_adapter(DB2EngineAdapter)
    
    assert adapter.DIALECT == "db2"
    assert adapter.dialect == "db2"


def test_db2_adapter_sql_generation(make_mocked_engine_adapter):
    """Test SQL generation through adapter."""
    adapter = make_mocked_engine_adapter(DB2EngineAdapter)
    
    # Test table creation
    table_exp = exp.to_table("test_schema.test_table")
    sql = adapter._to_sql(table_exp)
    
    # Should be properly quoted for DB2
    assert "test_schema" in sql
    assert "test_table" in sql


def test_db2_system_catalog_queries(make_mocked_engine_adapter):
    """Test that system catalog queries use correct DB2 syntax."""
    adapter = make_mocked_engine_adapter(DB2EngineAdapter)
    
    # Mock columns query
    adapter.cursor.fetchall.return_value = [
        ("id", "INTEGER", 4, 0),
        ("name", "VARCHAR", 100, 0),
    ]
    
    columns = adapter.columns("test_table")
    
    # Verify the query was executed
    assert adapter.cursor.execute.called
    
    # Get the executed SQL
    executed_sql = str(adapter.cursor.execute.call_args[0][0])
    
    # Should query SYSCAT.COLUMNS
    assert "SYSCAT.COLUMNS" in executed_sql


def test_db2_case_insensitive_matching(make_mocked_engine_adapter):
    """Test case-insensitive table existence check."""
    adapter = make_mocked_engine_adapter(DB2EngineAdapter)
    
    # Mock table exists with actual case
    adapter.cursor.fetchone.return_value = ("TEST_SCHEMA", "TEST_TABLE")
    
    # Check with lowercase
    exists = adapter.table_exists("test_schema.test_table")
    
    assert exists is True
    
    # Verify UPPER() function was used
    executed_sql = str(adapter.cursor.execute.call_args[0][0])
    assert "UPPER" in executed_sql


def test_db2_identifier_quoting(make_mocked_engine_adapter):
    """Test identifier quoting for special characters."""
    adapter = make_mocked_engine_adapter(DB2EngineAdapter)
    
    # Table with underscores (should be quoted)
    table_exp = exp.to_table("sqlmesh__test_table")
    sql = adapter._to_sql(table_exp, quote=True)
    
    # Should be quoted
    assert '"' in sql or '[' in sql


@pytest.mark.skipif(
    not pytest.config.getoption("--run-db2-integration", default=False),
    reason="Requires --run-db2-integration flag and real DB2 instance"
)
def test_real_db2_connection():
    """
    Test real DB2 connection (requires actual DB2 instance).
    
    Run with: pytest --run-db2-integration
    
    Requires environment variables:
        - DB2_HOST
        - DB2_PORT
        - DB2_DATABASE
        - DB2_USERNAME
        - DB2_PASSWORD
    """
    import os
    import ibm_db_dbi
    
    conn_str = (
        f"DATABASE={os.getenv('DB2_DATABASE')};"
        f"HOSTNAME={os.getenv('DB2_HOST')};"
        f"PORT={os.getenv('DB2_PORT', '50000')};"
        f"PROTOCOL=TCPIP;"
        f"UID={os.getenv('DB2_USERNAME')};"
        f"PWD={os.getenv('DB2_PASSWORD')};"
    )
    
    conn = ibm_db_dbi.connect(conn_str, "", "")
    cursor = conn.cursor()
    
    # Test basic query
    cursor.execute("SELECT CURRENT SERVER FROM SYSIBM.SYSDUMMY1")
    result = cursor.fetchone()
    
    assert result is not None
    assert len(result) > 0
    
    cursor.close()
    conn.close()


def test_db2_dialect_version_compatibility():
    """Test that sqlglot DB2 dialect version is compatible."""
    import sqlglot
    
    # Verify sqlglot version
    version = sqlglot.__version__
    print(f"sqlglot version: {version}")
    
    # Verify DB2 dialect is available
    assert hasattr(Dialects, "DB2") or "db2" in [d.value for d in Dialects]

