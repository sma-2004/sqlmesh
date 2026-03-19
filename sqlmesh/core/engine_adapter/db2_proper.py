"""
IBM DB2 Engine Adapter for SQLMesh

This module provides DB2 database support for SQLMesh by implementing
the EngineAdapter interface following SQLMesh's architecture patterns.

Supports:
- DB2 for Linux, UNIX, and Windows (LUW)
- DB2 for z/OS
- DB2 for i (AS/400)

Author: SQLMesh DB2 Integration Team
License: Apache 2.0
"""

from __future__ import annotations

import logging
import typing as t
from functools import cached_property

from sqlglot import exp

from sqlmesh.core.engine_adapter.base import EngineAdapter, _get_data_object_cache_key
from sqlmesh.core.engine_adapter.mixins import (
    GetCurrentCatalogFromFunctionMixin,
    PandasNativeFetchDFSupportMixin,
)
from sqlmesh.core.engine_adapter.shared import (
    CatalogSupport,
    CommentCreationTable,
    CommentCreationView,
    DataObject,
    DataObjectType,
    set_catalog,
)
from sqlmesh.core.dialect import to_schema
from sqlmesh.utils.errors import SQLMeshError

if t.TYPE_CHECKING:
    from sqlmesh.core._typing import SchemaName, TableName
    from sqlmesh.core.engine_adapter._typing import Query, QueryOrDF

logger = logging.getLogger(__name__)


# Db2 SQL Error Codes
class DB2ErrorCodes:
    """Common Db2 SQL error codes for better error handling."""
    OBJECT_NOT_FOUND = "SQL0204N"  # Table/view does not exist
    COLUMN_NOT_FOUND = "SQL0205N"  # Column does not exist
    DUPLICATE_OBJECT = "SQL0601N"  # Object already exists
    INDEX_EXISTS = "SQL0605W"  # Index already defined
    SCHEMA_NOT_EMPTY = "SQL0478N"  # Schema contains objects
    INVALID_IDENTIFIER = "SQL0104N"  # Invalid token
    AUTHORIZATION_ERROR = "SQL0551N"  # Authorization failure
    CREATE_SCHEMA_PRIVILEGE = "SQL0552N"  # No privilege to create schema
    CONNECTION_ERROR = "SQL30081N"  # Connection failed
    DEADLOCK = "SQL0911N"  # Deadlock or timeout
    LOCK_TIMEOUT = "SQL0913N"  # Lock timeout


def is_db2_error(exception: Exception, error_code: str) -> bool:
    """
    Check if an exception is a specific DB2 error.
    
    Args:
        exception: The exception to check
        error_code: The DB2 error code to match (e.g., "SQL0204N")
        
    Returns:
        True if the exception matches the error code
    """
    error_msg = str(exception)
    return error_code in error_msg


@set_catalog()
class DB2EngineAdapter(
    PandasNativeFetchDFSupportMixin,
    GetCurrentCatalogFromFunctionMixin,
    EngineAdapter,
):
    """
    Engine adapter for IBM DB2 databases.
    
    This adapter enables SQLMesh to work with DB2 by implementing
    the EngineAdapter interface using sqlglot for SQL generation
    and DB2-specific system catalog queries.
    
    Uses native DB2 dialect from sqlglot for proper SQL generation.
    """
    
    # Adapter Configuration
    DIALECT = "db2"  # Use native DB2 dialect for SQL generation
    DEFAULT_BATCH_SIZE = 400
    SUPPORTS_TRANSACTIONS = True
    SUPPORTS_INDEXES = True
    SUPPORTS_REPLACE_TABLE = False  # DB2 doesn't have REPLACE
    SUPPORTS_GRANTS = True  # DB2 supports GRANT/REVOKE
    CATALOG_SUPPORT = CatalogSupport.SINGLE_CATALOG_ONLY
    COMMENT_CREATION_TABLE = CommentCreationTable.COMMENT_COMMAND_ONLY
    COMMENT_CREATION_VIEW = CommentCreationView.COMMENT_COMMAND_ONLY
    SUPPORTS_QUERY_EXECUTION_TRACKING = True
    SUPPORTED_DROP_CASCADE_OBJECT_KINDS = ["SCHEMA", "TABLE", "VIEW"]
    HAS_VIEW_BINDING = False
    MAX_IDENTIFIER_LENGTH: t.Optional[int] = 128
    # DB2 requires FROM clause for CURRENT SERVER
    CURRENT_CATALOG_EXPRESSION = exp.column("CURRENT SERVER")
    SCHEMA_DIFFER_KWARGS = {
        "parameterized_type_defaults": {
            # DECIMAL without precision defaults to (5, 0)
            exp.DataType.build("DECIMAL", dialect=DIALECT).this: [(5, 0), (0,)],
            # CHAR without length defaults to 1
            exp.DataType.build("CHAR", dialect=DIALECT).this: [(1,)],
            # VARCHAR without length defaults to 1
            exp.DataType.build("VARCHAR", dialect=DIALECT).this: [(1,)],
            # TIMESTAMP defaults to 6 digits of fractional seconds
            exp.DataType.build("TIMESTAMP", dialect=DIALECT).this: [(6,)],
            # TIME defaults to 0 digits of fractional seconds
            exp.DataType.build("TIME", dialect=DIALECT).this: [(0,)],
        },
        "types_with_unlimited_length": {
            # CLOB can be used for unlimited text
            exp.DataType.build("CLOB", dialect=DIALECT).this: {
                exp.DataType.build("VARCHAR", dialect=DIALECT).this,
                exp.DataType.build("CHAR", dialect=DIALECT).this,
            },
        },
        "drop_cascade": False,  # DB2 requires explicit CASCADE keyword
    }
    
    # DB2 System Schemas (to exclude from operations)
    SYSTEM_SCHEMAS = {
        'SYSCAT', 'SYSFUN', 'SYSIBM', 'SYSIBMADM',
        'SYSPROC', 'SYSPUBLIC', 'SYSSTAT', 'SYSTOOLS'
    }
    
    def get_current_catalog(self) -> t.Optional[str]:
        """
        Returns the catalog name of the current connection.
        DB2 requires FROM SYSIBM.SYSDUMMY1 for special registers.
        Returns uppercase to match sqlglot's Db2 dialect behavior.
        """
        result = self.fetchone("SELECT CURRENT SERVER FROM SYSIBM.SYSDUMMY1")
        if result:
            return result[0].upper() if result[0] else None
        return None
    
    def _build_schema_exp(
        self,
        table: exp.Table,
        target_columns_to_types: t.Dict[str, exp.DataType],
        column_descriptions: t.Optional[t.Dict[str, str]] = None,
        expressions: t.Optional[t.List[exp.PrimaryKey]] = None,
        is_view: bool = False,
        materialized: bool = False,
    ) -> exp.Schema:
        """
        Build a schema expression for DB2.
        
        DB2 requires primary key columns to have NOT NULL constraint.
        """
        expressions = expressions or []
        
        # Extract primary key column names
        pk_columns = set()
        for expr in expressions:
            if isinstance(expr, exp.PrimaryKey):
                for col_expr in expr.expressions:
                    if isinstance(col_expr, exp.Column):
                        pk_columns.add(col_expr.name)
        
        # Build column definitions with NOT NULL for primary keys
        column_defs = []
        for column, col_type in target_columns_to_types.items():
            col_def = self._build_column_def(
                column,
                column_descriptions=column_descriptions,
                engine_supports_schema_comments=self.COMMENT_CREATION_TABLE.supports_schema_def if not is_view else self.COMMENT_CREATION_VIEW.supports_schema_def,
                col_type=None if is_view else col_type,
            )
            
            # Add NOT NULL constraint for primary key columns in DB2
            if column in pk_columns and not is_view:
                # Get existing constraints
                existing_constraints = col_def.args.get("constraints") or []
                # Check if NOT NULL already exists
                has_not_null = any(
                    isinstance(c, exp.NotNullColumnConstraint)
                    for c in existing_constraints
                )
                if not has_not_null:
                    # Add NOT NULL constraint
                    existing_constraints.append(exp.NotNullColumnConstraint())
                    col_def.set("constraints", existing_constraints)
            
            column_defs.append(col_def)
        
        return exp.Schema(
            this=table,
            expressions=column_defs + expressions,
        )
    
    def _to_sql(self, expression: exp.Expression, quote: bool = True, **kwargs: t.Any) -> str:
        """
        Converts an expression to SQL for DB2.
        
        DB2 uppercases unquoted identifiers, but we need to quote identifiers that:
        - Contain special characters (like double underscores)
        - Are mixed case
        - Are reserved words
        
        This ensures SQLMesh state tables (with __ in names) work correctly.
        """
        # Always quote when requested - this is needed for SQLMesh state tables
        # which have names like "sqlmesh__versions" that DB2 treats specially
        return super()._to_sql(expression, quote=quote, **kwargs)
    
    def create_index(
        self,
        table_name: TableName,
        index_name: str,
        columns: t.Tuple[str, ...],
        exists: bool = True,
    ) -> None:
        """
        Creates a new index for the given table.
        
        DB2 doesn't support IF NOT EXISTS for CREATE INDEX, so we need to
        check if the index exists first and only create it if it doesn't.
        
        Args:
            table_name: The name of the target table.
            index_name: The name of the index.
            columns: The list of columns that constitute the index.
            exists: Indicates whether to check if index exists (ignored for DB2, always checks).
        """
        if not self.SUPPORTS_INDEXES:
            return
        
        # Check if index already exists in DB2 system catalog
        table = exp.to_table(table_name)
        schema_name = table.db or self._get_current_schema()
        
        check_sql = f"""
            SELECT COUNT(*)
            FROM SYSCAT.INDEXES
            WHERE TABSCHEMA = '{schema_name.upper()}'
            AND TABNAME = '{table.alias_or_name.upper()}'
            AND INDNAME = '{index_name.upper()}'
        """
        
        try:
            result = self.fetchone(check_sql)
            if result and result[0] > 0:
                # Index already exists, skip creation
                logger.debug(f"Index {index_name} already exists on {table_name}, skipping creation")
                return
        except Exception as e:
            # If we can't check, try to create anyway and handle the warning
            logger.debug(f"Could not check if index exists: {e}")
        
        # Create index without IF NOT EXISTS clause (DB2 doesn't support it)
        expression = exp.Create(
            this=exp.Index(
                this=exp.to_identifier(index_name),
                table=exp.to_table(table_name),
                params=exp.IndexParameters(columns=[exp.to_column(c) for c in columns]),
            ),
            kind="INDEX",
            exists=False,  # DB2 doesn't support IF NOT EXISTS
        )
        
        try:
            self.execute(expression)
        except Exception as e:
            # DB2 returns SQL0605W warning when index already exists
            # Check if it's the "index already exists" warning
            error_msg = str(e)
            if 'SQL0605W' in error_msg or 'index' in error_msg.lower() and 'already exists' in error_msg.lower():
                logger.debug(f"Index {index_name} already exists (caught warning), continuing")
                return
            # Re-raise if it's a different error
            raise
    
    def columns(
        self, table_name: TableName, include_pseudo_columns: bool = False
    ) -> t.Dict[str, exp.DataType]:
        """
        Fetches column names and types for the target table from DB2 system catalog.
        
        Args:
            table_name: The table to get columns for
            include_pseudo_columns: Not used for DB2
            
        Returns:
            Dictionary mapping column names to their data types
        """
        table = exp.to_table(table_name)
        schema_name = table.db or self._get_current_schema()
        
        # Query DB2's SYSCAT.COLUMNS system catalog
        # Note: DB2 stores identifiers in the case they were created
        # Don't force uppercase - use the actual case from the table name
        sql = exp.select(
            exp.column("COLNAME").as_("column_name"),
            exp.column("TYPENAME").as_("data_type"),
            exp.column("LENGTH").as_("length"),
            exp.column("SCALE").as_("scale"),
        ).from_("SYSCAT.COLUMNS").where(
            exp.and_(
                exp.column("TABSCHEMA").eq(exp.Literal.string(schema_name)),
                exp.column("TABNAME").eq(exp.Literal.string(table.alias_or_name)),
            )
        ).order_by("COLNO")
        
        self.execute(sql)
        resp = self.cursor.fetchall()
        
        if not resp:
            raise SQLMeshError(
                f"Could not get columns for table '{table.sql(dialect=self.dialect)}'. Table not found."
            )
        
        columns = {}
        for column_name, data_type, length, scale in resp:
            # Convert DB2 types to sqlglot DataType
            # Keep column names in their original case from DB2
            db2_type = self._db2_type_to_sqlglot(data_type, length, scale)
            columns[column_name] = db2_type
        
        return columns
    
    def _db2_type_to_sqlglot(
        self, db2_type: str, length: int, scale: int
    ) -> exp.DataType:
        """
        Convert DB2 type to sqlglot DataType.
        
        Args:
            db2_type: DB2 type name
            length: Type length
            scale: Type scale
            
        Returns:
            sqlglot DataType expression
        """
        db2_type = db2_type.upper()
        
        # Comprehensive DB2 type mapping
        type_mapping = {
            # Numeric types
            "INTEGER": "INT",
            "INT": "INT",
            "BIGINT": "BIGINT",
            "SMALLINT": "SMALLINT",
            "DOUBLE": "DOUBLE",
            "REAL": "REAL",
            "FLOAT": "DOUBLE",
            "DECIMAL": f"DECIMAL({length},{scale})",
            "NUMERIC": f"DECIMAL({length},{scale})",
            "DECFLOAT": "DOUBLE",  # Map to DOUBLE for compatibility
            
            # Character types
            "VARCHAR": f"VARCHAR({length})",
            "CHAR": f"CHAR({length})",
            "CHARACTER": f"CHAR({length})",
            "CLOB": "CLOB",
            "GRAPHIC": f"CHAR({length})",  # Fixed-length graphic string
            "VARGRAPHIC": f"VARCHAR({length})",  # Variable-length graphic string
            "DBCLOB": "CLOB",  # Double-byte CLOB
            
            # Date/Time types
            "DATE": "DATE",
            "TIMESTAMP": "TIMESTAMP",
            "TIME": "TIME",
            
            # Binary types
            "BLOB": "BLOB",
            "BINARY": f"BINARY({length})",
            "VARBINARY": f"VARBINARY({length})",
            
            # Special types
            "XML": "TEXT",  # Map XML to TEXT for compatibility
            "ROWID": "VARCHAR(40)",  # ROWID is typically 40 bytes
            "BOOLEAN": "BOOLEAN",
        }
        
        sqlglot_type = type_mapping.get(db2_type, f"VARCHAR({length})")
        return exp.DataType.build(sqlglot_type, dialect="db2")
    
    @property
    def catalog_support(self) -> CatalogSupport:
        """DB2 supports single catalog only."""
        return CatalogSupport.SINGLE_CATALOG_ONLY
    
    def table_exists(self, table_name: TableName) -> bool:
        """
        Check if table exists in DB2 using SYSCAT.TABLES.
        
        DB2 stores identifiers in the case they were created (quoted or unquoted).
        We query the system catalog with case-insensitive matching to find the actual case.
        
        Args:
            table_name: The table to check
            
        Returns:
            True if table exists, False otherwise
        """
        table = exp.to_table(table_name)
        data_object_cache_key = _get_data_object_cache_key(table.catalog, table.db, table.name)
        
        if data_object_cache_key in self._data_object_cache:
            logger.debug("Table existence cache hit: %s", data_object_cache_key)
            return self._data_object_cache[data_object_cache_key] is not None
        
        schema_name = table.db or self._get_current_schema()
        table_name_str = table.alias_or_name
        
        # Query DB2's SYSCAT.TABLES with case-insensitive matching
        # Use UPPER() function for case-insensitive comparison
        sql = exp.select(
            exp.column("TABSCHEMA"),
            exp.column("TABNAME")
        ).from_("SYSCAT.TABLES").where(
            exp.and_(
                exp.func("UPPER", exp.column("TABSCHEMA")).eq(exp.Literal.string(schema_name.upper())),
                exp.func("UPPER", exp.column("TABNAME")).eq(exp.Literal.string(table_name_str.upper())),
            )
        )
        
        try:
            self.execute(sql)
            result = self.cursor.fetchone()
            
            exists = result is not None
            
            # Update cache with actual case from DB2
            if exists:
                actual_schema, actual_table = result
                self._data_object_cache[data_object_cache_key] = DataObject(
                    name=actual_table,
                    schema=actual_schema,
                    type=DataObjectType.TABLE,
                )
            
            return exists
        except Exception as e:
            # Handle DB2 deadlock/timeout errors (SQL0911N) and other exceptions
            # If we get an error checking if table exists, assume it doesn't exist
            # This allows the operation to proceed and potentially succeed
            logger.warning(
                f"Error checking if table {table} exists: {e}. Assuming table does not exist."
            )
            return False
    def _build_create_table_exp(
        self,
        table_name_or_schema: t.Union[exp.Schema, TableName],
        expression: t.Optional[exp.Expression],
        exists: bool = True,
        replace: bool = False,
        target_columns_to_types: t.Optional[t.Dict[str, exp.DataType]] = None,
        table_description: t.Optional[str] = None,
        table_kind: t.Optional[str] = None,
        **kwargs: t.Any,
    ) -> exp.Create:
        """
        Override to handle DB2's lack of support for CREATE TABLE IF NOT EXISTS.
        
        DB2 doesn't support IF NOT EXISTS clause in CREATE TABLE statements.
        We handle this by:
        1. Checking if table exists when exists=True
        2. Creating without IF NOT EXISTS clause
        
        Args:
            table_name_or_schema: Table name or schema expression
            expression: Optional query expression for CTAS
            exists: If True, check table existence before creating
            replace: If True, replace existing table
            target_columns_to_types: Column types mapping
            table_description: Optional table description
            table_kind: Kind of object (TABLE, VIEW, etc.)
            **kwargs: Additional create table properties
            
        Returns:
            CREATE TABLE expression without IF NOT EXISTS
        """
        # DB2 doesn't support IF NOT EXISTS, so we always set exists=False
        # The table existence check is handled in _create_table method below
        return super()._build_create_table_exp(
            table_name_or_schema=table_name_or_schema,
            expression=expression,
            exists=False,  # Always False for DB2
            replace=replace,
            target_columns_to_types=target_columns_to_types,
            table_description=table_description,
            table_kind=table_kind,
            **kwargs,
        )
    
    def _create_table(
        self,
        table_name_or_schema: t.Union[exp.Schema, TableName],
        expression: t.Optional[exp.Expression],
        exists: bool = True,
        replace: bool = False,
        target_columns_to_types: t.Optional[t.Dict[str, exp.DataType]] = None,
        table_description: t.Optional[str] = None,
        column_descriptions: t.Optional[t.Dict[str, str]] = None,
        table_kind: t.Optional[str] = None,
        track_rows_processed: bool = True,
        **kwargs: t.Any,
    ) -> None:
        """
        Override to handle DB2's lack of support for CREATE TABLE IF NOT EXISTS
        and to add WITH DATA clause for CREATE TABLE AS SELECT.
        
        Since DB2 doesn't support IF NOT EXISTS, we check table existence first
        and skip creation if the table already exists and exists=True.
        
        DB2 also requires WITH DATA clause for CREATE TABLE AS SELECT statements
        and doesn't allow subquery aliases in CTAS.
        
        Args:
            table_name_or_schema: Table name or schema expression
            expression: Optional query expression for CTAS
            exists: If True, check table existence before creating
            replace: If True, replace existing table
            target_columns_to_types: Column types mapping
            table_description: Optional table description
            column_descriptions: Optional column descriptions
            table_kind: Kind of object (TABLE, VIEW, etc.)
            track_rows_processed: Whether to track rows processed
            **kwargs: Additional create table properties
        """
        # Extract table name for existence check
        if isinstance(table_name_or_schema, exp.Schema):
            table_name = table_name_or_schema.this
        else:
            table_name = table_name_or_schema
        
        # CRITICAL FIX: Drop staging tables/views before creation
        # SQLMesh staging objects have names like "sqlmesh__SCHEMA.TABLE__HASH"
        # These must always be dropped before recreation to avoid SQL0601N errors
        table = exp.to_table(table_name)
        table_name_str = str(table.name) if hasattr(table, 'name') else str(table_name)
        schema_name = table.db or self._get_current_schema()
        
        if "sqlmesh__" in table_name_str.lower() or "sqlmesh__" in schema_name.lower():
            # This is a staging object - try to drop both TABLE and VIEW
            # Try multiple variations due to potential naming issues
            drop_attempts = []
            
            # Try both TABLE and VIEW for each format
            for object_type in ['TABLE', 'VIEW']:
                drop_attempts.extend([
                    # Standard format: schema.table
                    f'DROP {object_type} {self._to_sql(table)}',
                    # Quoted format
                    f'DROP {object_type} "{schema_name}"."{table_name_str}"',
                    # Uppercase format (DB2 default)
                    f'DROP {object_type} {schema_name.upper()}.{table_name_str.upper()}',
                    # Lowercase format (for quoted identifiers with underscores)
                    f'DROP {object_type} "{schema_name.lower()}"."{table_name_str.lower()}"',
                ])
            
            dropped = False
            for drop_sql in drop_attempts:
                try:
                    logger.info(f"Staging object drop attempt: {drop_sql}")
                    self.execute(drop_sql)
                    logger.info(f"Successfully dropped staging object with: {drop_sql}")
                    dropped = True
                    # Clear cache after dropping
                    data_object_cache_key = _get_data_object_cache_key(table.catalog, table.db, table.name)
                    if data_object_cache_key in self._data_object_cache:
                        del self._data_object_cache[data_object_cache_key]
                    break
                except Exception as e:
                    error_msg = str(e)
                    if 'SQL0204N' in error_msg or 'does not exist' in error_msg.lower() or 'SQL0205N' in error_msg:
                        logger.debug(f"Object doesn't exist with format: {drop_sql}")
                        continue
                    else:
                        logger.debug(f"Drop failed with: {drop_sql}, error: {e}")
                        continue
            
            if not dropped:
                logger.warning(f"Could not drop staging object {table_name} with any format - will try to create anyway")
        
        # For CREATE TABLE AS SELECT, we need to intercept and modify the SQL
        # to remove subquery aliases and ensure WITH DATA is present
        if expression and isinstance(expression, (exp.Select, exp.Subquery)):
            import re
            
            table = exp.to_table(table_name)
            
            # Check if table exists
            table_exists_flag = self.table_exists(table)
            
            # Handle table existence based on exists and replace parameters
            if table_exists_flag:
                if exists and not replace:
                    # Table exists and exists=True, skip creation
                    logger.info(f"CTAS: Table {table_name} already exists, skipping creation (exists=True)")
                    return
                elif replace or not exists:
                    # Drop the table if replace=True or exists=False
                    # DB2 doesn't support CREATE OR REPLACE TABLE, so we drop first
                    try:
                        drop_sql = f'DROP TABLE {self._to_sql(table)}'
                        logger.info(f"CTAS: Dropping existing table: {drop_sql}")
                        self.execute(drop_sql)
                        logger.info(f"CTAS: Successfully dropped table {table_name}")
                        # Clear cache after dropping
                        data_object_cache_key = _get_data_object_cache_key(table.catalog, table.db, table.name)
                        if data_object_cache_key in self._data_object_cache:
                            del self._data_object_cache[data_object_cache_key]
                    except Exception as e:
                        logger.warning(f"CTAS: Failed to drop table {table_name}: {e}")
                        raise
            
            # Don't use replace flag in CREATE statement since DB2 doesn't support it
            replace = False
            
            # Build the CREATE TABLE expression
            create_exp = self._build_create_table_exp(
                table_name_or_schema=table_name_or_schema,
                expression=expression,
                exists=False,  # Always False for DB2
                replace=replace,
                target_columns_to_types=target_columns_to_types,
                table_description=table_description,
                table_kind=table_kind,
                **kwargs,
            )
            
            # Convert to SQL
            sql = self._to_sql(create_exp)
            
            # DB2-specific fixes for CREATE TABLE AS SELECT:
            # 1. Remove the outer subquery alias that DB2 doesn't allow in CTAS
            #    Pattern: ...)) AS "_subquery" -> ...))
            sql = re.sub(
                r'\)\s+AS\s+"_subquery"',
                r')',
                sql,
                flags=re.IGNORECASE
            )
            
            # 2. DB2 requires WITH DATA but doesn't accept it after subquery parentheses
            #    We need to wrap the SELECT in parentheses: CREATE TABLE AS (...) WITH DATA
            if "WITH DATA" not in sql.upper() and "WITH NO DATA" not in sql.upper():
                # Find the position after "AS" in "CREATE TABLE ... AS"
                match = re.search(r'CREATE\s+TABLE\s+[^\s]+\s+AS\s+', sql, re.IGNORECASE)
                if match:
                    pos = match.end()
                    # Wrap the SELECT statement in parentheses
                    sql = sql[:pos] + '(' + sql[pos:].rstrip(";").rstrip() + ') WITH DATA'
                else:
                    # Fallback: just add WITH DATA at the end
                    sql = sql.rstrip(";").rstrip() + " WITH DATA"
            
            # Execute the modified SQL
            self.execute(sql, track_rows_processed=track_rows_processed)
            
            # Handle table comments if provided
            if table_description:
                self._create_table_comment(
                    table_name,
                    table_description,
                    table_kind=table_kind or "TABLE",
                )
            if column_descriptions:
                self._create_column_comments(
                    table_name,
                    column_descriptions,
                    table_kind=table_kind or "TABLE",
                )
        else:
            # For non-CTAS tables, use parent implementation
            super()._create_table(
                table_name_or_schema=table_name_or_schema,
                expression=expression,
                exists=False,  # Always False for DB2
                replace=replace,
                target_columns_to_types=target_columns_to_types,
                table_description=table_description,
                column_descriptions=column_descriptions,
                table_kind=table_kind,
                track_rows_processed=track_rows_processed,
                **kwargs,
            )
    
    def create_view(
        self,
        view_name: TableName,
        query_or_df: QueryOrDF,
        target_columns_to_types: t.Optional[t.Dict[str, exp.DataType]] = None,
        replace: bool = True,
        materialized: bool = False,
        materialized_properties: t.Optional[t.Dict[str, t.Any]] = None,
        table_description: t.Optional[str] = None,
        column_descriptions: t.Optional[t.Dict[str, str]] = None,
        view_properties: t.Optional[t.Dict[str, exp.Expression]] = None,
        source_columns: t.Optional[t.List[str]] = None,
        **create_kwargs: t.Any,
    ) -> None:
        """
        Create a view in DB2.
        
        DB2 has strict rules around view replacement similar to PostgreSQL.
        We drop the old view before creating a new one.
        """
        with self.transaction():
            if replace:
                self.drop_view(view_name, materialized=materialized)
            super().create_view(
                view_name,
                query_or_df,
                target_columns_to_types=target_columns_to_types,
                replace=False,
                materialized=materialized,
                materialized_properties=materialized_properties,
                table_description=table_description,
                column_descriptions=column_descriptions,
                view_properties=view_properties,
                source_columns=source_columns,
                **create_kwargs,
            )
    
    def drop_view(
        self,
        view_name: TableName,
        ignore_if_not_exists: bool = True,
        materialized: bool = False,
        **kwargs: t.Any,
    ) -> None:
        """
        Drop a view in DB2.
        
        DB2 doesn't support IF EXISTS or CASCADE for views.
        For staging views, try multiple naming formats due to case sensitivity issues.
        """
        table = exp.to_table(view_name)
        view_name_str = str(table.name) if hasattr(table, 'name') else str(view_name)
        schema_name = table.db or self._get_current_schema()
        
        # Check if this is a staging view
        is_staging = "sqlmesh__" in view_name_str.lower() or "sqlmesh__" in schema_name.lower()
        
        if is_staging:
            # For staging views, try multiple formats without checking existence
            drop_attempts = [
                f'DROP VIEW {self._to_sql(table)}',
                f'DROP VIEW "{schema_name}"."{view_name_str}"',
                f'DROP VIEW {schema_name.upper()}.{view_name_str.upper()}',
                f'DROP VIEW "{schema_name.lower()}"."{view_name_str.lower()}"',
            ]
            
            dropped = False
            for drop_sql in drop_attempts:
                try:
                    logger.info(f"Staging view drop attempt: {drop_sql}")
                    self.execute(drop_sql)
                    logger.info(f"Successfully dropped staging view with: {drop_sql}")
                    dropped = True
                    self._clear_data_object_cache(view_name)
                    break
                except Exception as e:
                    error_msg = str(e)
                    if 'SQL0204N' in error_msg or 'SQL0205N' in error_msg or 'does not exist' in error_msg.lower():
                        logger.debug(f"View doesn't exist with format: {drop_sql}")
                        continue
                    else:
                        logger.debug(f"Drop failed with: {drop_sql}, error: {e}")
                        continue
            
            if not dropped and not ignore_if_not_exists:
                raise SQLMeshError(f"Could not drop staging view {view_name} with any format")
        else:
            # For regular views, use the standard approach
            if ignore_if_not_exists:
                # Check if view exists
                view_only = table.name
                
                check_sql = exp.select("1").from_("SYSCAT.VIEWS").where(
                    exp.and_(
                        exp.column("VIEWSCHEMA").eq(exp.Literal.string(schema_name.upper())),
                        exp.column("VIEWNAME").eq(exp.Literal.string(view_only.upper())),
                    )
                )
                self.execute(check_sql)
                if not self.cursor.fetchone():
                    logger.debug(f"View {view_name_str} doesn't exist")
                    return
            
            # Drop view - DB2 doesn't support IF EXISTS or CASCADE for views
            drop_sql = f"DROP VIEW {self._to_sql(table)}"
            self.execute(drop_sql)
            self._clear_data_object_cache(view_name)
    
    def _get_data_objects(
        self, schema_name: SchemaName, object_names: t.Optional[t.Set[str]] = None
    ) -> t.List[DataObject]:
        """
        Returns all the data objects that exist in the given schema.
        
        Uses DB2's SYSCAT tables to query for tables and views.
        """
        catalog = self.get_current_catalog()
        schema = to_schema(schema_name).db
        
        # Query for tables
        table_query = exp.select(
            exp.Literal.string(schema).as_("schema_name"),
            exp.column("TABNAME").as_("name"),
            exp.Literal.string("TABLE").as_("type"),
        ).from_("SYSCAT.TABLES").where(
            exp.and_(
                exp.column("TABSCHEMA").eq(exp.Literal.string(schema.upper())),
                exp.column("TYPE").eq(exp.Literal.string("T")),
            )
        )
        
        # Query for views
        view_query = exp.select(
            exp.Literal.string(schema).as_("schema_name"),
            exp.column("VIEWNAME").as_("name"),
            exp.Literal.string("VIEW").as_("type"),
        ).from_("SYSCAT.VIEWS").where(
            exp.column("VIEWSCHEMA").eq(exp.Literal.string(schema.upper()))
        )
        
        # Union queries
        subquery = exp.union(table_query, view_query, distinct=False)
        query = exp.select("*").from_(subquery.subquery(alias="objs"))
        
        if object_names:
            query = query.where(exp.column("name").isin(*[n.upper() for n in object_names]))
        
        df = self.fetchdf(query)
        return [
            DataObject(
                catalog=catalog,
                schema=str(row[0]),  # schema_name column
                name=str(row[1]).lower(),  # name column
                type=DataObjectType.from_str(str(row[2])),  # type column
            )
            for row in df.itertuples(index=False, name=None)
        ]
    
    def _get_current_schema(self) -> str:
        """
        Returns the current default schema for the connection.
        
        Uses DB2's CURRENT SCHEMA special register.
        """
        result = self.fetchone("SELECT CURRENT SCHEMA FROM SYSIBM.SYSDUMMY1")
        if result and result[0]:
            return result[0].lower()
        return "public"
    
    def create_schema(
        self,
        schema_name: SchemaName,
        ignore_if_exists: bool = True,
        warn_on_error: bool = True,
        properties: t.Optional[t.List[exp.Expression]] = None,
        **kwargs: t.Any,
    ) -> None:
        """
        Create a schema in DB2.
        
        Args:
            schema_name: Name of the schema to create
            ignore_if_exists: If True, don't error if schema exists
        """
        schema = to_schema(schema_name)
        schema_name_str = schema.db
        
        if ignore_if_exists:
            # Check if schema exists (case-insensitive) - DB2 stores schema names in uppercase
            # Check both the provided case and uppercase version
            check_sql = f"""
                SELECT 1 FROM SYSCAT.SCHEMATA
                WHERE UPPER(SCHEMANAME) = UPPER('{schema_name_str}')
            """
            try:
                result = self.fetchone(check_sql)
                if result:
                    logger.debug(f"Schema {schema_name_str} already exists (case-insensitive match)")
                    return
            except Exception as e:
                logger.warning(f"Error checking schema existence: {e}")
                # Continue to try creating the schema
        
        # Create schema - DB2 will uppercase it automatically
        create_sql = exp.Create(
            this=exp.Schema(this=exp.to_identifier(schema_name_str)),
            kind="SCHEMA",
        )
        try:
            self.execute(create_sql)
        except Exception as e:
            # Only ignore if schema already exists (duplicate object error)
            # Do NOT ignore privilege errors - let them propagate
            if ignore_if_exists and is_db2_error(e, DB2ErrorCodes.DUPLICATE_OBJECT):
                logger.debug(f"Schema {schema_name_str} already exists (caught during creation)")
                return
            # Re-raise all other errors including privilege errors
            raise
    
    def drop_schema(
        self,
        schema_name: SchemaName,
        ignore_if_not_exists: bool = True,
        cascade: bool = False,
        **kwargs: t.Any,
    ) -> None:
        """
        Drop a schema in DB2.
        
        Args:
            schema_name: Name of the schema to drop
            ignore_if_not_exists: If True, don't error if schema doesn't exist
            cascade: If True, drop all objects in schema first
            
        Note:
            DB2 requires RESTRICT keyword and all objects to be dropped before dropping schema.
        """
        schema = to_schema(schema_name)
        schema_name_str = schema.db.upper()
        
        if ignore_if_not_exists:
            check_sql = exp.select("1").from_("SYSCAT.SCHEMATA").where(
                exp.column("SCHEMANAME").eq(exp.Literal.string(schema_name_str))
            )
            self.execute(check_sql)
            if not self.cursor.fetchone():
                logger.debug(f"Schema {schema_name_str} doesn't exist")
                return
        
        if cascade:
            # Drop all tables in schema first
            tables_sql = exp.select("TABNAME").from_("SYSCAT.TABLES").where(
                exp.and_(
                    exp.column("TABSCHEMA").eq(exp.Literal.string(schema_name_str)),
                    exp.column("TYPE").eq(exp.Literal.string("T")),
                )
            )
            self.execute(tables_sql)
            tables = [row[0] for row in self.cursor.fetchall()]
            
            for table in tables:
                drop_table_exp = exp.Drop(
                    this=exp.to_table(f"{schema_name_str}.{table}"),
                    kind="TABLE",
                )
                self.execute(drop_table_exp)
        
        # Drop schema - DB2 requires RESTRICT keyword, use raw SQL
        drop_sql = f"DROP SCHEMA {schema_name_str} RESTRICT"
        self.execute(drop_sql)
    
    def _merge(
        self,
        target_table: TableName,
        query: Query,
        on: exp.Expression,
        whens: exp.Whens,
    ) -> None:
        """
        Execute MERGE statement for DB2.
        
        DB2 has issues with double underscore aliases, so we use simple aliases.
        """
        # Use simple aliases without underscores for DB2
        this = exp.alias_(exp.to_table(target_table), alias="TARGET", table=True)
        using = exp.alias_(
            exp.Subquery(this=query), alias="SOURCE", copy=False, table=True
        )
        
        # Replace alias references in ON clause and WHEN clauses
        on_replaced = on.transform(
            lambda node: (
                exp.column(node.name, table="TARGET")
                if isinstance(node, exp.Column) and node.table == "__MERGE_TARGET__"
                else exp.column(node.name, table="SOURCE")
                if isinstance(node, exp.Column) and node.table == "__MERGE_SOURCE__"
                else node
            )
        )
        
        whens_replaced = whens.transform(
            lambda node: (
                exp.column(node.name, table="TARGET")
                if isinstance(node, exp.Column) and node.table == "__MERGE_TARGET__"
                else exp.column(node.name, table="SOURCE")
                if isinstance(node, exp.Column) and node.table == "__MERGE_SOURCE__"
                else node
            )
        )
        
        self.execute(
            exp.Merge(this=this, using=using, on=on_replaced, whens=whens_replaced),
            track_rows_processed=True
        )
    
    def _create_table_like(
        self,
        target_table_name: TableName,
        source_table_name: TableName,
        exists: bool,
        **kwargs: t.Any,
    ) -> None:
        """
        Create a table with the same structure as another table.
        
        DB2 supports LIKE clause in CREATE TABLE.
        """
        self.execute(
            exp.Create(
                this=exp.Schema(
                    this=exp.to_table(target_table_name),
                    expressions=[
                        exp.LikeProperty(this=exp.to_table(source_table_name))
                    ],
                ),
                kind="TABLE",
                exists=exists,
            )
        )
    def set_current_catalog(self, catalog: str) -> None:
        """
        Set the current catalog (database) for the connection.
        
        In DB2, this is done using CONNECT TO statement.
        
        Args:
            catalog: The catalog name to switch to
        """
        # DB2 uses CONNECT TO to switch databases
        self.execute(f"CONNECT TO {catalog}")
        logger.info(f"Switched to catalog: {catalog}")
    
    
    @cached_property
    def server_version(self) -> t.Tuple[int, int]:
        """
        Lazily fetch and cache major and minor DB2 server version.
        
        Returns:
            Tuple of (major_version, minor_version)
        """
        try:
            result = self.fetchone("SELECT SERVICE_LEVEL FROM SYSIBMADM.ENV_INST_INFO")
            if result and result[0]:
                version_str = result[0]
                # Parse version string (e.g., "DB2 v11.5.0.0")
                import re
                match = re.search(r"v?(\d+)\.(\d+)", version_str)
                if match:
                    return int(match.group(1)), int(match.group(2))
        except Exception as e:
            logger.warning(f"Could not determine DB2 version: {e}")
        
        return 11, 5  # Default to DB2 11.5

# Made with Bob
