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
    CATALOG_SUPPORT = CatalogSupport.SINGLE_CATALOG_ONLY
    COMMENT_CREATION_TABLE = CommentCreationTable.COMMENT_COMMAND_ONLY
    COMMENT_CREATION_VIEW = CommentCreationView.COMMENT_COMMAND_ONLY
    SUPPORTS_QUERY_EXECUTION_TRACKING = True
    SUPPORTED_DROP_CASCADE_OBJECT_KINDS = ["SCHEMA", "TABLE", "VIEW"]
    HAS_VIEW_BINDING = False
    MAX_IDENTIFIER_LENGTH: t.Optional[int] = 128
    # DB2 requires FROM clause for CURRENT SERVER
    CURRENT_CATALOG_EXPRESSION = exp.column("CURRENT SERVER")
    
    # DB2 System Schemas (to exclude from operations)
    SYSTEM_SCHEMAS = {
        'SYSCAT', 'SYSFUN', 'SYSIBM', 'SYSIBMADM',
        'SYSPROC', 'SYSPUBLIC', 'SYSSTAT', 'SYSTOOLS'
    }
    
    def get_current_catalog(self) -> t.Optional[str]:
        """
        Returns the catalog name of the current connection.
        DB2 requires FROM SYSIBM.SYSDUMMY1 for special registers.
        """
        result = self.fetchone("SELECT CURRENT SERVER FROM SYSIBM.SYSDUMMY1")
        if result:
            return result[0]
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
        
        type_mapping = {
            "INTEGER": "INT",
            "BIGINT": "BIGINT",
            "SMALLINT": "SMALLINT",
            "DOUBLE": "DOUBLE",
            "REAL": "REAL",
            "DECIMAL": f"DECIMAL({length},{scale})",
            "NUMERIC": f"DECIMAL({length},{scale})",
            "VARCHAR": f"VARCHAR({length})",
            "CHAR": f"CHAR({length})",
            "CHARACTER": f"CHAR({length})",
            "DATE": "DATE",
            "TIMESTAMP": "TIMESTAMP",
            "TIME": "TIME",
            "BLOB": "BLOB",
            "CLOB": "CLOB",
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
        
        DB2 stores identifiers differently based on quoting:
        - Unquoted identifiers: stored as UPPERCASE
        - Quoted identifiers (with underscores): stored as lowercase (case-preserved)
        
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
        
        # DB2 case handling:
        # - Names with underscores are quoted and stored in lowercase
        # - Names without underscores are unquoted and stored in UPPERCASE
        if "_" in schema_name:
            schema_name_normalized = schema_name.lower()
        else:
            schema_name_normalized = schema_name.upper()
            
        if "_" in table_name_str:
            table_name_normalized = table_name_str.lower()
        else:
            table_name_normalized = table_name_str.upper()
        
        # Query DB2's SYSCAT.TABLES
        sql = exp.select("1").from_("SYSCAT.TABLES").where(
            exp.and_(
                exp.column("TABSCHEMA").eq(exp.Literal.string(schema_name_normalized)),
                exp.column("TABNAME").eq(exp.Literal.string(table_name_normalized)),
            )
        )
        
        self.execute(sql)
        result = self.cursor.fetchone()
        
        exists = result[0] == 1 if result is not None else False
        
        # Update cache
        if exists:
            self._data_object_cache[data_object_cache_key] = DataObject(
                name=table_name_str,
                schema=schema_name,
                type=DataObjectType.TABLE,
            )
        
        return exists
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
        
        # For CREATE TABLE AS SELECT, we need to intercept and modify the SQL
        # to remove subquery aliases and ensure WITH DATA is present
        if expression and isinstance(expression, (exp.Select, exp.Subquery)):
            import re
            
            # For CTAS, drop the table if it exists (DB2 doesn't support CREATE OR REPLACE TABLE)
            # This handles the case where the table structure was created but not populated
            table = exp.to_table(table_name)
            
            # Try to drop the table - ignore errors if it doesn't exist
            try:
                drop_sql = f'DROP TABLE {self._to_sql(table)}'
                logger.info(f"CTAS: Attempting to drop existing table: {drop_sql}")
                self.execute(drop_sql)
                logger.info(f"CTAS: Successfully dropped table {table_name}")
                # Clear cache after dropping
                data_object_cache_key = _get_data_object_cache_key(table.catalog, table.db, table.name)
                if data_object_cache_key in self._data_object_cache:
                    del self._data_object_cache[data_object_cache_key]
            except Exception as e:
                # Table doesn't exist or other error - that's fine, we'll create it
                logger.info(f"CTAS: Table {table_name} doesn't exist or couldn't be dropped (expected): {e}")
            
            # Don't use replace flag since DB2 doesn't support CREATE OR REPLACE TABLE
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
        
        DB2 doesn't support IF EXISTS or CASCADE for views, so we check existence first.
        """
        table = exp.to_table(view_name)
        view_name_str = table.sql(dialect=self.dialect)
        
        if ignore_if_not_exists:
            # Check if view exists
            schema = table.db or self.get_current_catalog()
            view_only = table.name
            
            check_sql = exp.select("1").from_("SYSCAT.VIEWS").where(
                exp.and_(
                    exp.column("VIEWSCHEMA").eq(exp.Literal.string(schema.upper() if schema else "")),
                    exp.column("VIEWNAME").eq(exp.Literal.string(view_only.upper())),
                )
            )
            self.execute(check_sql)
            if not self.cursor.fetchone():
                logger.debug(f"View {view_name_str} doesn't exist")
                return
        
        # Drop view - DB2 doesn't support IF EXISTS or CASCADE for views
        drop_sql = f"DROP VIEW {view_name_str}"
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
        schema_name_str = schema.db.upper()
        
        if ignore_if_exists:
            # Check if schema exists
            check_sql = exp.select("1").from_("SYSCAT.SCHEMATA").where(
                exp.column("SCHEMANAME").eq(exp.Literal.string(schema_name_str))
            )
            self.execute(check_sql)
            if self.cursor.fetchone():
                logger.debug(f"Schema {schema_name_str} already exists")
                return
        
        # Create schema using sqlglot
        create_sql = exp.Create(
            this=exp.Schema(this=exp.to_identifier(schema_name_str)),
            kind="SCHEMA",
        )
        self.execute(create_sql)
    
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
