from sqlglot.dialects.dialect import DialectType

# Sizes based on a composite key/index of two text fields with 4 bytes per characters.
MAX_TEXT_INDEX_LENGTH = {
    "mysql": "250",  # 250 characters per column, <= 767 byte index size limit
    "tsql": "450",  # 450 bytes per column, <= 900 byte index size limit
    "db2": "255",  # DB2 has strict primary key size limits, keep it conservative
}


def index_text_type(dialect: DialectType) -> str:
    """
    MySQL, MSSQL, and DB2 cannot create indexes or primary keys on TEXT/CLOB fields; they
    require that the fields have a VARCHAR type of fixed length.

    This helper abstracts away the type of such fields.
    """

    return (
        f"VARCHAR({MAX_TEXT_INDEX_LENGTH[str(dialect)]})"
        if dialect in MAX_TEXT_INDEX_LENGTH
        else "TEXT"
    )


def blob_text_type(dialect: DialectType) -> str:
    """
    Returns the appropriate large text type for the dialect.
    
    - MySQL: LONGTEXT (supports larger data than TEXT)
    - DB2: VARCHAR(32000) (max VARCHAR length, avoids CLOB limitations in certain contexts)
    - Others: TEXT
    """
    if dialect == "mysql":
        return "LONGTEXT"
    elif dialect == "db2":
        return "VARCHAR(32000)"  # DB2's max VARCHAR length, avoids CLOB limitations
    else:
        return "TEXT"
