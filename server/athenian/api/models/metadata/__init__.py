def hack_sqlite_arrays():
    """Hack SQLite compiler to handle ARRAY fields."""
    from sqlalchemy.dialects.sqlite.base import SQLiteTypeCompiler

    SQLiteTypeCompiler.visit_ARRAY = lambda self, type_, **kw: "JSON"
