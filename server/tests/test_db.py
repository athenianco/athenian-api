async def test_parallel_database_str(sdb):
    assert str(sdb).startswith("ParallelDatabase")
