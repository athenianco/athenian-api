import logging
from lzma import LZMAFile
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

from freezegun import freeze_time
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import (
    TIMESTAMP,
    BigInteger,
    Column,
    Integer,
    Text,
    create_engine,
    delete,
    insert,
    select,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql.functions import count

from athenian.api.async_utils import gather
from athenian.api.defer import with_defer
from athenian.api.internal.miners.github.commit import (
    COMMIT_FETCH_COMMITS_COLUMNS,
    _empty_dag,
    fetch_repository_commits,
)
from athenian.api.models import check_collation
from athenian.api.models.metadata.github import (
    AccountMixin,
    GitHubSchemaMixin,
    IDMixinNode,
    ParentChildMixin,
    PushCommit,
)
from tests.conftest import connect_to_db

Base = declarative_base()


class NodeCommit(
    Base,
    GitHubSchemaMixin,
    IDMixinNode,
):
    __tablename__ = "node_commit"

    oid = Column(Text, nullable=False, info={"dtype": "S40", "erase_nulls": True})
    committed_date = Column(TIMESTAMP(timezone=True), nullable=False)
    fetched_at = Column(TIMESTAMP(timezone=True), nullable=False)
    repository_id = Column(
        BigInteger, nullable=False, default=499577466, server_default="499577466",
    )


class NodeCommitParent(
    Base,
    GitHubSchemaMixin,
    ParentChildMixin,
    AccountMixin,
):
    __tablename__ = "node_commit_edge_parents"

    index = Column(Integer, nullable=False)
    fetched_at = Column(TIMESTAMP(timezone=True), nullable=False)


def insert_table(file_name, model, date_columns, engine, preprocess):
    with LZMAFile(Path(__file__).with_name(file_name)) as fin:
        df = pd.read_csv(fin, parse_dates=date_columns, infer_datetime_format=True)
        if preprocess is not None:
            df = preprocess(df)
        batch = []
        for i, t in enumerate(df.itertuples(index=False)):
            batch.append(t)
            if len(batch) == 1000 or i == len(df) - 1:
                engine.execute(insert(model).values(batch))
                batch.clear()


def _fill_fetched_at_from_committed_date(df):
    nulls = df[NodeCommit.fetched_at.name].isnull().values
    df[NodeCommit.fetched_at.name].values[nulls] = df[NodeCommit.committed_date.name].values[nulls]
    return df


@pytest.fixture(scope="module")
def mdb_torture_file(worker_id) -> Path:
    metadata_db_path = Path(__file__).with_name(f"mdb-torture-{worker_id}.sqlite")
    conn_str = f"sqlite:///{metadata_db_path}"

    for table in Base.metadata.tables.values():
        if table.schema is not None:
            table.name = ".".join([table.schema, table.name])
            table.schema = None

    if not metadata_db_path.exists():
        engine = create_engine(conn_str)
        Base.metadata.create_all(engine)
        insert_table(
            "consistency_torture_commits.csv.xz",
            NodeCommit,
            ["committed_date", "fetched_at"],
            engine,
            _fill_fetched_at_from_committed_date,
        )
        insert_table(
            "consistency_torture_edges.csv.xz", NodeCommitParent, ["fetched_at"], engine, None,
        )
        check_collation(conn_str)

    return metadata_db_path


@pytest.fixture(scope="function")
async def mdb_torture(mdb_torture_file, metadata_db, event_loop, request, worker_id):
    if metadata_db.startswith("postgresql://"):
        raise pytest.skip("incompatible with dereferenced schemas")
    if getattr(request, "param", None) is not None:
        param_file = Path(
            NamedTemporaryFile(
                delete=False, prefix=mdb_torture_file.stem, suffix=f"_{request.param}.sqlite",
            ).name,
        )
        request.addfinalizer(param_file.unlink)
        shutil.copy(mdb_torture_file, param_file)
        mdb_torture_file = param_file
    conn_str = f"sqlite:///{mdb_torture_file}"
    db = await connect_to_db(conn_str, event_loop, request)

    async def truncate(preserve_node_commit=()):
        if getattr(request, "param", None) is None:
            return
        models = (NodeCommit, NodeCommitParent)
        log = logging.getLogger("mdb_torture")
        counts = await gather(
            *(db.fetch_val(select(count()).select_from(model)) for model in models),
        )
        log.info(", ".join(f"{model.__tablename__}: {n}" for n, model in zip(counts, models)))
        await gather(
            *(
                db.execute(
                    delete(model).where(
                        model.fetched_at > request.param,
                        *(
                            (NodeCommit.oid.notin_(preserve_node_commit),)
                            if preserve_node_commit and model is NodeCommit
                            else ()
                        ),
                    ),
                )
                for model in models
            ),
        )
        counts = await gather(
            *(db.fetch_val(select(count()).select_from(model)) for model in models),
        )
        log.info(
            "-> " + ", ".join(f"{model.__tablename__}: {n}" for n, model in zip(counts, models)),
        )

    return db, truncate


async def _test_consistency_torture(
    mdb_torture,
    pdb,
    heads,
    result_consistent,
    result_len,
    includes,
    excludes,
    preserve_node_commit=(),
):
    mdb_torture, truncate = mdb_torture
    rows = await mdb_torture.fetch_all(select(NodeCommit).where(NodeCommit.oid.in_(heads)))
    assert len(rows) == len(heads)
    log = logging.getLogger("_test_consistency_torture")
    for row in rows:
        log.info(
            "resolved %s %s %s @ %s",
            row[NodeCommit.graph_id.name],
            row[NodeCommit.oid.name],
            row[NodeCommit.committed_date.name],
            row[NodeCommit.fetched_at.name],
        )
    await truncate(preserve_node_commit=preserve_node_commit)
    dags = await fetch_repository_commits(
        {"org/repo": (True, _empty_dag())},
        pd.DataFrame(
            {
                PushCommit.sha.name: np.array(
                    [row[NodeCommit.oid.name] for row in rows], dtype="S40",
                ),
                PushCommit.node_id.name: [row[NodeCommit.graph_id.name] for row in rows],
                PushCommit.committed_date.name: [
                    row[NodeCommit.committed_date.name] for row in rows
                ],
                PushCommit.repository_full_name.name: ["org/repo"] * len(rows),
            },
        ),
        COMMIT_FETCH_COMMITS_COLUMNS,
        False,
        205,
        (137,),
        mdb_torture,
        pdb,
        None,
    )
    consistent, dag = dags["org/repo"]
    assert consistent == result_consistent
    for include in includes:
        assert include.encode() in dag[0]
    for exclude in excludes:
        assert exclude.encode() not in dag[0]
    assert len(dag[0]) == result_len


@with_defer
async def test_consistency_torture_base(mdb_torture, pdb):
    await _test_consistency_torture(
        mdb_torture,
        pdb,
        ["717d445f45263d29429c0a76f2e9336c46cb229b"],
        True,
        162992,
        ["717d445f45263d29429c0a76f2e9336c46cb229b"],
        [],
    )


@pytest.mark.parametrize(
    "mdb_torture",
    [pd.Timestamp("2022-10-07 16:16:00+00")],
    indirect=["mdb_torture"],
)
@with_defer
async def test_consistency_torture_pure(mdb_torture, pdb):
    await _test_consistency_torture(
        mdb_torture,
        pdb,
        ["8fe7ea710314c0d850e09875a1c77e9fe6d2ecc4"],
        False,
        0,
        [],
        [],
    )


@pytest.mark.parametrize(
    "mdb_torture",
    [pd.Timestamp("2022-10-07 16:16:00+00")],
    indirect=["mdb_torture"],
)
@with_defer
@freeze_time("2022-10-07 16:16:00+00")
async def test_consistency_torture_oct7(mdb_torture, pdb):
    # 96815f9c8ba3cd92a9620fcb11035ee896563a39 is the main branch
    # 96815f9c8ba3cd92a9620fcb11035ee896563a39 -> ... -> ec38c1f54bbe008fb60a4e19f42199a520eec73e
    # ec38c1f54bbe008fb60a4e19f42199a520eec73e -> f1576430c35492d824255a4d2b1892517429232c
    # that edge was fetched the following day 2022-10-08 07:19:21
    #
    # 8fe7ea710314c0d850e09875a1c77e9fe6d2ecc4 -> ... -> 45b176dd4689c791126bbfb2f9b3a140d79f0b4a
    # 45b176dd4689c791126bbfb2f9b3a140d79f0b4a -> 0672cf47016ff70929632f25c8fb864919af4a75
    # 0672cf47016ff70929632f25c8fb864919af4a75 fetched 17:47
    # must discard all of them
    await _test_consistency_torture(
        mdb_torture,
        pdb,
        ["8fe7ea710314c0d850e09875a1c77e9fe6d2ecc4", "96815f9c8ba3cd92a9620fcb11035ee896563a39"],
        False,
        162555,
        [],
        [
            "96815f9c8ba3cd92a9620fcb11035ee896563a39",
            "ec38c1f54bbe008fb60a4e19f42199a520eec73e",
            "8fe7ea710314c0d850e09875a1c77e9fe6d2ecc4",
            "0672cf47016ff70929632f25c8fb864919af4a75",
        ],
    )


@pytest.mark.parametrize(
    "mdb_torture",
    [pd.Timestamp("2022-10-07 16:16:00+00")],
    indirect=["mdb_torture"],
)
@with_defer
@freeze_time("2022-10-07 16:16:00+00")
async def test_consistency_torture_second_line_of_defense(mdb_torture, pdb):
    # what happens if we fetched 0672cf47016ff70929632f25c8fb864919af4a75
    await _test_consistency_torture(
        mdb_torture,
        pdb,
        ["45b176dd4689c791126bbfb2f9b3a140d79f0b4a", "96815f9c8ba3cd92a9620fcb11035ee896563a39"],
        False,
        162555,
        [],
        [
            "96815f9c8ba3cd92a9620fcb11035ee896563a39",
            "ec38c1f54bbe008fb60a4e19f42199a520eec73e",
            "45b176dd4689c791126bbfb2f9b3a140d79f0b4a",
            "0672cf47016ff70929632f25c8fb864919af4a75",
        ],
        preserve_node_commit=["0672cf47016ff70929632f25c8fb864919af4a75"],
    )
