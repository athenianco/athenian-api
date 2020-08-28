from sqlalchemy import ARRAY, BigInteger, Boolean, Column, JSON, Text, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()
Base.__table_args__ = {"schema": "jira"}


class Epic(Base):
    __tablename__ = "epic"

    acc_id = Column(BigInteger, primary_key=True)
    id = Column(BigInteger, primary_key=True)
    key = Column(Text, nullable=False)
    name = Column(Text, nullable=False)
    title = Column(Text, nullable=False)
    done = Column(Boolean, nullable=False)


class Issue(Base):
    __tablename__ = "issue"

    acc_id = Column(BigInteger, primary_key=True)
    id = Column(BigInteger, primary_key=True)
    key = Column(Text, nullable=False)
    title = Column(Text, nullable=False)
    type = Column(Text, nullable=False)
    status = Column(Text)
    labels = ARRAY(Text).with_variant(JSON, "sqlite")
    epic_id = Column(BigInteger)
    created = Column(TIMESTAMP(timezone=True), nullable=False)
    updated = Column(TIMESTAMP(timezone=True), nullable=False)
    resolved = Column(TIMESTAMP(timezone=True))
