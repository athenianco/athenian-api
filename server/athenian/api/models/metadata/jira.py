from sqlalchemy import BigInteger, Boolean, Column, JSON, SmallInteger, Text, TIMESTAMP
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.ext.declarative import declarative_base


class AccountIDMixin:
    acc_id = Column(BigInteger, primary_key=True)


Base = declarative_base(cls=AccountIDMixin)
Base.__table_args__ = {"schema": "jira"}


class Epic(Base):
    __tablename__ = "epic"

    id = Column(Text, primary_key=True)
    key = Column(Text, nullable=False)
    name = Column(Text, nullable=False)
    title = Column(Text, nullable=False)
    done = Column(Boolean, nullable=False)


class Issue(Base):
    __tablename__ = "issue"

    id = Column(Text, primary_key=True)
    project_id = Column(Text, nullable=False)
    parent_id = Column(Text)
    key = Column(Text, nullable=False)
    title = Column(Text, nullable=False)
    type = Column(Text, nullable=False)
    status = Column(Text)
    labels = Column(postgresql.ARRAY(Text).with_variant(JSON(), sqlite.dialect.name))
    components = Column(postgresql.ARRAY(Text).with_variant(JSON(), sqlite.dialect.name))
    epic_id = Column(Text)
    created = Column(TIMESTAMP(timezone=True), nullable=False)
    updated = Column(TIMESTAMP(timezone=True), nullable=False)
    resolved = Column(TIMESTAMP(timezone=True))
    reporter_id = Column(Text, nullable=False)
    reporter_display_name = Column(Text, nullable=False)
    assignee_id = Column(Text)
    assignee_display_name = Column(Text)
    commenters_ids = Column(
        postgresql.ARRAY(BigInteger).with_variant(JSON(), sqlite.dialect.name))
    commenters_display_names = Column(
        postgresql.ARRAY(Text).with_variant(JSON(), sqlite.dialect.name))
    priority_id = Column(Text)  # TODO(vmarkovtsev): make it nullable=False
    priority_name = Column(Text, nullable=False)


class Component(Base):
    __tablename__ = "component"

    id = Column(Text, primary_key=True)
    name = Column(Text, nullable=False)


class User(Base):
    __tablename__ = "user"

    id = Column(Text, primary_key=True)
    type = Column(Text, nullable=False)
    display_name = Column(Text, nullable=False)
    avatar_url = Column(Text, nullable=False)


class Priority(Base):
    __tablename__ = "priority"

    id = Column(Text, primary_key=True)
    name = Column(Text, nullable=False)
    rank = Column(SmallInteger, nullable=False)
    status_color = Column(Text, nullable=False)
    icon_url = Column(Text)


class AthenianIssue(Base):
    __tablename__ = "athenian_issue"

    id = Column(Text, primary_key=True)
    work_began = Column(TIMESTAMP(timezone=True))
    resolved = Column(TIMESTAMP(timezone=True))


class Installation(Base):
    __tablename__ = "installation"

    base_url = Column(Text)


class Project(Base):
    __tablename__ = "project"

    id = Column(Text, primary_key=True)
    key = Column(Text)
    name = Column(Text)
    avatar_url = Column(Text)


class IssueType(Base):
    __tablename__ = "issue_type"

    id = Column(Text, primary_key=True)
    project_id = Column(Text, nullable=False)
    name = Column(Text, nullable=False)
    description = Column(Text)
    icon_url = Column(Text, nullable=False)
