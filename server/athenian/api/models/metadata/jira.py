from sqlalchemy import (
    JSON,
    REAL,
    TIMESTAMP,
    BigInteger,
    Boolean,
    Column,
    Integer,
    Numeric,
    SmallInteger,
    Text,
    cast,
)
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import column_property


class AccountIDMixin:
    acc_id = Column(BigInteger, primary_key=True)


Base = declarative_base(cls=AccountIDMixin)
Base.__table_args__ = {"schema": "jira"}
HSTORE = postgresql.HSTORE().with_variant(JSON(), sqlite.dialect.name)


class Epic(Base):
    __tablename__ = "epic"

    id = Column(Text, primary_key=True)
    key = Column(Text, nullable=False)
    name = Column(Text, nullable=False)
    title = Column(Text, nullable=False)
    done = Column(Boolean, nullable=False)


class Issue(Base):
    __tablename__ = "issue"

    id = Column(Text, primary_key=True, info={"dtype": "S12"})
    project_id = Column(Text, nullable=False, info={"dtype": "S8", "reset_nulls": True})
    parent_id = Column(Text, info={"dtype": "S12", "reset_nulls": True})
    key = Column(Text, nullable=False)
    title = Column(Text, nullable=False)
    type = Column(Text, nullable=False)
    type_id = Column(Text, nullable=False, info={"dtype": "S8", "reset_nulls": True})
    status = Column(Text)
    status_id = Column(Text, nullable=False, info={"dtype": "S8", "reset_nulls": True})
    labels = Column(
        postgresql.ARRAY(Text).with_variant(JSON(), sqlite.dialect.name), nullable=False,
    )
    components = Column(postgresql.ARRAY(Text).with_variant(JSON(), sqlite.dialect.name))
    epic_id = Column("athenian_epic_id", Text, info={"dtype": "S12", "reset_nulls": True})
    created = Column(TIMESTAMP(timezone=True), nullable=False)
    updated = Column(TIMESTAMP(timezone=True), nullable=False)
    resolved = Column(TIMESTAMP(timezone=True))
    reporter_id = Column(Text, nullable=False)
    reporter_display_name = Column(Text, nullable=False)
    assignee_id = Column(Text)
    assignee_display_name = Column(Text)
    commenters_ids = Column(postgresql.ARRAY(BigInteger).with_variant(JSON(), sqlite.dialect.name))
    commenters_display_names = Column(
        postgresql.ARRAY(Text).with_variant(JSON(), sqlite.dialect.name),
    )
    comments_count = Column(Integer, nullable=False)
    priority_id = Column(Text, nullable=True, info={"dtype": "S8"})
    priority_name = Column(Text, nullable=True)
    url = Column(Text, nullable=False)
    is_deleted = Column(Boolean, nullable=False, default=False, server_default="false")
    story_points_ = Column(Numeric, name="story_points")
    story_points = column_property(cast(story_points_, REAL))


Issue.story_points.default = Issue.story_points.server_default = None
Issue.story_points.nullable = True


class Component(Base):
    __tablename__ = "component"

    id = Column(Text, primary_key=True)
    name = Column(Text, nullable=False)


class User(Base):
    __tablename__ = "user"

    id = Column(Text, primary_key=True)
    type = Column(Text, nullable=False)
    display_name = Column(Text, nullable=False)
    active = Column(Boolean)
    avatar_url = Column(Text, nullable=False)


class Priority(Base):
    __tablename__ = "priority"

    id = Column(Text, primary_key=True, info={"dtype": "S8", "erase_nulls": True})
    name = Column(Text, nullable=False)
    rank = Column(SmallInteger, nullable=False)
    status_color = Column(Text, nullable=False)
    icon_url = Column(Text)


class Status(Base):
    __tablename__ = "api_statuses"

    CATEGORY_TODO = "To Do"
    CATEGORY_IN_PROGRESS = "In Progress"
    CATEGORY_DONE = "Done"

    id = Column(Text, primary_key=True, info={"dtype": "S8", "erase_nulls": True})
    name = Column(Text, nullable=False)
    category_name = Column(Text, nullable=False)
    color = Column(Text)
    icon_url = Column(Text)


class AthenianIssue(Base):
    __tablename__ = "athenian_issue"

    id = Column(Text, primary_key=True)
    work_began = Column(TIMESTAMP(timezone=True))
    resolved = Column(TIMESTAMP(timezone=True))
    updated = Column(TIMESTAMP(timezone=True))
    nested_assignee_display_names = Column(HSTORE, nullable=False)


class Installation(Base):
    __tablename__ = "installation"

    base_url = Column(Text)


class Project(Base):
    __tablename__ = "project"

    id = Column(Text, primary_key=True, info={"dtype": "S8", "erase_nulls": True})
    key = Column(Text)
    name = Column(Text)
    avatar_url = Column(Text)
    is_deleted = Column(Boolean, default=False, server_default="false")


class IssueType(Base):
    __tablename__ = "issue_type"

    id = Column(Text, primary_key=True, info={"dtype": "S8", "erase_nulls": True})
    project_id = Column(
        Text, primary_key=True, nullable=False, info={"dtype": "S8", "reset_nulls": True},
    )
    name = Column(Text, nullable=False)
    description = Column(Text)
    icon_url = Column(Text, nullable=False)
    is_subtask = Column(Boolean, nullable=False)
    is_epic = Column(Boolean, nullable=False)
    normalized_name = Column(Text, nullable=False)


class Progress(Base):
    __tablename__ = "progress"

    event_id = Column(Text, primary_key=True)
    event_type = Column(Text, primary_key=True)
    current = Column(BigInteger, nullable=False)
    total = Column(BigInteger, nullable=False)
    is_initial = Column(Boolean, nullable=False)
    started_at = Column(TIMESTAMP(timezone=True), nullable=False)
    end_at = Column(TIMESTAMP(timezone=True))
