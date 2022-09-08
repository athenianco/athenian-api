"""Tools for factories targeting SQLAlchemy models."""

from typing import Any, Optional

from factory.base import BaseFactory, FactoryMetaClass, resolve_attribute
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.orm.decl_api import DeclarativeMeta


class SQLAlchemyModelFactoryMetaClass(FactoryMetaClass):
    """Extend `FactoryMetaClass` to add declarations detected from sqlalchemy Model."""

    def __new__(mcs, class_name, bases, attrs):
        Model = mcs._get_factory_model_class(bases, attrs)

        if Model is not None:
            # iterate Model fields, skipping those private and those already
            # explicitly defined in factory class
            for field_name, model_field in Model.__dict__.items():
                if (
                    field_name.startswith("_")
                    or field_name in attrs
                    or not isinstance(model_field, InstrumentedAttribute)
                ):
                    continue
                # if a factory declaration can be extracted from the the model attribute
                # then set it as a factory class field
                try:
                    attrs[field_name] = mcs._model_field_declaration(model_field)
                except mcs.UnhandledModelAttr:
                    continue
        # call `FactoryMetaClass` __new__ that will apply automatic and explicit attrs
        return super().__new__(mcs, class_name, bases, attrs)

    @classmethod
    def _get_factory_model_class(mcs, bases, attrs) -> Optional[DeclarativeMeta]:
        # get Model from class Meta attr or bases _meta object,
        attrs_meta = attrs.get("Meta", None)
        base_meta = resolve_attribute("_meta", bases)
        try:
            return attrs_meta.model
        except AttributeError:
            try:
                return base_meta.model
            except AttributeError:
                return None

    @classmethod
    def _model_field_declaration(mcs, model_field: InstrumentedAttribute) -> Any:
        """Return the factory field declaration for the model field, if possible."""

        # of course a lot more could be implemented here, for instance using `factory.Sequence`
        # for primary keys

        if model_field.default:
            arg = model_field.default.arg
            # sqlalchemy default can be a callable, it should receive the statemente context
            # but for now we pass None and let break those requiring
            if callable(arg):
                try:
                    return arg(None)
                except AttributeError:
                    raise mcs.UnhandledModelAttr()
            else:
                return arg

        if model_field.server_default:
            if str(model_field.type) == "ARRAY":
                # for ARRAY the server_default must be expressed as a python array
                if model_field.server_default.arg == "{}":
                    return []
            else:
                return model_field.server_default.arg
        if model_field.nullable:
            return None

        raise mcs.UnhandledModelAttr()

    class UnhandledModelAttr(Exception):
        pass


class SQLAlchemyModelFactory(BaseFactory, metaclass=SQLAlchemyModelFactoryMetaClass):
    """Factory capable to add some fields automatically by parsing sqlalchemy model.

    This is different from `factory` SQLAlchemyModelFactory, which can automatically
    store objects in the session but does not declare fields automatically.

    Factory `Model` should derive from sqlalchemy `DeclarativeMeta`.

    """
