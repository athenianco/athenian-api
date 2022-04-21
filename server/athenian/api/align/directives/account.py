from typing import Any, Union

from ariadne import SchemaDirectiveVisitor
from graphql import GraphQLArgument, GraphQLField, GraphQLInputField, GraphQLInputObjectType, \
    GraphQLInterfaceType, GraphQLObjectType, GraphQLResolveInfo

from athenian.api.controllers.account import check_account_expired, \
    get_user_account_status_from_request
from athenian.api.models.web import UnauthorizedError
from athenian.api.response import ResponseError


class AccountDirective(SchemaDirectiveVisitor):
    """
    GraphQL directive to mark account IDs.

    We check whether the user is an active account member.
    """

    def visit_argument_definition(
        self,
        argument: GraphQLArgument,
        field: GraphQLField,
        object_type: Union[GraphQLObjectType, GraphQLInterfaceType],
    ) -> GraphQLArgument:
        """Override the resolver for the field to check the account membership and status."""
        original_resolve = field.resolve

        async def check_account_membership(obj: Any, info: GraphQLResolveInfo, **kwargs) -> Any:
            account = None
            for key, val in kwargs.items():
                if field.args[key] is argument:
                    account = val
            assert account is not None
            info.context.account = account
            await get_user_account_status_from_request(info.context, account)
            if await check_account_expired(info.context, info.context.app["auth"].log):
                raise ResponseError(UnauthorizedError(f"Account {account} has expired."))
            return await original_resolve(obj, info, **kwargs)

        field.resolve = check_account_membership
        return argument

    def visit_input_field_definition(
        self, field: GraphQLInputField, object_type: GraphQLInputObjectType,
    ) -> GraphQLInputField:
        """Override the resolver in mutations."""
        raise NotImplementedError


directives = {"account": AccountDirective}
