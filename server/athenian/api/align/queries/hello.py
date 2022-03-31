from ariadne import QueryType

query = QueryType()


@query.field("hello")
def resolve_hello(_, info):
    """Return `Query.hello`."""
    request = info.context
    user_agent = request.headers.get("user-agent", "guest")
    return "Hello, %s!" % user_agent
