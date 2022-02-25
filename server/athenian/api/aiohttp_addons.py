import asyncio
import functools

import aiohttp


def create_aiohttp_closed_event(session: aiohttp.ClientSession) -> asyncio.Event:
    """Work around aiohttp issue that doesn't properly close transports on exit.

    See https://github.com/aio-libs/aiohttp/issues/1925#issuecomment-639080209

    Not needed in 4.0+.

    Returns:
       An event that will be set once all transports have been properly closed.
    """
    all_is_lost = asyncio.Event()
    if session.connector is None:
        all_is_lost.set()
        return all_is_lost
    transports = 0

    def connection_lost(exc, orig_lost):
        nonlocal transports
        try:
            orig_lost(exc)
        finally:
            transports -= 1
            if transports == 0:
                all_is_lost.set()

    def eof_received(orig_eof_received):
        try:
            orig_eof_received()
        except AttributeError:
            # It may happen that eof_received() is called after
            # _app_protocol and _transport are set to None.
            pass

    for conn in session.connector._conns.values():
        for handler, _ in conn:
            if (proto := getattr(handler.transport, "_ssl_protocol", None)) is None:
                continue
            transports += 1
            orig_lost = proto.connection_lost
            orig_eof_received = proto.eof_received
            proto.connection_lost = functools.partial(
                connection_lost, orig_lost=orig_lost,
            )
            proto.eof_received = functools.partial(
                eof_received, orig_eof_received=orig_eof_received,
            )

    if transports == 0:
        all_is_lost.set()

    return all_is_lost
