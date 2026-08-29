from unittest import mock

from deployment.model_server.tools import websocket_policy_client


def test_connect_keepalive_matches_websockets_api():
    calls = []

    class Connection:
        def recv(self):
            return b"metadata"

    def connect_legacy(uri, *, compression, max_size, additional_headers, open_timeout):
        calls.append({})
        return Connection()

    def connect_current(
        uri, *, compression, max_size, additional_headers, open_timeout, ping_interval, ping_timeout
    ):
        calls.append({"ping_interval": ping_interval, "ping_timeout": ping_timeout})
        return Connection()

    with mock.patch.object(websocket_policy_client.msgpack_numpy, "unpackb", return_value={}):
        for connect in (connect_legacy, connect_current):
            with mock.patch.object(websocket_policy_client.websockets.sync.client, "connect", connect):
                websocket_policy_client.WebsocketClientPolicy()

    assert calls == [{}, {"ping_interval": None, "ping_timeout": 60}]
