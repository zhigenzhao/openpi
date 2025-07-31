import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, 
                    compression=None, 
                    max_size=None, 
                    additional_headers=headers,
                    ping_timeout=60,  # 60 second ping timeout
                    ping_interval=30,  # ping every 30 seconds
                    close_timeout=30   # 30 second close timeout
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        try:
            data = self._packer.pack(obs)
            self._ws.send(data)
            response = self._ws.recv()
            if isinstance(response, str):
                # we're expecting bytes; if the server sends a string, it's an error.
                raise RuntimeError(f"Error in inference server:\n{response}")
            return msgpack_numpy.unpackb(response)
        except (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedError) as e:
            logging.warning(f"WebSocket connection lost: {e}. Attempting to reconnect...")
            # Reconnect and retry once
            self._ws, self._server_metadata = self._wait_for_server()
            data = self._packer.pack(obs)
            self._ws.send(data)
            response = self._ws.recv()
            if isinstance(response, str):
                raise RuntimeError(f"Error in inference server:\n{response}")
            return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass
