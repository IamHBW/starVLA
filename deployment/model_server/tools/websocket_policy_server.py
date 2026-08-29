# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import asyncio
import contextlib
import logging
import time
import traceback

import websockets.asyncio.server
import websockets.frames

# from openpi_client import base_policy as _base_policy
from . import msgpack_numpy


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 10093,
        idle_timeout: int = -1,  # Idle timeout in seconds, -1 means never auto-close
        metadata: dict | None = None,
        max_batch_size: int = 1,
        batch_wait_ms: float = 0,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must be non-negative")
        self._policy = policy  #
        self._host = host
        self._port = port
        self._metadata = {
            **(metadata or {}),
            "max_batch_size": max_batch_size,
            "batch_wait_ms": batch_wait_ms,
        }
        self._idle_timeout = idle_timeout
        self._last_active = time.time()
        self._max_batch_size = max_batch_size
        self._batch_wait_s = batch_wait_ms / 1000
        self._inference_queue = asyncio.Queue()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        batch_worker = None
        if self._max_batch_size > 1:
            batch_worker = asyncio.create_task(self._batch_worker())
        try:
            async with websockets.asyncio.server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
            ) as server:
                if self._idle_timeout > 0:
                    await self._idle_watchdog(server)
                else:
                    await server.serve_forever()
        finally:
            if batch_worker is not None:
                batch_worker.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await batch_worker

    async def _idle_watchdog(self, server):
        """Monitor idle time and shut down the server on timeout."""
        while True:
            await asyncio.sleep(5)
            if time.time() - self._last_active > self._idle_timeout:
                logging.info(f"Idle timeout ({self._idle_timeout}s) reached, shutting down server.")
                server.close()
                await server.wait_closed()
                break

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                self._last_active = time.time()  # Refresh active time on each received message
                if self._max_batch_size > 1 and msg.get("type", "infer") in ("infer", "predict_action"):
                    ret = await self._enqueue_inference(msg)
                else:
                    ret = self._route_message(msg)  # route message
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    async def _enqueue_inference(self, msg: dict) -> dict:
        future = asyncio.get_running_loop().create_future()
        await self._inference_queue.put((msg, future))
        return await future

    async def _batch_worker(self) -> None:
        while True:
            first = await self._inference_queue.get()
            items = [first]
            deadline = asyncio.get_running_loop().time() + self._batch_wait_s
            while len(items) < self._max_batch_size:
                try:
                    if self._batch_wait_s == 0:
                        items.append(self._inference_queue.get_nowait())
                    else:
                        remaining = deadline - asyncio.get_running_loop().time()
                        if remaining <= 0:
                            break
                        items.append(await asyncio.wait_for(self._inference_queue.get(), remaining))
                except (asyncio.QueueEmpty, asyncio.TimeoutError):
                    break
            try:
                self._route_inference_batches(items)
            except Exception as exc:
                logging.exception("Failed to route inference micro-batch")
                for msg, future in items:
                    if not future.done():
                        future.set_result(
                            {
                                "status": "error",
                                "ok": False,
                                "type": "inference_result",
                                "request_id": msg.get("request_id", "default"),
                                "error": {"message": str(exc)},
                            }
                        )
            finally:
                for _ in items:
                    self._inference_queue.task_done()

    @staticmethod
    def _slice_batch_value(value, start: int, end: int, total: int):
        shape = getattr(value, "shape", ())
        if shape and shape[0] == total:
            return value[start:end]
        if isinstance(value, (list, tuple)) and len(value) == total:
            return value[start:end]
        return value

    def _route_inference_batches(self, items) -> None:
        groups = {}
        packer = msgpack_numpy.Packer()
        for msg, future in items:
            payload = msg.get("payload", msg)
            examples = payload.get("examples") if isinstance(payload, dict) else None
            if not isinstance(examples, list) or not examples:
                if not future.done():
                    future.set_result(self._route_message(msg))
                continue
            kwargs = {key: value for key, value in payload.items() if key != "examples"}
            groups.setdefault(packer.pack(kwargs), []).append((msg, examples, kwargs, future))

        for group in groups.values():
            all_examples = []
            spans = []
            for msg, examples, _, future in group:
                start = len(all_examples)
                all_examples.extend(examples)
                spans.append((msg, future, start, len(all_examples)))
            try:
                output = self._policy.predict_action(examples=all_examples, **group[0][2])
                actions = output.get("actions") if isinstance(output, dict) else None
                if not getattr(actions, "shape", ()) or actions.shape[0] != len(all_examples):
                    raise ValueError("Batched policy output must have one action chunk per example")
                for msg, future, start, end in spans:
                    data = {
                        key: self._slice_batch_value(value, start, end, len(all_examples))
                        for key, value in output.items()
                    }
                    if not future.done():
                        future.set_result(
                            {
                                "status": "ok",
                                "ok": True,
                                "type": "inference_result",
                                "request_id": msg.get("request_id", "default"),
                                "data": data,
                            }
                        )
            except Exception as exc:
                logging.exception(
                    "Batched policy inference error (request_ids=%s)",
                    [msg.get("request_id", "default") for msg, _, _, _ in spans],
                )
                for msg, future, _, _ in spans:
                    if not future.done():
                        future.set_result(
                            {
                                "status": "error",
                                "ok": False,
                                "type": "inference_result",
                                "request_id": msg.get("request_id", "default"),
                                "error": {"message": str(exc)},
                            }
                        )

    # route logic: recognize request from client
    def _route_message(self, msg: dict) -> dict:
        """
        Route rules (fault-tolerant):
        - Supports messages of form:
            {"type": "ping|init|infer|reset", "request_id": "...", "payload": {...}}
          or a flat dict (will be treated as payload).
        - Does NOT raise inside this function: all exceptions are caught and encoded in response.
        """
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")  # default = infer
        payload = msg.get("payload", msg)  # when no explicit payload, treat top-level as payload

        # ping
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}

        # infer --> framework.predict_action
        elif mtype == "infer" or mtype == "predict_action":
            # Basic payload sanity
            if not isinstance(payload, dict):
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {"message": "Payload must be a dict", "payload_type": str(type(payload))},
                }
            try:
                output_dict = self._policy.predict_action(**payload)
            except Exception as e:
                logging.exception("Policy inference error (request_id=%s)", req_id)
                logging.exception(e)

                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {
                        "message": str(e),
                    },
                }
            data = output_dict
            return {
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "request_id": req_id,
                "data": data,
            }

        # unknow request type
        else:
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": req_id,
                "error": {"message": f"Unsupported message type '{mtype}'"},
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()
    raise NotImplementedError("This module is not intended to be run directly.")
#
#  Instead, it should be imported and used in a server context.
