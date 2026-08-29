import asyncio
import contextlib
import unittest

import numpy as np

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


class _FakePolicy:
    def __init__(self):
        self.batch_sizes = []

    def predict_action(self, examples, **kwargs):
        self.batch_sizes.append(len(examples))
        markers = np.asarray([example["marker"] for example in examples], dtype=np.float32)
        return {"actions": np.broadcast_to(markers[:, None, None], (len(examples), 8, 7)).copy()}


class WebsocketPolicyServerBatchingTest(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_requests_share_one_forward_and_are_split(self):
        policy = _FakePolicy()
        server = WebsocketPolicyServer(policy, max_batch_size=4, batch_wait_ms=20)
        worker = asyncio.create_task(server._batch_worker())
        try:
            responses = await asyncio.gather(
                *(
                    server._enqueue_inference(
                        {
                            "type": "infer",
                            "request_id": str(marker),
                            "payload": {"examples": [{"marker": marker}], "unnorm_key": "franka"},
                        }
                    )
                    for marker in range(4)
                )
            )
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker

        self.assertEqual(policy.batch_sizes, [4])
        for marker, response in enumerate(responses):
            self.assertEqual(response["request_id"], str(marker))
            self.assertEqual(response["data"]["actions"].shape, (1, 8, 7))
            self.assertTrue((response["data"]["actions"] == marker).all())

if __name__ == "__main__":
    unittest.main()
