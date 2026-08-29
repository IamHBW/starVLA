import unittest
from unittest import mock

import numpy as np

from examples.simBenchmarks.LIBERO.eval_files import model2libero_interface


class _NoRequestClient:
    def predict_action(self, _):
        raise AssertionError("cached action must not issue an RPC")


class LiberoModelClientCacheTest(unittest.TestCase):
    def test_cached_step_skips_resize_and_rpc(self):
        model = model2libero_interface.ModelClient.__new__(model2libero_interface.ModelClient)
        model.client = _NoRequestClient()
        model.action_chunk_size = 8
        model.raw_actions = np.arange(56, dtype=np.float32).reshape(8, 7)
        model.task_description = "task"
        model.image_size = (224, 224)
        model.unnorm_key = "franka"
        model.use_ddim = True
        model.num_ddim_steps = 4
        example = {"image": [np.zeros((256, 256, 3), dtype=np.uint8)] * 2, "lang": "task"}

        with mock.patch.object(
            model2libero_interface.Image,
            "fromarray",
            side_effect=AssertionError("unexpected resize"),
        ):
            action = model.step(example, step=1)["raw_action"]

        np.testing.assert_array_equal(action["world_vector"], model.raw_actions[1, :3])


if __name__ == "__main__":
    unittest.main()
