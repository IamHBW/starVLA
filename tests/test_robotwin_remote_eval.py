import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from deployment.model_server.policy_norm_processor import _build_dataset_metadata
from examples.simBenchmarks.Robotwin.eval_files import model2robotwin_interface
from examples.simBenchmarks.Robotwin.eval_files import aggregate_eval
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


class RoboTwinRemoteEvalTest(unittest.TestCase):
    def test_action_only_checkpoint_builds_empty_state_metadata(self):
        metadata = _build_dataset_metadata(
            stats_for_key={
                "action": {
                    "min": [0.0],
                    "max": [1.0],
                    "mean": [0.5],
                    "std": [0.1],
                    "q01": [0.0],
                    "q99": [1.0],
                }
            },
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            action_keys=["action.joint"],
            state_keys=[],
        )

        self.assertEqual(metadata.statistics.state, {})
        self.assertEqual(metadata.modalities.state, {})

    @mock.patch.object(model2robotwin_interface, "WebsocketClientPolicy")
    def test_client_rejects_wrong_server_checkpoint(self, client_type):
        client_type.return_value.get_server_metadata.return_value = {
            "ckpt_path": "/server/another_checkpoint.pt",
            "action_chunk_size": 32,
        }

        with self.assertRaisesRegex(RuntimeError, "checkpoint mismatch"):
            model2robotwin_interface.ModelClient("steps_10000_pytorch_model")

    @mock.patch.object(model2robotwin_interface, "WebsocketClientPolicy")
    def test_client_requires_32_step_chunks(self, client_type):
        client_type.return_value.get_server_metadata.return_value = {
            "ckpt_path": "/server/steps_10000.pt",
            "action_chunk_size": 16,
        }
        with self.assertRaisesRegex(RuntimeError, "action_chunk_size=32"):
            model2robotwin_interface.ModelClient("steps_10000.pt")

    @mock.patch.object(model2robotwin_interface, "WebsocketClientPolicy")
    def test_client_validates_actions_and_omits_state(self, client_type):
        client = client_type.return_value
        client.get_server_metadata.return_value = {
            "ckpt_path": "/server/steps_10000.pt",
            "action_chunk_size": 32,
        }
        client.predict_action.return_value = {
            "data": {"actions": np.zeros((1, 32, 14), dtype=np.float32)}
        }
        model = model2robotwin_interface.ModelClient("steps_10000.pt")
        action = model.step(
            {
                "lang": "unseen instruction",
                "image": [np.zeros((10, 20, 3), dtype=np.uint8) for _ in range(3)],
                "state": np.zeros(14),
            }
        )
        self.assertEqual(action.shape, (14,))
        request = client.predict_action.call_args.args[0]
        self.assertNotIn("state", request["examples"][0])
        self.assertEqual([image.shape for image in request["examples"][0]["image"]], [(224, 224, 3)] * 3)

        model.raw_actions = None
        client.predict_action.return_value = {
            "data": {"actions": np.full((1, 32, 14), np.nan)}
        }
        with self.assertRaisesRegex(RuntimeError, "finite=False"):
            model.step(
                {
                    "lang": "unseen instruction",
                    "image": [np.zeros((10, 20, 3), dtype=np.uint8) for _ in range(3)],
                    "state": np.zeros(14),
                }
            )

    def test_remote_manifest_supplies_eight_fixed_slots_without_checkpoint_file(self):
        repo_root = Path(__file__).resolve().parents[1]
        launcher = (
            repo_root
            / "examples/simBenchmarks/Robotwin/eval_files/start_eval.sh"
        )
        listeners = []
        threads = []
        try:
            ports = []
            for _ in range(8):
                listener = socket.socket()
                listener.bind(("127.0.0.1", 0))
                listener.listen(1)
                ports.append(listener.getsockname()[1])
                listeners.append(listener)
                thread = threading.Thread(target=lambda sock=listener: sock.accept()[0].close())
                thread.start()
                threads.append(thread)

            with tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                robotwin = root / "RoboTwin"
                (robotwin / "script").mkdir(parents=True)
                capture = root / "captured_args.jsonl"
                (robotwin / "script/eval_policy.py").write_text(
                    "import json, os, sys\n"
                    "with open(os.environ['CAPTURE_PATH'], 'a', encoding='utf-8') as f:\n"
                    "    f.write(json.dumps(sys.argv) + '\\n')\n",
                    encoding="utf-8",
                )
                manifest = root / "server_manifest.json"
                manifest.write_text(
                    json.dumps(
                        {
                            "backend": "starvla",
                            "evaluation_instruction_type": "unseen",
                            "evaluation": {"episodes_per_task_phase": 20},
                            "checkpoint": {"id": "steps_10000"},
                            "slots": [
                                {
                                    "slot": index,
                                    "client_host": "127.0.0.1",
                                    "client_port": port,
                                }
                                for index, port in enumerate(ports)
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

                result = subprocess.run(
                    [
                        "bash",
                        str(launcher),
                        "--mode",
                        "demo_clean",
                        "--name",
                        "remote-test",
                        "--remote-manifest",
                        str(manifest),
                        *[f"task_{index}" for index in range(8)],
                    ],
                    cwd=repo_root,
                    env={
                        **os.environ,
                        "ROBOTWIN_PATH": str(robotwin),
                        "ROBOTWIN_PYTHON": sys.executable,
                        "ROBOTWIN_LOG_ROOT": str(root / "logs"),
                        "ROBOTWIN_SERVER_TIMEOUT": "4",
                        "CAPTURE_PATH": str(capture),
                    },
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
                captured_text = capture.read_text(encoding="utf-8")

            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            self.assertIn("ckpt=steps_10000", result.stdout)
            for port in ports:
                self.assertIn(f"port={port}", result.stdout)
            self.assertIn("slots=8", result.stdout)
            captured_args = [json.loads(line) for line in captured_text.splitlines()]
            self.assertEqual(len(captured_args), 8)
            for args in captured_args:
                self.assertEqual(args[args.index("--instruction_type") + 1], "unseen")
                self.assertEqual(args[args.index("--eval_num_episodes") + 1], "20")
                self.assertEqual(args[args.index("--ckpt_setting") + 1], "steps_10000")
                self.assertNotIn("--policy_ckpt_path", args)
        finally:
            for listener in listeners:
                listener.close()
            for thread in threads:
                thread.join(timeout=2)

    def test_log_aggregation_and_missing_cell_recovery(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            task = "adjust_bottle"
            phase_root = root / "clean"
            cell = phase_root / "results" / task / "demo_clean"
            cell.mkdir(parents=True)
            (cell / "_result_clean.txt").write_text(
                "Timestamp: now\n\nInstruction Type: unseen\n\n0.5", encoding="utf-8"
            )
            for index in range(20):
                (cell / f"episode{index}_clean_success-{str(index % 2 == 0).lower()}.mp4").write_bytes(b"video")
            logs = phase_root / "logs"
            logs.mkdir()
            lines = []
            successes = 0
            for index in range(20):
                lines.append(
                    '[ROBOTWIN_EPISODE] {"instruction": "prompt %d", "instruction_split": "unseen"}' % index
                )
                lines.append(
                    '[ROBOTWIN_ACTION] {"action_chunk_size": 32, "finite": true, "shape": [32, 14]}'
                )
                successes += index % 2 == 0
                lines.append(f"Success rate: {successes}/{index + 1} => 0%, current seed: {100000 + index * 2}")
            (logs / f"{task}_demo_clean_slot3_gpu3_port5697_eval.log").write_text(
                "\n".join(lines), encoding="utf-8"
            )

            rows = aggregate_eval.parse_cell(root, "clean", task, 20)
            self.assertEqual(len(rows), 20)
            self.assertEqual(rows[0]["worker"], {"slot": 3, "gpu": "3", "port": 5697})
            self.assertEqual(rows[1]["rejected_seeds"], [100001])
            self.assertFalse(rows[1]["success"])

            output = []
            with mock.patch("builtins.print", side_effect=output.append):
                aggregate_eval.missing(SimpleNamespace(run_root=str(root), phase="clean", episodes=20))
            self.assertNotIn(task, output)
            self.assertEqual(len(output), 49)


if __name__ == "__main__":
    unittest.main()
