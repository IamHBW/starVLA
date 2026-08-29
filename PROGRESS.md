# Qwen3-VL QwenPI LIBERO 4-in-1 30K 收尾

更新时间：2026-08-29T09:55:00Z

## 结论

- h8/r8/GB128 任务 `q3pi-h8g128-n2`（JOBID `94d09980-72fa-42c7-8dd4-cef0d089ea0b`）已于 2026-08-28 成功结束；HTrain 状态为 `Succeeded`。
- 训练从 step 0 完成 30,000 optimizer steps、3,840,000 samples（约 9.46 epochs），最终 `action_dit_loss=0.00479147`。
- 30K checkpoint 严格加载及有限 action 门禁通过；SHA256 为 `c6a1b15c4ecff5a589c9f2fa8569113c0635750c33324fc7b3839bd718b39492`。
- Standard 2,000 与 LIBERO-Plus 10,030 均完整完成，`infra_retries=0`。

| 运行 | Spatial | Object | Goal | Long | Standard | Plus |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| h8/r8/GB128 30K | 98.2% | 99.4% | 99.2% | 96.4% | 98.30% | 77.84% |
| 旧 h32/r24/GB256 30K | 90.8% | 91.2% | 93.8% | 83.2% | 89.75% | 54.38% |
| 论文对照 | 98.8% | 99.6% | 95.8% | 88.4% | 95.7% | 77.0% |

Standard 原始计数为 1,966/2,000；Plus 原始计数为 7,807/10,030。论文值只作上下文，不是验收门槛。

## 产物与溯源

- 运行目录：`/mnt/data/users/bowen/workspace/ckpt/qwen3_pi_libero4in1_30k_h8_r8_gb128`
- 汇总：`eval/slots_32/summary.json`；完整报告：`eval/slots_32/report.md`
- checkpoint 门禁：`checkpoint_validation.json`；训练/数据/环境溯源：`run_manifest.json`
- 基座 revision：`ebb281ec70b05090aa6165b016eac8ec08e71b17`，owner `bowen:ai_researchers`
- 共享原始数据：`/mnt/data/public_data/libero`，owner `tianyu:ai_researchers`；四套固定 revision 已记录在 manifest。
- 训练 Python realpath：`/mnt/data/public_tools/miniconda3/envs/fastwam/bin/python3.10`，owner `tianyu:ai_researchers`；manifest 同时保留了完整 `sys.path`，便于后续控制变量。

## 历史 h32/r24/GB256 任务

旧任务 `q3pi-lib30k-h32d`（JOBID `44cb5608-d1dd-43b7-89f7-1570b0ee4083`）已 `Suspended`，最后记录到 optimizer step 8,370，未生成 10K checkpoint。旧目录 `/mnt/data/users/bowen/workspace/ckpt/qwen3_pi_libero4in1_30k` 保持只读。

## 验证

- 目标测试：11 passed。
- 四个训练/评测 shell：`bash -n` 通过。
- WebSocket client 兼容性已分别在 websockets 13.1 与 16.0 环境验证。
- `git diff --check` 通过。
