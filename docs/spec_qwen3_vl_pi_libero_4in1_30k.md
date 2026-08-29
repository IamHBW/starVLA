# Qwen3-VL QwenPI LIBERO 4-in-1 30K h32/r24/GB256

本文档是唯一任务来源；断线读 `PROGRESS.md` 续做，疑点写 `BLOCKED.md` 后继续可做项。
目标：从固定 Qwen3-VL 底模和随机 PI head 训练 30K QwenPI LIBERO-4in1，再完整评测 LIBERO/Plus。优先级：可信 > 对版 > 覆盖 > 速度。

## 门禁与契约

- 已确认：GB 256、30K optimizer steps；在 `32×8×1` 与 `16×8×2` 真实 OOM 后，用户授权使用 `8×8×4=256`。
- 用户覆盖原建议并固定 `action_horizon=32`、`replan_steps=24`。launcher 断言 per-device×world size×accumulation=256，禁止静默改契约。
- 从头训练：用户自有 `Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17` + seed 42 随机 LayerwiseFM head，全参训练；首次 `is_resume=false`、无 `pretrained_checkpoint`，禁用发布权重。
- 训练/评测均 `use_canonical_forward=false`；7D action、horizon 32、`repeated_diffusion_steps=8`、推理积分 4 steps，每个 chunk 执行 24 步后重规划。
- `libero_all_h32` 四套联合训练、registry 权重各 1.0；两路 RGB 224×224，primary/agentview 在前、wrist 在后，无 state。验证 action 为 `[Δx,Δy,Δz,Δroll,Δpitch,Δyaw,gripper]`；`action_type=delta_qpos` 只是旧标签，normalization/gripper 以 loader、统计和 worker 为准。
- 标准评测：Spatial/Object/Goal/Long 各 10 tasks×50 valid trials=2,000 episodes，不含 `libero_90`。Plus 只评测：10,030 个唯一 scenarios（2,402/2,518/2,591/2,519），覆盖 Camera/Robot/Language/Light/Background/Noise/Layout 七类扰动。
- 官方 30K LIBERO 95.7%、Plus 77.0%仅作对照；官方 batch 128、本实验 256，不设门槛或声称严格复现。

## 边界

- 只改/增 `examples/simBenchmarks/LIBERO/train_files/qwen3_pi_4in1_30k.*`、`examples/simBenchmarks/LIBERO*/eval_files/` 必需文件、`tests/test_libero*.py`、`PROGRESS.md`、`BLOCKED.md`。产物只写 `/mnt/data/users/bowen/workspace/{data/starvla_libero_lerobot,ckpt/qwen3_pi_libero4in1_30k}/`；其余只读。
- 不改任务/init states/成功条件/rollout 数/seed/max steps/聚合口径，不碰 RoboTwin 改动及标准 LIBERO checkout 用户文件；Plus 不用于训练、统计、调参或选模。
- 版本：StarVLA=`02861ead680ea648c367ed41cf0d0976581f0467`；LIBERO=`8f1084e3132a39270c3a13ebe37270a43ece2a01`（有用户文件）；Plus=`4c83d77c807983abf01da2c23bc8d2f72a204912`。共享环境属 `wanqi/heyang/tianyu`，使用前告知并记 realpath/owner/版本。W&B 只 source `/mnt/data/users/bowen/workspace/tokens.sh`，online。

## 任务 0：静态预检

门禁前仅保存仓库 HEAD/status/diff、GPU/盘、HTrain usableProjects/队列、环境与 resolved config。四数据 repo 为 `IPEC-COMMUNITY/libero_{spatial,object,goal,10}_no_noops_1.0.0_lerobot`，revision 依次为 `bf14d6258218d12c2e3c1a3b9922e163cdf6455d`、`15657dac2ad1c01b4e94bf54ab0493b46a8d63f9`、`222cf888ed360fad0a5f983748c1cc40743d43e7`、`e1a223d30b896c1613f270a2bfc63d382b3de7e1`。固定 SHA 下载，核对 owner/hash/info，复制并哈希 `modality.json`；总计 1,693 episodes、273,465 frames、1,881,101,992 bytes，否则停。

确认后不跑独立训练 smoke，直接提交完整 30K，在 live log 验证真实 forward/loss、batch 公式和 step 推进。静态核对相机/action/state/统计；`train_starvla.py` 的 `strict_contract/validate_checkpoint` 是 Clean50 硬编码，禁用。

## 任务 1：训练 30K

复用 `train_starvla.py`、Accelerate/ZeRO2、现有 launcher，仅新增专用 YAML/launcher。固定上述契约、bf16、全参、LR=`3e-5/1e-5/1e-4`、AdamW=`0.9/0.95`、warmup 5K、cosine min=`1e-6`、save 10K、`full_state_checkpoints=true`、W&B online。直接提交完整 30K；保存 resolved config/provenance/命令/diff/W&B/metrics、10/20/30K weights 和最新 full state。

按 `alaya-htrain` 动态选 project；任务名≤20字符且不重名，用绝对 launcher 提交 1×8 GPU。最终授权配置必须为 `8×8×4=256`；仅 `Running`、首个 loss finite 且 step 推进才报启动。崩溃/配置错/OOM/NaN/卡步时保留 JOBID/日志/config；中断只恢复本 run；最终严格加载 30K 权重/统计，step、SHA256、action smoke 全通过。

## 任务 2：评测与闭合

checkpoint/config/normalization/hash 过门禁后，在同一 HTrain job 放 server 与 workers；标准/Plus 共用权重、统计、forward、4-step 推理和 horizon/replan=32/24。每 slot 须收到 finite action；动态记录 registry、顺序、init state、max steps。

标准每 suite 先 1×1，再以正式 slot 数 smoke，最后 2,000；Plus 先覆盖七类扰动 smoke，再跑 10,030 scenarios。互斥 `[start,end)` 分片、逐条原子写、单 writer，续跑只补缺 ID；RPC/加载/渲染/超时另列重试，不计模型失败。

每条含 suite、task/scenario ID、seed/init、success、checkpoint/hash、统计、推理参数、代码 revision、时间、infra error。可重算 task/suite/overall 和 Plus category，另列重试；保留 manifest、resolved configs、命令、JOBID/日志、W&B、逐条结果、失败清单、必要视频、summary 和≤2页报告。

## 验收与停止

- manifest 指向发布 100K 须红、还原转绿；删除/重复结果 ID 须红。最终标准=2,000（suite 500/task 50），Plus=10,030、七类齐、`complete=true`。
- 禁止 mock、skip/todo、吞错、放宽判断、改计数、`|| true`、借 W&B 身份、新增提交封装。契约冲突、NaN/Inf、严格加载失败或须改变四项确认值时停下问用户。
- 完成：独立 30K 权重严格加载并推理；provenance 只含固定 VLM/四数据 revision/本 run 状态；两基准无重无缺、可重算、越界改动 0。`BLOCKED.md` 空也写“无”；7 天到则交接证据、覆盖、卡点、续跑命令。
