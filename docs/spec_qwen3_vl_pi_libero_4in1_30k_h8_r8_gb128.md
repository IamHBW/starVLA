# Qwen3-VL-4B π：LIBERO 官方三参数对齐

这是唯一任务书：从头训练、完整评测 h8/r8/GB128 的 Qwen3-VL-4B π。
中断先读 `PROGRESS.md`、`BLOCKED.md`，核对 JOBID 和产物再续跑。
这活为什么干：旧 h32/r24/GB256 得分 89.75%，不能直接对比 StarVLA Table 2 的 95.7%；本次只消除三个混杂变量。
可比、可追溯高于速度，不按成功率反调参；事实源为论文（https://arxiv.org/pdf/2604.05014）、运行时配置和产物。

## 我替领导拍的板

- 基座 `ebb281ec70b05090aa6165b016eac8ec08e71b17`，随机 action head，seed=42，30K steps，禁止续训。
- 仅改 horizon `32→8`、replan `24→8`、GB `256→128`。冻结 LR `1e-5/1e-4`（Qwen/action）、diffusion repeat=8、inference=4、双路 224 图像、无 proprio、全参、SDPA、非 canonical π。
- 用户授权使用 2 节点×8 GPU、per-device=8、GA=1，即 `16×8×1=128`；GB 保持不变。
- 只用 `libero_all`，不加 VLM 数据/增强/状态。共享数据属用户 `tianyu`，manifest 记 `/mnt/data/public_data/libero` 及 SHA：Spatial `bf14d6258218d12c2e3c1a3b9922e163cdf6455d`、Object `15657dac2ad1c01b4e94bf54ab0493b46a8d63f9`、Goal `222cf888ed360fad0a5f983748c1cc40743d43e7`、Long `e1a223d30b896c1613f270a2bfc63d382b3de7e1`。
- 这是三变量受控复现，不宣称严格复现 Table 2；论文值无验收门槛。

## 界限

- 白名单：新增 train 入口、最小参数化现有 QwenPI eval、对应 tests、进度文件。复用 YAML+CLI 覆盖，禁复制配置。
- 仅写 `/mnt/data/users/bowen/workspace/ckpt/qwen3_pi_libero4in1_30k_h8_r8_gb128` 及独立 eval；旧 h32 全部只读。
- 不改结构、优化器、LR、数据/仿真/依赖，不提交已有改动。W&B 只用 `/mnt/data/users/bowen/workspace/tokens.sh`，online。

## 现状与任务 0

旧 30K：四套=`90.8/91.2/93.8/83.2`，89.75%；Plus 54.3769%。Table 2=`98.8/99.6/95.8/88.4`，95.7%，chunk=8、8×A100×16=GB128。

记录 Git HEAD/status、GPU、数据 SHA、旧产物校验。实测目标 pytest：`8 passed`。

## 任务 1：改造与门禁

新增入口，覆盖 horizon=8、mix=`libero_all`、batch、GA。参数化 eval：旧入口 h32/r24，新入口 h8/r8；gate 校验 horizon、mix、模型、基座。

测试覆盖新旧；horizon=32、h32 mix、replan≠8 或 GB≠128 必须被新门禁拒绝。跑目标 pytest、`bash -n`，不得削弱断言。

## 任务 2：训练

预检断言 world=16、device=8、GA=1、GB=128、action `[B,8,7]`、future=7、30K、非 resume。提交双节点 16 卡 HTrain；记录 JOBID、命令、代码 SHA/diff、解析配置、数据 SHA、W&B URL。确认 loss 有限、step 递增且无数据/NCCL 错误，勿重复提交。

30K 后验证可加载、step=30000、无 NaN/Inf；记录真实 epoch（预期约 9.54），禁凑数。

## 任务 3：全量评测

同一 checkpoint 固定 h8/r8。标准 revision `8f1084e3132a39270c3a13ebe37270a43ece2a01`：4×10×50=2000；Plus revision `4c83d77c807983abf01da2c23bc8d2f72a204912`：10030。不得挑 seed、跳失败项或小样本外推。

保存逐 task 计数、日志、manifest；聚合拒绝重复/缺失/越界。手算复核后与旧结果、Table 2 比较，注明未控制变量。

## 规矩

- 每阶段更新 `PROGRESS.md`；阻塞写 `BLOCKED.md`：命令、错误、已试方案、下一步。仅启动日志/W&B 图不算产物。
- 不碰无关任务或旧产物。越界/改参数即停并记录。训练前只做静态门禁，不产出缩短训练的 checkpoint。

## 完成条件

- GB128 从头完成 30K；checkpoint、manifest、JOBID/W&B/日志齐全，配置证明 h8 和冻结项。
- 标准 2000、Plus 10030 全量完成；原始计数=聚合，反向门禁及测试通过。
- 进度文件给出新旧/Table 2 逐套表、绝对差、硬件/数据溯源、未控制变量；无已知错误。无分数门槛，真实结果即验收。
