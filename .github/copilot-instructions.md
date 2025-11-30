## 快速目标

为 AI 代码代理（Copilot/自动化机器人）提供尽快在本仓库中上手并给出高质量修改的可执行指令：关注架构要点、常用开发/测试流程、项目特有约定、以及修改时必须检查的位置。

## 1) 大局观（重要模块与边界）
- `verl/`：核心库，实现 HybridFlow 编程模型（actor/rollout/critic/奖励等）。修改模型训练逻辑、调度或数据流时首选此目录。
- `examples/`, `recipe/`：具体算法/训练示例（例如 `examples/ppo_trainer/`、`recipe/dapo/`），用来理解如何配置和运行真实训练任务。
- `docs/`：设计与使用说明（架构说明、性能优化、后端集成细节）。当不确定为何这样设计，优先查看对应 docs 页面。
- `docker/` 和 `scripts/`：运行/部署、依赖安装及 CI helper 脚本（常包含不可见的运行约定）。
- 根目录文件：`pyproject.toml`、`setup.py`、多个 `requirements*.txt`（可见不同目标环境），跟依赖/打包相关的更改要小心。

为什么这些重要：项目明显将训练（FSDP/Megatron/DeepSpeed）与生成（vLLM/SGLang/HF）解耦，配置通过 Hydra-esque 的层级 cfg 管理（见 examples 与 docs），因此对配置和策略字段（例如 `actor_rollout_ref.*.strategy`）的更改会跨多处影响运行时行为。

## 2) 关键开发/构建/测试工作流（可直接执行的命令）
- 快速开发（Python-only）:
  - pip 安装开发依赖：`pip install -e .[test,vllm]` 或 `pip install -e .[test,sglang]`（见 `CONTRIBUTING.md`）
  - 启用 pre-commit：`pip install pre-commit && pre-commit install`，常用 hooks: `ruff`, `autogen-trainer-cfg`。
- 文档本地构建：
  - `pip install -e .[test] && pip install -r requirements-docs.txt`
  - `make -C docs clean html`，本地预览：`python -m http.server -d docs/_build/html/`。
- 查 CI 流程：`.github/workflows/` 下有 `gpu_unit_tests.yml`、`cpu_unit_tests.yml`、`vllm.yml`、`sgl.yml` ——新增测试请参考这些 workflow 的 paths 与最小化测试脚本。

## 3) 项目特有约定与模式（不要盲目替换）
- 配置中心化：大量行为通过 runtime 配置控制（hydra 风格），例如将 FSDP2 打开是通过 `*.strategy=fsdp2`。修改默认配置前，搜索 repo 中的 `*.yaml`/`cfg` 示例并更新 `examples/`。
- 自动生成/校验：仓库使用 pre-commit hook `autogen-trainer-cfg` 来生成训练配置；如果改动涉及训练 config，务必运行该 hook 或在 PR 中解释为何跳过。
- 多后端兼容：vLLM、SGLang、Megatron、FSDP/DeepSpeed 都有专门集成点（`verl/workers/*`、`verl/backends/*`）。当改动触及接口（比如 rollout 接口），请同时检查这些后端实现。
- 本仓库包含大二进制/轮子（如 root 下的 `flash_attn-*.whl`），暗示某些 CI/本地依赖可能不是通过 pip 公共索引安装；不要假设环境能自动解析这些依赖。

## 4) 编辑/提 PR 时的具体检查清单
- 变更范围：改动训练/rollout/actor/critic/奖励逻辑前，列出受影响的后端（vLLM/SGLang/Megatron）并在 PR 描述中注明测试平台。
- 运行 lint & hooks：`pre-commit run --all-files` 并修复 `ruff` 报错，若涉及 cfg，运行 `autogen-trainer-cfg`。
- 更新文档：任何用户可见行为或配置变化必须在 `docs/` 或 `README.md` 中更新并给出示例命令（例如：如何设置 `fsdp2`）。
- 小型单元验证：尽可能添加/修改 tests 下的最小单元（见 `tests/`），并在 PR 中指定需要的 CI workflow（gpu/cpu/vllm/sgl）。

## 5) 便利定位示例（供 agent 打点查阅）
- 训练调度与 Ray 示例：`verl/trainer/ppo/ray_trainer.py`（当前打开文件），inspect actor-rollout glue 逻辑。
- PPO 示例运行脚本：`examples/ppo_trainer/run_qwen2-7b_seq_balance.sh`（了解实际命令行参数）。
- 集成脚本：`scripts/install_vllm_sglang_mcore.sh`（查看安装步骤与依赖顺序）。
- 性能/后端文档：`docs/perf/` 与 `docs/advance/` 下的页面，常包含调优开关（例如序列打包、Liger kernel）。

## 6) 和 AI 代理相关的行为准则（对生成代码的硬性要求）
- 行为最小化：优先做最小可行改动；若需重构，拆成多个 PR，每个 PR 包含编译/测试通过的单一目的变更。
- 不要移除/硬编码后端兼容性开关（例如删除 `strategy` 支持），除非确有替代实现并包含迁移说明。
- 在修改大型配置或训练逻辑前，先在 PR 描述中列出回归测试计划与所需 CI（例如要跑 GPU 测试请在 PR 中声明）。

## 结束语与反馈
如果这里有不准确或不完整的部分，请指出你希望 agent 强化的区域（例如：更详细的工程路径、常见调试命令、或特定后端的接口说明），我会基于仓库内容迭代更新该文件。
