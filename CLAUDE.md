# Repository agent instructions

本文件是本仓库自动化开发的约束。`AGENTS.md` 与 `CLAUDE.md` 必须保持内容一致并提交。

## 开始前

```bash
git status --short --branch
git log origin/main..HEAD --oneline
```

- 不假设工作区、录制、审核、数据集、task spec 或容器的固定路径。
- 用户未提供路径时先发现候选，再要求用户确认；路径只写入用户 session。
- `outputs/`、`reviews/`、`datasets/`、`.rda/` 和原始录制均为本地产物，不提交。
- `tests/` 是本地验证资产：必须保留并运行，但禁止加入 Git、上传 GitHub 或打入安装包。

## 分层边界

- `core/` 只包含格式无关的决策、审计、状态机、插件加载和命令编排，不导入 ROS、
  OpenCV、PyArrow、LeRobot 或采集产品代码。
- `tasks/` 只包含任务语义和候选边界算法。无法可靠判断时返回 `unresolved`，不能伪造
  边界或自动丢弃源数据。
- `adapters/` 隔离 rosbag、Insight、LeRobot 等外部格式与可选依赖。
- Insight 源读取必须让 rosbag2 根据 bag metadata 自动选择 MCAP 或 SQLite3 存储插件，不得
  硬编码单一存储格式。
- `task_specs/` 定义稳定 task ID、原子动作、最短帧数和可选插件入口。
- 自动结果只是人工审核提示；除非另有批准，所有源片段和必需视角都要穷尽审核。
- 二维码标定上下文必须在源 review 中独立保留；全局变换写入 JSON，不得静默改写源 pose。
  变换落盘后，最终训练 decisions 可按任务要求省略该 context。
- 左右手训练子语义必须独立审核；短 pose 尖峰、持续坐标跳变及其稳定残余跳变只能写入保留
  原始值与审计证据的新 manifest，模糊的首次跳变必须返回 `NEEDS_REVIEW`。
- `screw-nut-sorting` 必须逐 episode 对比双手 native VIO 与 Insight Global；只允许同帧全局
  抵消的漂移通过，延迟或缺失修正必须复核，且不得把前一 episode 的结论传播到后续 episode。
- 修复 pose 的逐 episode 审计必须用保留的 raw 数组复现源 bag、用 corrected 数组判断漂移；
  按 decisions 限定修复门禁时须绑定 decisions 哈希和精确选区，并保留选区外未解决事件。
- 多录制 LeRobot 合并必须在同一官方 writer 中完成并全局重编号 episode；每帧保留来源录制
  索引和源帧，逐录制绑定 decisions、pose 审计、同步、外参和输入哈希，不得事后拼接目录。
- 夹爪 marker 短缺失只可在同一 episode 内由前后可靠值插值最多 3 帧；不得跨 episode 或填充
  无边界/长缺失，且 state/action 必须保留 direct、单 marker 推算、插值或 invalid 来源代码。
- LeRobot 工作数据集保留 `rda/` 验证证据；严格交付副本只复制 schema 实际引用的官方
  数据目录，以及任务明确要求的不可变标定 sidecar（例如 `qr_transform.json`）。仅在确认根目录
  `images/` 不含文件时，才可省略官方写入器遗留的空目录骨架。

## 修改与验证

- 新任务必须提供 task spec、插件、正常/失败/缺失数据/模糊边界/多 episode 测试。
- 修改决策 schema 或状态机时保持旧生产 decisions 的兼容读取，并增加迁移测试。
- 运行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest -q \
  -o cache_dir=/tmp/robot-dataset-annotator-pytest
skill_validator="$(find "${CODEX_HOME:-$HOME/.codex}/skills" \
  -path '*/skill-creator/scripts/quick_validate.py' -print -quit)"
test -n "$skill_validator"
python3 "$skill_validator" skills/process-robot-datasets
```

- Skill 和源码中不得出现设备绝对路径。检查：

```bash
rg -n '/home/|/workspaces/' skills src configs docs README.md
```

- 代码、注释和 Git 提交使用英文；对话和用户文档使用中文。

## 文档、提交与上传

- 代码行为、目录、命令、schema、任务扩展方式或 Skill 改变时，同步更新 README、相关
  `docs/`、`SKILL.md`、`AGENTS.md` 和 `CLAUDE.md`；禁止只改代码不改文档。
- 每次代码修改完成并验证后，默认创建英文 conventional commit，并推送当前分支到
  `origin`。只有用户明确要求保留本地或暂不上传时才停止。
- 推送前运行 `git diff --check`，只暂存本次范围内的文件；不得夹带生成数据或未知提交。
- 推送前确认 `git ls-files tests` 为空，并检查 wheel/sdist 文件列表不包含 `tests/` 或
  `test_*.py`。
- 推送失败时保留本地提交，明确报告认证、网络或远端权限问题。
