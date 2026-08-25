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
- `task_specs/` 定义稳定 task ID、原子动作、最短帧数和可选插件入口。
- 自动结果只是人工审核提示；除非另有批准，所有源片段和必需视角都要穷尽审核。
- 二维码标定上下文必须用独立 context 边界保留；全局变换写入 JSON，不得静默改写源 pose。
- LeRobot 工作数据集保留 `rda/` 验证证据；严格交付副本只复制 schema 实际引用的官方
  数据目录。仅在确认根目录 `images/` 不含文件时，才可省略官方写入器遗留的空目录骨架。

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
