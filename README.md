# Robot Dataset Annotator

一个与采集系统解耦的机器人数据分段、自动标注、人工审核和批次验收框架。核心库不假设
rosbag、相机名称、任务类型、工作目录或导出格式；这些差异由任务插件、数据源适配器和会话
配置提供。

当前内置 `cup-pick-place` 和 `cup-stacking` 两个任务。前者保留 Insight rosbag 生产中验证
过的四阶段边界推断方法；后者把三杯速叠循环定义为“搭塔”和“收拢”两阶段，并能从同步
双手位姿提示同一源片段中的多个循环。新任务可以复用批次状态机、多人审核 schema、穷尽
审核门和验收报告，不需要复制已有任务代码。

## 快速开始

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/rda copy-task cup-pick-place --output .rda/cup-pick-place.json
.venv/bin/rda configure --output .rda/session.json
.venv/bin/rda audit --config .rda/session.json
.venv/bin/rda resume --config .rda/session.json
```

`configure` 不提供机器相关的默认路径；缺少参数时会询问工作区、输入、审核、数据集和任务
配置路径。可迁移的 Codex Skill 位于 `skills/process-robot-datasets/`。

安装 Skill：

```bash
skill_root="${CODEX_HOME:-$HOME/.codex}/skills"
mkdir -p "$skill_root"
cp -a skills/process-robot-datasets "$skill_root/"
```

Skill 不保存项目路径；每次新环境首次执行时会要求确认路径并生成 session。
在 session 的 `commands` 中以 argv 数组配置数据源、导出器和验证器。`resume` 默认只预览，
明确增加 `--execute` 才执行下一条转换；支持 `{input}`、`{review_dir}`、`{decisions}`、
`{dataset_dir}`、`{task_spec}` 等路径占位符，不经过宿主 shell。

规范化的低维状态可通过任务插件生成候选边界：

```bash
.venv/bin/rda suggest --task-spec .rda/cup-pick-place.json \
  --observations segment.npz --format npz
```

NPZ 包含 `state` 和 `state_valid`；Insight review parquet 可改用
`--format insight-parquet`。已有同步审核 manifest 可直接用于叠杯任务：

```bash
.venv/bin/rda suggest --task-spec .rda/cup-stacking.json \
  --observations review/manifest.json --format insight-review
```

`insight-review` 适配器从左右手同步位姿构造规范化状态；review manifest 不包含夹爪宽度，
因此对应列保持无效。候选边界始终需要视觉确认，多循环结果位于 `episodes` 数组。

## 分层

```text
src/robot_dataset_annotator/
  core/          决策校验、产物驱动状态机、原子文件写入
  tasks/         任务插件；杯子只是其中一个任务
  adapters/      rosbag、LeRobot 等外部格式边界
  task_specs/    可版本化的任务语义和质量门配置
  cli.py         稳定命令入口
skills/          可随仓库复制或安装的标准 Codex Skill
```

架构约束和新增任务流程见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 与
[docs/ADDING_TASK.md](docs/ADDING_TASK.md)。

仓库自动化开发约束见 [AGENTS.md](AGENTS.md)；`CLAUDE.md` 与其同步并一同提交。代码行为、
命令、schema 或 Skill 发生变化时，必须同步文档、运行测试、提交并推送当前分支。
本地 `tests/` 用于开发验证，但不会提交到 GitHub，也不会进入 wheel 或 sdist。
