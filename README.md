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

## 导出 LeRobot

Insight MCAP 在完成穷尽审核后可导出为 LeRobotDataset v3.0。导出器要求 ROS 2 环境提供
`rosbag2_py` 和 MCAP 存储插件；LeRobot 放在独立环境中安装，当前 Python 3.10 兼容的验证
版本固定为 0.4.4：

```bash
.venv/bin/pip install -r configs/lerobot-validator-py310.txt
.venv/bin/pip install -e . --no-deps
.venv/bin/rda export-lerobot \
  --source recording/ \
  --review-manifest recording/review/manifest.json \
  --annotation-manifest recording/review/annotation_manifest.json \
  --decisions recording/review/decisions.json \
  --task-spec .rda/cup-stacking.json \
  --output outputs/recording-segmented \
  --repo-id local/recording-segmented
```

输出包含左右手和头部三路独立视频、18 维双手位姿观测、9 维头部位姿观测、有效性掩码、
源帧索引和原子动作索引。`action` 是下一帧双手位姿目标；episode 最后一帧保持当前目标。
它仍处于源追踪坐标系，未经过机器人重定向且不含夹爪命令，导出清单会显式保留这一限制。
导出使用同目录临时产物完成后原子改名，拒绝覆盖已有数据集。

在固定 LeRobot 环境中完成内部不变量检查、全部视频逐帧解码、官方 loader 索引和一个
DataLoader batch：

```bash
.venv/bin/rda validate-lerobot --dataset outputs/recording-segmented
```

通过后会写入 `rda/internal_validation.json` 与 `rda/official_validation.json`；前者还包含
Parquet/视频哈希和每个视频键的实际解码帧数。

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
