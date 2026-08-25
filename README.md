# Robot Dataset Annotator

一个与采集系统解耦的机器人数据分段、自动标注、人工审核和批次验收框架。核心库不假设
rosbag、相机名称、任务类型、工作目录或导出格式；这些差异由任务插件、数据源适配器和会话
配置提供。

当前内置 `cup-pick-place` 和 `cup-stacking` 两个任务。前者采用人工穷尽审核，保留二维码
上下文并标注四阶段任务边界；不再包含旧夹杯自动分割算法。后者把三杯速叠循环定义为“搭塔”
和“收拢”两阶段，并能从同步双手位姿提示同一源片段中的多个循环。新任务可以复用批次状态机、
多人审核 schema、穷尽审核门和验收报告，不需要复制已有任务代码。

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

带有候选插件的任务可从规范化低维状态生成候选边界，例如：

```bash
.venv/bin/rda suggest --task-spec .rda/cup-stacking.json \
  --observations review/manifest.json --format insight-review
```

`insight-review` 适配器从左右手同步位姿构造规范化状态；review manifest 不包含夹爪宽度，
因此对应列保持无效。候选边界始终需要视觉确认，多循环结果位于 `episodes` 数组。
NPZ 输入包含 `state` 和 `state_valid`；Insight review parquet 可改用
`--format insight-parquet`。

`cup-pick-place` 没有自动建议插件，必须人工审核三路视频。其 decisions 可以为 episode
增加 `context_start_frame`，它早于
`episode_start_frame`，用于保留头部相机仍能看到二维码的标定上下文。上下文帧不会归入四个
操作阶段；导出时标记为 `atomic_action_index=-1` 和 `qr_localization_context`。四阶段边界均
由审核者在原始 source frame 坐标下填写，不能删除二维码前缀。

```json
{
  "context_start_frame": 0,
  "episode_start_frame": 120,
  "episode_end_frame_exclusive": 920,
  "atomic_boundaries": [120, 260, 520, 760, 920]
}
```

已知二维码黑色方形区域的实际边长后，可用保留的头部帧、相机内参、头部全局 pose 和 MCAP
中的静态外参估计二维码在全局坐标系下的变换：

```bash
.venv/bin/rda calibrate-qr \
  --source recording/ \
  --review-manifest recording/review/manifest.json \
  --marker-size-m 0.10 \
  --frame-start 0 --frame-end-exclusive 120 \
  --output recording/review/qr_transform.json
```

若桌面标记实际为 ArUco，可显式指定类型、字典和 ID，例如
`--marker-type aruco --aruco-dictionary DICT_4X4_50 --aruco-marker-id 4`。普通 QR 仍是
默认检测方式。两种方式都使用打印图案黑色方块的实测边长，并在标定 JSON 中记录检测器
配置。

输出 JSON 保存 `global_from_qr`、`qr_from_global`、参与估计的源帧、重投影误差、坐标系名称、
二维码尺寸和输入哈希。工具不修改三路相机 pose，也不自动把训练数据转换到二维码坐标系；
后续使用者可按 JSON 中的矩阵自行变换。二维码坐标原点位于图案中心，X 指向图案右侧，Y
指向图案上方，Z 从印刷面向外。

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

输出包含左右手和头部三路独立视频、18 维双手位姿观测、9 维头部 tracking pose、应用
MCAP 静态外参后的 9 维头部 RGB 相机全局 pose、有效性掩码、源帧索引和原子动作索引。
`action` 是下一帧双手位姿目标；episode 最后一帧保持当前目标。
它仍处于源追踪坐标系，未经过机器人重定向且不含夹爪命令，导出清单会显式保留这一限制。
每帧还包含左右手独立 subtask ID、左右手 subtask 内进度和整体操作进度；ID 到英文语义的
映射写入 `rda/export_manifest.json`。二维码上下文的整体操作进度固定为 0，正式任务区间从
0 线性增长到 1。
头部 pose 默认按相机名推断对应的 `<name>_camera_imu` tracking frame；设备命名不符合该规则
时，导出和二维码标定都应显式传入 `--head-pose-child-frame`。
导出使用同目录临时产物完成后原子改名，拒绝覆盖已有数据集。
默认使用流式视频编码：相机帧从 MCAP 解码后直接送入三路编码器，避免先写临时 PNG 再
读回的磁盘往返，并在 `rda/export_manifest.json` 中记录 `video_encoding_mode`。若需要
诊断编码器兼容问题，可显式传入 `--video-encoding-mode staged-png` 恢复分阶段编码。

在固定 LeRobot 环境中完成内部不变量检查、全部视频逐帧解码、官方 loader 索引和一个
DataLoader batch：

```bash
.venv/bin/rda validate-lerobot --dataset outputs/recording-segmented
```

通过后会写入 `rda/internal_validation.json` 与 `rda/official_validation.json`；前者还包含
Parquet/视频哈希和每个视频键的实际解码帧数。

验证完成后要区分“工作数据集”和“严格交付包”。工作数据集保留 `rda/`，其中是导出清单、
哈希和验证证据；该目录是 RDA 扩展，不属于 LeRobot v3 schema。官方写入器把逐帧图像编码成
视频后，可能留下空的 `images/observation.images.*` 临时目录；必须先确认 `images/` 中没有
任何文件，才能把这些空目录视为可清理产物。当前导出的图像特征均为 `dtype: video`，
`meta/info.json` 通过 `video_path` 指向 `videos/` 下的 MP4，并不依赖空的根目录 `images/`。

需要只含本任务官方数据文件的交付物时，应从已验证的工作数据集另外复制 `data/`、`meta/`、
`videos/` 三个目录，并把 `rda/` 作为旁路证据与交付包并列保存；不要直接修改已经验证通过的
工作数据集，也不能在 `images/` 仍有文件时删除它。

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
