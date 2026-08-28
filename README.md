# Robot Dataset Annotator

一个与采集系统解耦的机器人数据分段、自动标注、人工审核和批次验收框架。核心库不假设
rosbag、相机名称、任务类型、工作目录或导出格式；这些差异由任务插件、数据源适配器和会话
配置提供。

当前内置 `cup-pick-place` 和 `cup-stacking` 两个任务。前者采用人工穷尽审核，先独立保留二维码
标定上下文，再标注四阶段任务边界；二维码变换落盘后，训练导出可以不包含标定上下文。该任务
不再包含旧夹杯自动分割算法。后者把三杯速叠循环定义为“搭塔”
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

`insight-review` 适配器从左右手同步位姿构造规范化状态；review manifest 本身不包含夹爪宽度，
因此对应列保持无效。候选边界始终需要视觉确认，多循环结果位于 `episodes` 数组。
NPZ 输入包含 `state` 和 `state_valid`；Insight review parquet 可改用
`--format insight-parquet`。

`cup-pick-place` 没有自动建议插件，必须人工审核三路视频。标定阶段不能从源 review 中删除
头部相机仍能看到二维码的前缀；可用独立 context decisions 或明确帧范围生成
`qr_transform.json`。若训练集也需要这些帧，episode 可增加早于 `episode_start_frame` 的
`context_start_frame`，导出后它们标记为 `atomic_action_index=-1` 和
`qr_localization_context`。若只需要全局二维码变换，标定完成后最终训练 decisions 可以省略
`context_start_frame`，此时导出清单的 `context_frames` 必须为 0。四阶段边界仍由审核者在
原始 source frame 坐标下填写。

```json
{
  "context_start_frame": 0,
  "episode_start_frame": 120,
  "episode_end_frame_exclusive": 920,
  "atomic_boundaries": [120, 260, 520, 760, 920],
  "hand_subtask_boundaries": {
    "left_hand": [120, 260, 520, 760, 920],
    "right_hand": [120, 470, 560, 760, 920]
  }
}
```

`atomic_boundaries` 描述整体操作阶段；左右手训练语义由两个独立边界流描述，不要求与整体
阶段对齐。`cup-pick-place` 的左手阶段是接近并夹取、搬运、释放、撤离；右手阶段是投放区
等待、接取到达的杯子、引导或稳定、撤离。静止等待的右手不能标成夹取或搬运。

已知二维码黑色方形区域的实际边长后，可用保留的头部帧、相机内参、头部全局 pose 和 MCAP
中的静态外参估计二维码在全局坐标系下的变换：

```bash
.venv/bin/rda calibrate-qr \
  --source recording/ \
  --review-manifest recording/review/manifest_pose_corrected.json \
  --marker-size-m 0.06 \
  --frame-start 0 --frame-end-exclusive 120 \
  --marker-type aruco --aruco-dictionary DICT_4X4_50 --aruco-marker-id 4 \
  --output recording/review/qr_transform.json
```

这里的 UMI 桌面 ArUco 使用 `DICT_4X4_50`、ID 4，官方配置的黑色方块边长为 0.06 m。
普通 QR 仍是默认检测方式；其他打印件必须使用实际测量的黑色方块边长。两种方式都会在
标定 JSON 中记录检测器配置。

输出 JSON 保存 `global_from_qr`、`qr_from_global`、参与估计的源帧、重投影误差、坐标系名称、
二维码尺寸和输入哈希。工具不修改三路相机 pose，也不自动把训练数据转换到二维码坐标系；
后续使用者可按 JSON 中的矩阵自行变换。二维码坐标原点位于图案中心，X 指向图案右侧，Y
指向图案上方，Z 从印刷面向外。
读取 `/tf_static` 时会合并分开发出的静态消息，直到获得头部 IMU 到 RGB 相机的完整外参链，
避免因静态消息发布顺序不同而误报缺少头部相机外参。
若当前录制确实缺少这条链，但同一套未重新装配的头部设备在另一录制中保留了完整外参，可在
二维码标定和导出时显式传入 `--head-static-calibration-source <recording-dir>`。工具仍从当前
录制读取图像、内参和全局 pose，只借用参考录制的静态外参，并在二维码 JSON 与导出清单中
记录参考 take 和 `static_calibration_borrowed`，不得用该参数跨设备或跨重新装配过程复用外参。

若全局 pose 只有瞬时尖峰，或发生一次坐标系跳变后相对运动仍连续，可先生成不覆盖原始值的
修复 manifest：

```bash
.venv/bin/rda correct-pose-drift \
  --review-manifest recording/review/manifest.json \
  --output-manifest recording/review/manifest_pose_corrected.json \
  --audit recording/review/pose_drift_audit.json
```

高置信度短尖峰使用相邻有效 pose 插值；高置信度持续跳变及其后续稳定残余跳变从跳变帧起拼接回
上一坐标段；短时、同方向且累计位移不可能由真实手部运动产生的渐进坐标漂移，会逐跳拼接并保留
漂移后的相对运动。
连续 2–4 个异常步进后恢复稳定的 settling jump 也会作为一个坐标系切换过程逐步拼接；若同一
pose 流中存在已确认的大幅坐标跳变，其前后独立出现的中等稳定跳变会按同一追踪失稳链处理。
其余中等幅度或连续异常会标成 `NEEDS_REVIEW`，不会自动修改。修复 manifest 同时
保留 `raw_positions`、`raw_quaternions_xyzw` 和逐帧 correction mask；导出与 ArUco 标定都应
使用审核通过的修复 manifest。
旋转阈值同时考虑 review 帧率可能高于 pose 发布频率；重复采样后集中到单个 review 帧的合理
手部旋转不会被误判成坐标漂移。

## 3D 轨迹质检

完成漂移审计后，可以批量生成 raw/corrected 三路 pose 的交互式 3D 对比、逐帧步长曲线和
机器可读质量报告。工具优先读取每个 take 的 PASS 漂移审计所引用的修复 manifest；若存在
二维码标定，则用 `qr_from_global` 把轨迹统一到二维码坐标系，避免把不同 take 的全局原点
差异误当成真实运动：

```bash
.venv/bin/pip install -e '.[visualization]'
.venv/bin/rda visualize-trajectories \
  --input-root recordings/ \
  --output outputs/trajectory-qc \
  --write-png
```

输出包括可拖动旋转和缩放的 `index.html`、详细 `report.json`、表格 `summary.csv`，以及可选
的逐 take PNG 和总览图。HTML 和逐 take PNG 只绘制最终 decisions 选择的 context/训练区间，
被截断的漂移区间不会混入轨迹；raw 为低透明虚线、corrected 为实线，黄色点表示所选区间内
发生 pose 修正的区域。

默认质量提示阈值是：训练区间内手部单帧位移 0.12 m、头部单帧位移 0.05 m、pose 有效率
95%、训练帧手部到头部距离的 99 分位数 1.25 m、二维码平移标准差 0.02 m、二维码最大重投影
误差 3 px。手部到头部距离仍写入报告供人工参考，但不参与漂移判定或自动评级；漂移判断依赖
pose 的时间连续性、有效率和人工画面复核。二维码坐标下的轨迹中心还会做跨 take 的稳健离群
提示。`REVIEW_REQUIRED` 只是人工复核提示，不能据此自动删除数据；仅当最终所选区间内实际
包含 pose 修正且修正后通过检查时才标记 `PASS_AFTER_CORRECTION`。

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
  --gripper-calibration configs/umi-insight3-gripper.json \
  --output outputs/recording-segmented \
  --repo-id local/recording-segmented
```

输出包含左右手和头部三路独立视频、双手状态、9 维头部 tracking pose、应用
MCAP 静态外参后的 9 维头部 RGB 相机全局 pose、有效性掩码、源帧索引和原子动作索引。
未传 `--gripper-calibration` 时双手状态维持原有 18D pose。传入标定时，导出器在左右腕图中
检测 `DICT_4X4_50` 的 0/1 号 marker，把中心像素距离映射为 `0–0.083 m` 的物理开口宽度，
并在每只手的 9D pose 后插入宽度，形成 20D state/action。两个 marker 均唯一检测到时直接
测量；配置显式的 `symmetric_midpoint` 单 marker 推算和参考图像高度后，可在仅一侧 marker
唯一可见且图像纵横比匹配时，按对称中点恢复总距离。重复、全缺失或几何不匹配仍保留为零并
由 validity mask 标出。direct/inferred 帧数、推算所用 marker、成对中点误差、端点裁剪、
原始距离和检测覆盖率均写入审计清单。`action` 是下一帧完整双手状态；episode 最后一帧保持
当前状态。pose 仍处于源追踪坐标系，且未经过机器人重定向。
每帧还包含左右手独立 subtask ID、左右手 subtask 内进度和整体操作进度；ID 到英文语义的
映射写入 `rda/export_manifest.json`。导出器同时生成 `meta/manifest.json` 和
`meta/modality.json`，记录维度、夹爪物理语义、标定参数及字段分组。二维码上下文的整体操作进度固定为 0，正式任务区间从
0 线性增长到 1。
头部 pose 默认按相机名推断对应的 `<name>_camera_imu` tracking frame；设备命名不符合该规则
时，导出和二维码标定都应显式传入 `--head-pose-child-frame`。
导出使用同目录临时产物完成后原子改名，拒绝覆盖已有数据集。
默认使用流式视频编码：相机帧从 MCAP 解码后直接送入三路编码器，避免先写临时 PNG 再
读回的磁盘往返，并在 `rda/export_manifest.json` 中记录 `video_encoding_mode`。每路编码器
默认使用 256 帧的有界队列，吸收头部高清流的短时编码积压；可用
`--encoder-queue-maxsize` 调整。每路编码器默认限制为 2 个线程，避免三路相机或批量导出
时在有限 CPU 上过度抢占；可用 `--encoder-threads` 调整。队列和线程实际值都会写入导出
清单。若需要诊断编码器兼容问题，可
显式传入 `--video-encoding-mode staged-png` 恢复分阶段编码。

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

内置 `screw-nut-sorting` 以“一整批混合件分类完成”为 episode：固定左盒放螺母、固定右盒
放螺丝，同一录制中的人工重新摆料只作为多 episode 之间的间隔。第一版不提供自动分段插件，
必须人工确认每轮完整性、分类正确性以及左右夹爪各自的分类和撤离边界。

仓库自动化开发约束见 [AGENTS.md](AGENTS.md)；`CLAUDE.md` 与其同步并一同提交。代码行为、
命令、schema 或 Skill 发生变化时，必须同步文档、运行测试、提交并推送当前分支。
本地 `tests/` 用于开发验证，但不会提交到 GitHub，也不会进入 wheel 或 sdist。
