# 架构

依赖方向固定为：`CLI → core ← tasks`，外部格式通过 `adapters` 调用 core。core 不得导入
ROS、OpenCV、PyArrow、LeRobot 或任何设备仓库模块。

批次状态由文件证据推导，不把 watcher PID 或 campaign report 当作事实来源：

```text
DISCOVERED
  → REVIEW_READY
  → REVIEW_DECIDED
  → EXPORTED
  → INTERNAL_VALID
  → COMPLETE
```

每个 source segment 必须有且只有一个审核结论。PASS 结论可包含多个 episode，从而保留一个
停顿片段中的多次完整操作。每个 episode 的语义边界必须单调、覆盖完整区间，并满足任务配置
规定的最短模型上下文。

episode 可选的 `context_start_frame` 允许保留正式任务开始前的连续上下文，且必须满足
`context_start_frame <= episode_start_frame`。旧 decisions 缺少该字段时视为两者相等。
任务配置只有定义 `context_action` 才能接受上下文帧。上下文使用原子动作索引 `-1`，不会
改变原有动作边界；左右手 subtask 使用各自的完整边界流，允许跨越整体动作边界。subtask ID
使用独立的全局编号，逐帧进度由导出器确定性生成。旧 decisions 未提供手部边界时，仅当手部
阶段数与整体动作数相等才按整体边界兼容读取。

任务插件只负责从规范化观测提出候选边界。候选是审核提示，不可替代人工视觉结论。数据源
适配器负责把原始数据变成规范化观测；导出适配器负责生成训练格式并写验收证据。
插件可以返回单个候选，也可以返回同一源片段中的多个有序 episode 候选；core decisions
始终使用源片段帧坐标验证覆盖、顺序和最短动作长度。

任务可通过 `episode_pose_quality` 声明逐 episode 的 pose 质量插件。Insight 适配器负责从
ROS 2 bag 读取 native VIO 与 Insight Global、按 header 时间配对并重采样到 review 时间轴；
`tasks/screw_nut_sorting/pose_quality.py` 只接收规范化矩阵，判断 native 跳变是否被 Global
同帧抵消。Global 也跳变且未修正、延迟修正、数据缺口或对齐变换不连续均返回
`NEEDS_REVIEW`。episode 边界外的事件不会进入当前轮次，因此重置间隔或前一轮漂移不会污染
后续 episode。导出门校验审计与 manifest、decisions、task spec 的哈希，且只接受所有待导出
episode 均为 `training_usable=true` 的 `PASS` 审计；它不修改 pose，也不自动删除源数据。

`adapters/lerobot_export.py` 是可选依赖边界：它延迟导入 ROS、OpenCV 和 LeRobot，按审核
时间轴对 MCAP 三路相机做最近邻同步，并只导出 decisions 接受的帧。低维观测来自已同步的
review manifest，视频来自原始相机消息；动作语义、同步误差、输入哈希和 LeRobot 版本写入
数据集内的 `rda/export_manifest.json`。视频默认在源消息解码时直接流式送入三路编码器，
省去逐帧 PNG 暂存；兼容模式仍可使用 PNG 分阶段编码。导出器当前只接受单 source segment，
避免把多个独立时间轴静默拼接。

可选的 `--gripper-calibration` 在适配器边界内从左右腕 RGB 帧检测 ArUco 0/1 marker。
像素中心距离按相机名选择标定，再映射到统一的物理夹爪宽度；左右各 9D pose 后插入宽度形成
20D state，action 使用同一 episode 下一帧的完整 20D 状态。可选的 `symmetric_midpoint`
标定为固定腕相机提供参考图像尺寸和两 jaw marker 的对称中点；仅一个 marker 唯一可见时，
适配器可用该 marker 到中点距离的两倍恢复总距离。该模式必须显式配置，且输入纵横比必须与
标定匹配。双 marker 可见时仍优先直接测量；重复、全缺失或几何不匹配时保持零并标记
invalid。检测覆盖率、direct/inferred 来源、成对中点误差、裁剪和原始距离统计写入
provenance。导出器始终写 `meta/manifest.json` 与 `meta/modality.json`，记录维度、物理语义
和字段分组。

头部 review pose 是 tracking frame 的全局 pose。导出器读取 MCAP 的 `tf_static` 和彩色相机
`CameraInfo`，沿静态坐标树计算 tracking frame 到 RGB 相机的外参，额外输出
`observation.head_camera_pose_global` 及其有效性掩码。原有 `observation.head_pose` 保留，
避免静默改变旧字段语义。

`adapters/qr_calibration.py` 只负责标定元数据：从明确保留的头部上下文帧检测普通 QR 或
指定字典与 ID 的 ArUco 方形标记，使用
已确认的实物边长和相机内参求解 PnP，再结合头部全局 pose 与静态外参得到
`global_from_qr`。结果和逆矩阵写入独立 JSON；它不变换或覆盖数据集中的任何相机 pose。

`adapters/pose_drift.py` 在导出前审计三路低维全局 pose。成对的大跳变会被视为短尖峰并在
SE(3) 上插值；短时、同方向且累计位移明显不可能的渐进漂移会逐跳拼接坐标系，同时保留漂移
结束后的相对运动；单次极大跳变且后续局部运动稳定时，会从跳变点起拼接坐标段。
不满足高置信度条件的跳变只写 `NEEDS_REVIEW`。修复结果写到新 manifest，保留原始数组、
逐帧 correction mask、阈值和事件审计，不覆盖源 review manifest。

`adapters/trajectory_qc.py` 消费 decisions、最终 PASS 漂移审计、修复 manifest 和可选二维码
标定，生成 raw/corrected 三路轨迹的自包含交互式 HTML、JSON/CSV 指标和可选 PNG。交互轨迹
与逐 take PNG 只绘制 decisions 选择的 context/训练帧，不显示被截断区间。二维码变换只在
显示和跨 take 比较时应用，不回写 pose。步长和有效率参与所选训练区间的时序评级；手部到
头部的相对距离仅作为人工导航信息，不参与评级。二维码质量和稳健轨迹中心仍可独立触发标定或
空间一致性复核，但不能当作 VIO 时序漂移证据，也不改变 decisions 或训练数据。只有所选区间
包含实际修正时才可能标记 `PASS_AFTER_CORRECTION`。

`adapters/lerobot_validation.py` 先直接检查 Parquet 的 row、episode、timestamp、source
frame、18D/20D 状态、物理夹爪非负性、有效性和 next-frame action 不变量，并核对
`meta/manifest.json` 与 `meta/modality.json`，再完整解码每一个视频文件。随后用
官方 `LeRobotDataset` 做代表性索引并通过 PyTorch DataLoader 读取一个 batch；两个阶段分别
写独立 PASS 证据，官方检查失败时不会伪造完成状态。
Python 3.10 验证环境的直接版本锁定位于 `configs/lerobot-validator-py310.txt`，不得把这组
通用机器学习 wheel 安装进厂商设备 Python 环境。

LeRobot 工作数据集与严格交付包是两个产物。工作数据集内的 `rda/` 保存导出 provenance 和
验证 PASS 证据，是本项目扩展而不是 LeRobot v3 schema；严格交付副本只复制本视频数据集所
引用的 `data/`、`meta/`、`videos/`，并把 `rda/` 证据保存在副本外。官方写入器可能遗留空的
`images/<video-key>/` 编码暂存目录：只有确认其中没有文件时才可在交付副本中省略；如果存在
文件，必须先查明 `meta/info.json` 的 feature dtype 和引用路径，不能按目录名直接删除。

路径只出现在用户生成的会话配置中。仓库源码、任务配置和 Skill 禁止写入设备绝对路径。
