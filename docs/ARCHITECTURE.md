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

任务插件只负责从规范化观测提出候选边界。候选是审核提示，不可替代人工视觉结论。数据源
适配器负责把原始数据变成规范化观测；导出适配器负责生成训练格式并写验收证据。
插件可以返回单个候选，也可以返回同一源片段中的多个有序 episode 候选；core decisions
始终使用源片段帧坐标验证覆盖、顺序和最短动作长度。

路径只出现在用户生成的会话配置中。仓库源码、任务配置和 Skill 禁止写入设备绝对路径。
