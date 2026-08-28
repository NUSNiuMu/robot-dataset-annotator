# 添加新任务

1. 在包内 `task_specs/` 增加任务 JSON，定义任务 ID、原子动作和最短帧数；有可靠的自动候选
   算法时再增加插件导入路径；
   用 `rda copy-task` 生成用户可编辑副本。
2. 自动候选任务在 `tasks/<task>/` 实现 `suggest(observations, valid)`；无法可靠判断时返回
   `status=unresolved`，不要伪造边界。
3. 为正常、短片段、缺失观测、边界模糊和多 episode 审核各增加测试。
4. 如需新数据格式，在 `adapters/` 增加适配器，禁止把格式依赖带入 core。
5. 用 `rda validate-decisions` 检查穷尽覆盖，再接入导出器和官方验证器。

需要与整体动作独立的左右手训练子语义时，在 task 顶层定义
`hand_subtasks.left_hand` 和 `hand_subtasks.right_hand` 数组，并在每个 PASS episode 的
`hand_subtask_boundaries` 中分别给出覆盖完整操作区间的边界流。未定义顶层数组时，core 仍
支持在每个 `atomic_actions` 条目中定义手部语义，并使用整体动作边界。需要保留正式任务前
上下文时，任务配置还要定义 `context_action`，decisions 才能使用 `context_start_frame`。
上下文不属于原子任务或手部子任务边界，导出索引固定为 `-1`。

插件可以为单个候选返回 `boundaries`，也可以为同一源片段内的多个完整尝试返回按时间排序
的 `episodes`。每个 episode 都必须包含源帧坐标下的 `episode_start_frame`、
`episode_end_frame_exclusive` 和覆盖全部原子动作的 `atomic_boundaries`。这些结果仍只是审核
提示，不能直接替代 decisions 中的视觉结论。

自动门的优化指标至少同时报告候选精度和候选召回率。高精度低召回只适合加速人工确认，
不能用于自动拒绝数据。

`screw-nut-sorting` 把“一批混合件全部分完”作为 episode，而不是把左右夹爪并发且互相
重叠的单件抓放强行拆开。螺母放入固定左盒，螺丝放入固定右盒；最后一个物体释放后进入完成
和撤离阶段。人工重新摆料属于 episode 间隔，人工干预、错放、盒外掉落或中央区域残留均不能
作为 PASS episode。该任务没有自动插件，必须在所有视角中人工确认每轮边界，并分别记录
左右夹爪结束分类、等待或撤离的子任务边界。
