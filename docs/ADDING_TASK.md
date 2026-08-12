# 添加新任务

1. 在包内 `task_specs/` 增加任务 JSON，定义任务 ID、原子动作、最短帧数和插件导入路径；
   用 `rda copy-task` 生成用户可编辑副本。
2. 在 `tasks/<task>/` 实现 `suggest(observations, valid)`；无法可靠判断时返回
   `status=unresolved`，不要伪造边界。
3. 为正常、短片段、缺失观测、边界模糊和多 episode 审核各增加测试。
4. 如需新数据格式，在 `adapters/` 增加适配器，禁止把格式依赖带入 core。
5. 用 `rda validate-decisions` 检查穷尽覆盖，再接入导出器和官方验证器。

自动门的优化指标至少同时报告候选精度和候选召回率。高精度低召回只适合加速人工确认，
不能用于自动拒绝数据。
