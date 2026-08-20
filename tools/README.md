# tools · 复核页面的生成

两步，都在评分之后跑，都不影响任何分数。

```bash
PYTHONPATH=. python tools/render_evidence.py   # 读 out/pilot/*.json → /tmp/shots.json
PYTHONPATH=. python tools/build_review.py      # → docs/p3-review.html
```

`render_evidence.py` 重建每个资产的采样计划（确定性的，同一份代码同一个结果），
按结果里记录的构型标签取回 qpos，重新摆位、渲染。
渲染只读冻结的结果，改不了它——这是 P3「判分不看图」那条规则的落地方式。
