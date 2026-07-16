from __future__ import annotations

import json


def markdown_review(record: dict) -> str:
    lines = [
        "# 本地代码审查结果", "",
        f"- 状态：{record.get('status', 'UNKNOWN')}",
        f"- 合并就绪度：{record.get('decision', 'UNKNOWN')}",
        f"- 风险：{record.get('risk_level', 'unknown')}",
        f"- Profile：{record.get('profile', 'unknown')}",
        f"- 模型：{record.get('model', 'unknown')}",
        f"- 创建时间：{record.get('created_at', '')}",
        "", "## 摘要", "", record.get("summary", ""),
        "", "## 理由", "", record.get("reasoning", ""),
        "", "## 审查意见", "",
        "| 严重级别 | 文件 | 行号 | 类别 | 问题 | 建议 |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for finding in record.get("findings", []):
        values = [
            finding.get("severity", ""), finding.get("path", ""), finding.get("line") or "",
            finding.get("category", ""), finding.get("message", ""), finding.get("suggestion", ""),
        ]
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in values) + " |")
    coverage = record.get("coverage", {})
    lines.extend([
        "", "## 覆盖情况", "",
        f"- 已审查批次：{coverage.get('reviewed_batches', 0)}/{coverage.get('planned_batches', 0)}",
        f"- 未审查项：{', '.join(coverage.get('unreviewed_items', [])) or '无'}",
        f"- 排除项：{json.dumps(coverage.get('excluded_items', []), ensure_ascii=False)}",
        "", "## 传统模型参考", "",
        record.get("ml_reference", {}).get("disclaimer", "未提供传统模型参考。"),
        "", "## 调用详情", "",
        f"- Token：{record.get('usage', {}).get('total_tokens', 0)}",
        f"- 模型耗时：{record.get('latency', {}).get('model_seconds', 0):.3f}s",
        f"- 缓存命中：{'是' if record.get('cached') else '否'}",
    ])
    return "\n".join(lines) + "\n"