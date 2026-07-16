## 计划：完成实验七本地 Web 代码审查工作台

根据[实验七设计规约](/home/wzsyh/ai-software-engineer/docs/specs/2026-07-16-exp7-local-web-code-review-design.md)，新建完整的 `Experiment7/`，交付 FastAPI 后端、原生 Web 前端、DeepSeek 主审查、实验二传统模型参考、JSONL 历史、自动测试和双语 README。当前优先完成可验收软件，LaTeX 报告只建立可编译骨架和真实结果占位。

**实施阶段**

1. **工程骨架与启动服务**
   - 增加 FastAPI、Uvicorn、Pydantic、pytest、httpx、Playwright 依赖。
   - 实现 app factory、统一错误、request ID、静态托管和 `/api/v1/health`。
   - 验证锁定仓库、Git 工作树、有效 `HEAD` 和监听地址。
   - 唯一推荐启动方式：进入 `Experiment7/` 后运行 `uv run python -m src.app serve --repo <path>`。

2. **Git 工作区与不可变快照**
   - 使用 porcelain v2 和 `HEAD -> worktree` diff 合并 staged/unstaged 修改。
   - 支持新增、修改、删除、重命名和未跟踪文本文件。
   - 实现统一 diff parser、精确 hunk 行号和确定性 `snapshot_hash`。
   - 完成 workspace、diff API，以及路径遍历、符号链接和二进制防护。

3. **安全预检与本地上下文**
   - 实现敏感路径、凭据模式、文件大小和文本类型检查。
   - 上下文顺序固定为 diff、结构块/完整文件、相关测试、符号引用、项目约定。
   - 按文件和 hunk 稳定分批，记录所有排除、截断和未覆盖内容。
   - 实现 `/preflight` 和 `expected_snapshot_hash` 的 HTTP 409 乐观并发控制。

4. **LLM 主审查**
   - 从实验四迁移 OpenAI 兼容 transport、重试、usage 和内容缓存机制。
   - 从实验六迁移 P1/P5 思想，改写为本地工作树的 `fast`/`strict` Prompt。
   - 使用 Pydantic 严格校验联合 JSON 输出。
   - 校验 blocker、风险、finding 路径及修改后行号；非法意见丢弃并记录警告。
   - JSON 失败仅允许一次格式修复，禁止自由文本降级为有效结论。

5. **审查编排与大改动处理**
   - 单批直接生成结果，多批才执行结构化汇总。
   - 强制执行 REJECT、blocker 和 `INCOMPLETE` 的确定性不变量。
   - 所有代码审查和格式修复合计最多 4 次真实请求，另最多 1 次汇总。
   - 用客户端 request ID 和只读 progress API 展示当前批次。
   - 调用期间工作区变化时保留结果，但明确标记 stale。

6. **传统机器学习参考**
   - 复用实验二四个 `pre-review` 模型和 `scaler_pre.pkl`。
   - 以 scaler 中实际的 93 列为唯一特征顺序。
   - 迁移实验五固定词表和人类 PR 语料 IDF 恢复方法，只执行 `transform`。
   - 固定缺失策略：`num_commits=0`，HEAD 消息仅作为有限代理，PR 标题/正文派生特征归零并明确披露。
   - ML 失败不影响 LLM；非 Python scope 不运行 ML。

7. **历史与导出**
   - 实现线程安全、原子写入的 UTF-8 JSONL 存储。
   - 支持最近 200 条淘汰、分页筛选、动态 stale、删除和清空。
   - 默认不保存 diff；显式启用时仍受安全检查和字符预算约束。
   - 支持单条 JSON/Markdown 和全部历史 JSON 导出。

8. **原生 Web 工作台**
   - Review/History 两个一级视图。
   - 桌面三栏：修改文件、只读 diff、审查结果；窄屏切换为三个页签。
   - 审查前展示预检，审查中展示进度，审查后展示结果、coverage、ML、token、耗时和缓存。
   - finding 点击后定位并高亮对应 hunk。
   - 提供完整空状态、degraded、sensitive blocked、incomplete 和 stale 状态。
   - 本地提供字体与 Lucide 图标，不依赖 CDN 或 Node 构建。

9. **测试、文档与报告骨架**
   - TestClient 作为唯一主测试接缝，使用真实临时 Git 仓库和确定性 LLM 桩。
   - 测试 staged/unstaged、未跟踪、重命名、安全阻止、409、stale、4+1、ML、历史和导出。
   - Python Playwright 集成 pytest，覆盖桌面与移动核心流程和截图检查。
   - 完成中英文 README、显式付费冒烟入口和故障排查。
   - 同步 [AGENTS.md](AGENTS.md) 与 [CLAUDE.md](CLAUDE.md) 中实验七的 Web 载体说明。
   - 建立可编译报告骨架，不虚构实验数字、截图或案例。

**关键复用点**

- [Experiment4/src/llm_client.py](Experiment4/src/llm_client.py)：调用、重试、usage 和缓存思想。
- [Experiment6/src/prompts.py](Experiment6/src/prompts.py)：P1/P5 Prompt 机制。
- [Experiment6/src/context_builder.py](Experiment6/src/context_builder.py)：确定性上下文与预算先例。
- [Experiment2/src/feature_extraction.py](Experiment2/src/feature_extraction.py)：AST、CFG 和统计特征。
- [Experiment2/results/models](Experiment2/results/models)：四个 pre-review 模型与 scaler。
- [Experiment5/src/ml_features.py](Experiment5/src/ml_features.py)：固定 TF-IDF 词表与 IDF 恢复。
- [Experiment5/src/ml_predict.py](Experiment5/src/ml_predict.py)：模型加载、列对齐与概率输出。

**验证门槛**

1. `uv run pytest tests -q` 默认完全离线且不产生 API 费用。
2. Playwright 验证桌面三栏、移动页签、finding 定位和无重叠/溢出。
3. 验证真实请求严格不超过 4 个代码请求加 1 个汇总请求。
4. 验证敏感内容不会外发或出现在错误、历史和日志中。
5. 配置 `DEEPSEEK_API_KEY` 后，显式执行真实 `strict` 冒烟并记录 token、耗时、缓存与覆盖率。
6. 完成规约列出的启动、MERGE/REJECT、四模型参考、stale、历史导出、降级状态和响应式布局验收。
