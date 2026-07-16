# 实验七设计规约：本地 Web ,能代码审查工作台

> 本文件是实验七后续开发、测试和验收的唯一权威设计依据。
> 生成日期：2026-07-16。
> 上游依据：`docs/实验指导书_代码审查.md` 第 7 章、实验二传统机器学习产物、
> 实验四 LLM 调用层，以及实验五、实验六对 AI 代码迁移与输入优化的结论。

## Problem Statement

前六个实验已经完成代码审查数据构建、传统机器学习合并预测、LLM 合并预测与审查意见生成、
AI 生成代码泛化分析和上下文/Prompt 优化，但产物仍以离线脚本和实验矩阵为主，开发者无法在
实际本地代码修改上方便地调用这些能力。

实验指导书原本要求开发 VS Code 插件，具备获取当前编辑文件、获取代码修改、调用审查模型、
展示 Merge Prediction、展示自动审查意见和查询历史记录等能力。教师允许将交互载体简化，
因此本实验不开发 VS Code Extension，而要提供一个浏览器可访问的本地 Web 软件，同时尽量保留
原要求中的模型集成、HTTP 前后端分离、结果可视化、历史记录、状态检测和工程部署目标。

本地未提交修改不具备 PR 标题、平台讨论、最终维护决策等完整信息，所以软件不能把结果描述为
对 GitHub 最终合并状态的可靠预测。用户真正需要的是：在提交或创建 PR 前，查看当前 Git 工作区
修改，获得基于前序模型的合并就绪度判断、带代码位置的审查意见、传统模型参考和可追溯的历史记录。

## Solution

构建一个由 FastAPI 托管的本地 Web 代码审查工作台。用户启动服务时通过 `--repo <path>` 锁定
一个本地 Git 仓库，然后在浏览器中完成全部操作。前端左侧展示修改文件，中间展示选中文件的只读
diff，右侧展示审查结果；另设历史页签。修改文件列表中的当前选中项即 Web 版本的“当前编辑文件”。

软件默认读取相对 `HEAD` 的全部本地修改，包括 staged、unstaged 和未跟踪文本文件。用户可以手动
发起“审查当前文件”或“审查全部修改”。FastAPI 对工作区建立不可变快照，进行路径和敏感信息预检，
构建本地软件工程上下文，再调用 DeepSeek LLM。默认采用实验六 P5 Self-Reflection 的审慎策略，
并提供 P1 快速策略。一次结构化推理同时返回合并就绪度和最多 5 条带位置的意见；大改动按预算分批，
最多执行 4 次分批审查和 1 次汇总。

实验二的 SVM、Random Forest、XGBoost 和 LightGBM `pre-review` 模型仅对 Python 修改运行，结果
作为“实验模型参考”展示，不参与 LLM 主结论。由于本地输入缺少部分 PR 元数据，界面和导出内容必须
明确标注该结果属于实验外迁移参考。审查记录由后端写入 JSONL，默认不保存代码 diff，只保存输入哈希、
结构化结果和调用元数据；历史最多保留 200 条。

## Requirements

### 1. 启动与工作区

- **R1** 提供服务启动入口，语义等价于 `ai-review serve --repo <path>`；实现可采用
  `uv run python -m Experiment7.src...` 或项目脚本入口，但最终 README 必须给出单一推荐命令。
- **R2** `--repo` 必须指向存在且可读取的 Git 工作树。路径在启动时解析为规范绝对路径，并在整个
  服务生命周期内保持不变。
- **R3** 服务默认仅监听 `127.0.0.1`。若用户显式改为非回环地址，启动日志必须警告该服务能够读取
  被锁定仓库中的代码。
- **R4** 浏览器访问根路径即可打开完整工作台。FastAPI 同时托管 REST API 和静态前端，不要求单独
  启动 Node.js 开发服务器。
- **R5** 后端不得接受任意本地仓库路径作为普通 API 参数。所有文件操作必须限定在启动时锁定的仓库内。
- **R6** 仓库不存在、不是 Git 工作树、`HEAD` 不存在或无读取权限时，服务应以明确错误拒绝启动；
  不得在未知目录静默运行。

### 2. 修改发现与当前文件

- **R7** 默认修改范围是相对 `HEAD` 的全部本地修改：已暂存、未暂存和未跟踪文本文件。
- **R8** 同一文件同时存在 staged 与 unstaged 修改时，展示和送审内容必须等价于当前工作树相对
  `HEAD` 的最终净差异，不能重复计算同一修改。
- **R9** 未跟踪文本文件按“空文件到当前文件”的新增 diff 处理；二进制、过大文件、符号链接目标和
  敏感路径不得作为普通代码送审。
- **R10** 修改文件列表至少展示相对路径、状态（新增/修改/删除/重命名）、增加行数、删除行数、语言
  和是否可审查。
- **R11** 用户在文件列表中选中的文件定义为“当前编辑文件”。页面初次加载时默认选择第一个可审查
  修改文件；没有修改时显示明确空状态。
- **R12** 中间区域以统一 diff 形式只读展示当前文件修改，并正确区分上下文行、增加行、删除行和 hunk
  行号。前端不得编辑或保存源文件。
- **R13** 提供“刷新工作区”操作。刷新后保留仍存在的文件选择；若当前文件不再修改，则选择下一个
  可审查文件或进入空状态。

### 3. 审查触发与快照一致性

- **R14** 提供“审查当前文件”和“审查全部修改”两个明确的手动操作。文件变化本身不得自动触发
  付费 LLM 调用。
- **R15** 发起审查时，后端必须重新读取 Git 状态并创建工作区快照。快照至少包含仓库标识、scope、
  目标文件、规范化 diff、文件状态、内容哈希和创建时间。
- **R16** 每次审查使用确定性的 `snapshot_hash` 绑定输入。审查完成后若当前工作区对应 scope 的哈希
  已变化，前端必须将结果标记为“已过期”，而不能继续显示为当前代码结论。
- **R17** 同一 `snapshot_hash + profile + scope + 模型配置` 的并发重复请求应被合并或拒绝，避免重复
  付费调用。已完成结果可以复用缓存，但必须向用户显示缓存状态。
- **R18** 审查运行期间展示进行状态，并禁止对同一 scope 重复提交；用户仍可浏览已有 diff 和历史。

### 4. 预检与外发边界

- **R19** 模型调用前必须生成可展示的预检结果，包括：拟送审文件、上下文来源、总字符数、预计批次、
  排除文件和排除原因。
- **R20** 自动排除二进制文件、常见密钥/证书、环境配置、凭据目录、构建产物、依赖目录和超预算文件。
  初始敏感路径至少覆盖 `.env*`、`*.pem`、`*.key`、`*.p12`、`id_rsa*`、`.ssh/`、`.aws/`、
  `node_modules/`、`.venv/`、`dist/`、`build/` 和 Git 内部目录。
- **R21** 对拟发送文本执行凭据模式扫描，至少检查常见 API key、GitHub token、私钥头、AWS access key、
  bearer token 和高置信度密码赋值。命中高置信度敏感内容时必须阻止该文件外发，并显示原因，不提供
  绕过按钮。
- **R22** 所有路径在读取前都要解析并验证仍位于仓库根目录内。拒绝 `..`、绝对路径逃逸、符号链接
  逃逸和 URL 编码后的等价逃逸。
- **R23** 调用外部模型前，界面必须使用户能够看到实际发送范围。用户点击审查即表示确认发送预检中
  未被排除的代码上下文。
- **R24** 被阻止或超预算而未审查的文件/hunk 必须列入覆盖报告；不得静默忽略。

### 5. 本地软件工程上下文

- **R25** 上下文从实验六的 GitHub PR 语义改造为本地工作树语义，不依赖 GitHub PR、Issue、Review
  或在线仓库 API。
- **R26** 每个审查批次的必需上下文为规范化 diff，附文件路径、状态和 hunk 的修改后行号。
- **R27** 对可读取文本文件加入修改后的完整文件或与 hunk 相邻的结构块。Python 优先使用 tree-sitter
  定位 enclosing function/class；无法解析或其他语言时回退到确定性的行窗口/文件截断。
- **R28** 自动检索与改动相关的本地测试文件。优先规则为同名测试、相邻 `tests/`、被改符号在测试路径
  中的词法引用；按固定排序和预算截断。
- **R29** 自动检索被改顶层符号在仓库其他文本代码文件中的词法引用。忽略 Git 排除目录、二进制、
  生成文件和当前改动文件；限制符号数、命中文件数和单片段长度。
- **R30** 加入项目约定摘要，来源优先级为仓库根目录的 `AGENTS.md`、`CLAUDE.md`、`CONTRIBUTING*`、
  `README*` 和可识别的测试/格式化配置。只取与代码风格、测试、贡献约定直接相关的预算片段。
- **R31** 上下文构建必须确定性：相同快照和配置产生相同顺序、片段和哈希，不使用随机抽样。
- **R32** 前端结果详情可展开查看实际使用的上下文来源、包含/排除文件、字符预算和截断状态，但默认
  不展示完整重复源码。

### 6. LLM 主审查

- **R33** 执行模型默认复用实验四/六的 `deepseek-v4-flash` 和现有 OpenAI 兼容调用层，包括超时、
  重试、计时、usage 记录与内容哈希缓存。模型 ID 和 API base URL 由后端配置，API key 只从环境读取。
- **R34** 默认 profile 为 `strict`，采用实验六 P5 Self-Reflection 的核心机制：先主动寻找 blocking
  缺陷，再基于缺陷反推合并就绪度。另提供 `fast` profile，采用 P1 单轮快速策略。
- **R35** 软件使用面向本地变更的新结构化 Prompt，不得直接声称输入是 GitHub PR。Prompt 要求一次
  返回总体结论和逐 hunk 意见。
- **R36** 单批输出必须是可校验 JSON，核心结构如下：

  ```json
  {
    "decision": "MERGE",
    "risk_level": "low",
    "summary": "当前修改具备合并条件",
    "reasoning": "未发现阻断性缺陷",
    "blocking_defects": [],
    "findings": [
      {
        "path": "src/example.py",
        "line": 42,
        "severity": "warning",
        "category": "correctness",
        "message": "边界条件未处理",
        "suggestion": "在进入分支前显式校验空输入"
      }
    ]
  }
  ```

- **R37** `decision` 仅允许 `MERGE` 或 `REJECT`；界面主标题使用“合并就绪度”，并说明它不等价于
  对 GitHub 最终合并状态的预测。
- **R38** `risk_level` 仅允许 `low`、`medium`、`high`。只要存在 blocker 严重级别意见，最终结论
  必须为 `REJECT` 且风险至少为 `high`；没有 blocker 不强制判定 `MERGE`，模型可因证据不足拒绝。
- **R39** finding 严重级别为 `info`、`warning` 或 `blocker`；类别使用受控值集合，例如 correctness、
  security、compatibility、performance、maintainability、testing、style、documentation。
- **R40** finding 的 `path` 必须属于本次送审批次；`line` 必须为空或落在该文件修改后 hunk 行范围内。
  解析器要丢弃无法定位的意见并记录解析警告，不能伪造位置。
- **R41** 最终展示最多 5 条去重后的最高价值意见，排序为 blocker、warning、info，再按文件与行号稳定排序。
- **R42** JSON 解析失败时允许一次格式修复重试。再次失败则该批次标记失败，保留错误信息和原始调用
  元数据，但不把自由文本当成有效结论。

### 7. 大改动分批与汇总

- **R43** 单次上下文超预算时，优先按文件边界分批；单文件仍超预算时再按 hunk 边界切分。禁止在一行
  或一个 diff 元信息块中间截断。
- **R44** 每次用户操作最多产生 4 个分批审查调用和 1 个汇总调用。可通过配置降低预算，但不能静默
  提高该默认硬上限。
- **R45** 批次选择优先覆盖可审查 Python 文件，再覆盖其他文本代码；同类文件按相对路径稳定排序。
  未进入前 4 批的内容列为 `unreviewed_items`。
- **R46** 只有一个批次时直接使用该批结果，不产生汇总调用。多个批次时，汇总调用只接收各批结构化
  结果和覆盖摘要，不再次发送完整代码。
- **R47** 汇总结论遵循保守不变量：任一有效批次为 `REJECT` 或含 blocker 时，总体不得为 `MERGE`；
  所有批次均有效且无阻断问题时才允许总体 `MERGE`。
- **R48** 部分批次失败时，总体状态为 `INCOMPLETE`，界面不得将其渲染为可靠 MERGE。结果仍可展示
  已成功批次的意见和失败原因。

### 8. 传统机器学习参考

- **R49** 仅对包含 Python 修改的审查 scope 运行实验二四个 `pre-review` 模型：SVM、Random Forest、
  XGBoost、LightGBM。禁止使用含审查后过程变量的 `full` 模型。
- **R50** 复用实验二 AST/CFG、代码统计和文本特征定义、保存的 scaler 与模型权重；复用实验五通过
  固定词表和实验二人类语料恢复 TF-IDF IDF 的方法。所有下游输入只 `transform`，禁止重新训练、
  `fit` scaler 或在当前仓库文本上重估 IDF。
- **R51** 本地可获得特征由 diff 和 Git 元数据构造：additions、deletions、changed_files、Python patch、
  最近提交消息等。PR 标题/正文及不可获得字段按明确的缺失策略构造，并随结果记录 `missing_features`。
- **R52** 四模型结果包括模型名、MERGE/REJECT 标签和模型支持时的 MERGE 概率。任一模型加载或推理
  失败不得阻断 LLM 主审查，错误应在 ML 参考区域单独显示。
- **R53** ML 区域必须标注：“传统模型在 GitHub PR 特征上训练；当前本地输入缺少部分 PR 元数据，
  结果仅作实验外迁移参考，不参与合并就绪度结论。”
- **R54** 非 Python scope 不运行 ML，界面显示“该语言未经过实验二模型验证”，但允许 LLM 降级审查。

### 9. API 契约

- **R55** API 统一以 `/api/v1` 为前缀，至少提供以下端点：

  | 方法     | 路径                     | 外部行为                                    |
  | -------- | ------------------------ | ------------------------------------------- |
  | `GET`    | `/health`                | 服务、仓库、LLM 配置、ML 资产和历史存储状态 |
  | `GET`    | `/workspace`             | 仓库摘要、当前快照哈希、修改文件列表        |
  | `GET`    | `/diff?path=...`         | 单个修改文件的规范化 diff 与 hunk 元数据    |
  | `POST`   | `/preflight`             | 对当前文件或全部修改生成预检与预计批次      |
  | `POST`   | `/reviews`               | 基于当前快照发起同步审查并返回完整结果      |
  | `GET`    | `/reviews/{id}`          | 获取单条历史详情并返回是否已过期            |
  | `GET`    | `/history`               | 分页/筛选历史摘要                           |
  | `DELETE` | `/history/{id}`          | 删除单条历史                                |
  | `DELETE` | `/history`               | 清空历史                                    |
  | `GET`    | `/exports/{id}.{format}` | 导出单次结果，format 为 json 或 md          |
  | `GET`    | `/history/export`        | 导出历史 JSON                               |

- **R56** `POST /preflight` 和 `POST /reviews` 请求核心结构为：

  ```json
  {
    "scope": "file",
    "path": "src/example.py",
    "profile": "strict",
    "include_diff_in_history": false,
    "expected_snapshot_hash": "sha256:..."
  }
  ```

  `scope` 仅允许 `file` 或 `workspace`；workspace scope 的 `path` 必须为空。
- **R57** 写操作必须携带前端最后观察到的 `expected_snapshot_hash`。哈希不匹配时返回 HTTP 409 和新
  哈希，要求用户刷新预检，禁止对意外变化后的代码继续调用模型。
- **R58** 成功审查响应至少包含：`id`、`status`、`snapshot`、`profile`、`decision`、`risk_level`、
  `summary`、`reasoning`、`findings`、`coverage`、`ml_reference`、`usage`、`latency`、`cached`、
  `created_at` 和 `stale`。
- **R59** 错误响应采用统一结构 `{code, message, details, request_id}`。至少区分：无修改、路径非法、
  快照冲突、敏感内容阻止、上下文为空、LLM 未配置、LLM 调用失败、模型解析失败和内部错误。
- **R60** DeepSeek API key 未配置或模型不可用时，`health` 返回 degraded 而非让服务退出。workspace、
  diff、历史和可独立运行的 ML 参考仍可访问；主审查按钮禁用并显示具体原因。

### 10. 历史记录与导出

- **R61** 后端是历史记录的唯一写入和查询方。历史存储采用 UTF-8 JSONL，默认位于
  `Experiment7/results/history/reviews.jsonl`，可通过配置覆盖。
- **R62** 每条历史记录是自包含 JSON 对象，至少保存审查 ID、仓库标识、scope、文件列表、
  `snapshot_hash`、profile、主结论、findings、覆盖状态、ML 参考、模型/usage/耗时、缓存状态、创建时间
  和 schema version。
- **R63** 默认不保存完整 diff 或完整源码。只保存 diff 哈希、行数和上下文来源摘要。用户单次启用
  `include_diff_in_history` 时可保存受字符预算限制的送审 diff，并在历史详情中显示敏感提示。
- **R64** 历史最多保留最近 200 条。写入新记录后按创建时间淘汰最旧记录；清理使用临时文件加原子
  替换，避免中途写坏主文件。
- **R65** 历史查询支持按时间倒序分页，并按路径子串、decision/status 和时间范围筛选。摘要接口不得
  返回可选保存的 diff 正文。
- **R66** 支持删除单条、清空全部和导出全部历史 JSON。删除和清空必须由明确的前端确认对话触发。
- **R67** 单条结果可导出 JSON 和 Markdown。Markdown 至少包含合并就绪度、风险、摘要、理由、意见
  表、覆盖情况、传统模型限制、调用模型与耗时，便于直接用于实验报告和功能验证。

### 11. 前端体验与视觉设计

- **R68** 前端是实际可用的审查工作台，不是营销落地页。一级导航仅包含 `Review` 和 `History`。
- **R69** Review 桌面布局采用稳定三栏：左侧修改文件列表，中间 diff，右侧审查结果。顶部状态栏
  展示仓库名、分支、修改数、服务/LLM 状态、刷新和审查操作。
- **R70** 三栏使用响应式约束和最小宽度，动态结果不得导致布局跳动。窄屏改为“文件 / Diff / 结果”
  三个页签，不把三栏压成不可读窄列。
- **R71** 视觉方向为浅色中性工程工具。青绿用于 MERGE/健康状态，朱红用于 REJECT/blocker，琥珀用于
  warning/degraded；不能只靠颜色传递状态，必须同时有文本和图标。
- **R72** diff 使用清晰的等宽字体和固定行号列；一般界面使用有辨识度且可本地回退的非默认字体。
  不使用负字距，不按 viewport 宽度缩放字号。
- **R73** 结果区首先展示合并就绪度、风险和摘要，然后是意见列表，再是覆盖与上下文、传统 ML 参考、
  调用详情。主结果不能与 ML 参考混在同一视觉层级。
- **R74** 每条意见展示 severity 图标、文件和行号、问题说明与建议；点击意见应滚动并高亮对应 diff
  hunk。没有有效意见时显示“未发现可定位问题”，但仍展示总体判断和覆盖率。
- **R75** 审查前显示预检摘要；审查中显示当前批次和总批次；审查后显示缓存、token、推理时间、覆盖率
  和未审查项。不得用可点击但无功能的装饰控件。
- **R76** History 页使用紧凑表格或列表，支持筛选、分页、展开详情、删除、清空和导出。不得使用卡片
  嵌套卡片。
- **R77** 所有错误、空状态、加载状态、degraded 状态和 stale 状态必须有明确可执行提示。敏感内容
  阻止时列出文件和规则，但不得在错误信息中回显完整密钥。
- **R78** 前端不提供自动编辑、补丁应用或源码保存功能。可以选择文本和下载导出，但不需要“复制建议”
  专用功能。

### 12. 性能、鲁棒性与可观测性

- **R79** 不含 LLM 的 workspace 和单文件 diff API 在课程项目规模的普通仓库中应保持交互式响应；
  对昂贵检索使用快照哈希缓存，避免每次 UI 切换重复扫描全仓库。
- **R80** 服务不得一次性将整个大仓库读入内存。Git 查询、文本扫描和 JSONL 查询应流式或受硬预算约束，
  适配约 7.6 GB WSL2 内存环境。
- **R81** 每次审查记录 request ID、模型 ID、profile、批次数、缓存命中、prompt/completion token、各阶段
  耗时和总耗时。不得记录 API key 或疑似凭据正文。
- **R82** LLM 调用沿用有限重试和指数退避。浏览器请求断开不应损坏历史文件；只有形成可校验最终结果后
  才写成功历史，失败记录可保存状态和诊断但不能伪装为有效审查。
- **R83** 所有内部时间使用 UTC ISO 8601；前端按浏览器本地时区展示。
- **R84** Python 是完整验证路径。其他文本语言允许 LLM 降级审查，但界面和导出必须明确标注“未经本
  项目前序实验验证”，且不展示传统 ML 结果。

### 13. 指导书需求映射与验收

- **获取当前编辑文件**：修改列表当前选中文件即当前文件，可单独预览和审查。
- **获取代码修改内容**：读取工作树相对 `HEAD` 的 staged、unstaged 和未跟踪文本修改，并展示 diff。
- **调用代码审查模型**：浏览器通过 FastAPI HTTP 接口调用 LLM 主模型和实验二传统模型参考。
- **展示 Merge Prediction**：以更准确的“合并就绪度”展示 MERGE/REJECT、风险、理由和语义限制。
- **展示自动生成的代码审查意见**：展示最多 5 条可定位、分级、具体且可操作的逐 hunk 意见。
- **查看历史代码审查记录**：History 页提供筛选、详情、删除、清空和导出。
- **模型状态检测**：`/health` 和顶部状态栏展示 LLM、ML、仓库与历史状态。
- **结果可视化**：提供状态、风险等级、diff 高亮、意见定位、推理时间、token 和覆盖信息。
- **前后端通信**：Web 前端只通过版本化 HTTP API 获取工作区和模型结果。
- **功能测试**：使用 FastAPI 高层接缝和 Playwright 核心流程完成验证。

验收时至少演示：启动并打开页面、修改文件发现、当前文件 diff、当前文件审查、全部修改审查、
MERGE/REJECT 两类结果、逐行意见定位、四个传统模型参考、结果过期、敏感内容阻止、历史查询、
Markdown/JSON 导出、LLM 不可用时的降级状态，以及桌面和窄屏布局。

## Implementation Decisions

本节是后续开发的内部实现约束；在不改变 Requirements 外部行为的前提下，普通辅助函数命名可调整。

### 1. 技术路线与目录

- 新建 `Experiment7/`，代码使用 Python 3.12，继续由根级 `uv` 环境管理。
- 后端采用 FastAPI + Uvicorn；数据校验使用 Pydantic；HTTP 内部测试使用 FastAPI `TestClient`。
- 前端采用原生 HTML、CSS 和 JavaScript ES modules，由 FastAPI 静态托管，不引入 React、Node 或打包器。
- 图标使用前端可本地提供的成熟图标集；若新增依赖，资产必须随项目提供，不能让核心 UI 依赖公共 CDN。
- 历史为 JSONL，不引入数据库。
- 预计目录：

  ```text
  Experiment7/
    README.md
    README-zh.md
    src/
      __init__.py
      app.py
      config.py
      schemas.py
      workspace.py
      context_builder.py
      security.py
      review_engine.py
      llm_review.py
      ml_reference.py
      history.py
      export.py
      static/
        index.html
        styles.css
        app.js
    tests/
      test_api.py
      fixtures/
    results/
      history/
    report/
  ```

### 2. 唯一主测试接缝与分层边界

实验七以 **FastAPI HTTP API** 为唯一主测试接缝。测试通过临时 Git 仓库准备 staged、unstaged、
未跟踪、重命名和敏感文件状态，使用 `TestClient` 调用 `/api/v1`，并将 LLM transport 替换为确定性桩。
从该接缝验证工作区发现、预检、审查、过期、历史和导出的完整外部行为。

内部保持以下薄边界，以便替换真实 I/O，但不为每个边界建立重复的集成套件：

| 边界                  | 责任                                                | 复用/新增                         |
| --------------------- | --------------------------------------------------- | --------------------------------- |
| `WorkspaceRepository` | 在锁定根目录内生成 Git 快照、diff 与 hunk           | 新增                              |
| `ContextBuilder`      | 从快照构建确定性本地上下文与分批计划                | 改造实验六思想                    |
| `ReviewModel`         | `review(batch, profile)` 返回经 schema 校验的批结果 | 复用实验四调用层，新增联合 Prompt |
| `MLReference`         | 本地 Python diff 到实验二四模型参考结果             | 复用实验二/五资产                 |
| `HistoryStore`        | JSONL 追加、查询、淘汰、删除和导出                  | 新增                              |
| FastAPI `/api/v1`     | 组合上述边界并定义唯一外部契约                      | 新增主接缝                        |

依赖通过 FastAPI app factory 注入。生产环境注入真实 Git/LLM/ML/历史实现，测试注入临时仓库和桩模型。

### 3. 核心数据结构

- `FileChange`：`path`、`old_path`、`status`、`language`、`additions`、`deletions`、`binary`、
  `reviewable`、`excluded_reason`、`hunks`。
- `DiffHunk`：`header`、`old_start/count`、`new_start/count`、`lines`、`changed_new_lines`。
- `WorkspaceSnapshot`：`repo_root_id`（不可暴露完整敏感路径时使用名称+哈希）、`branch`、`head_sha`、
  `scope`、`files`、`normalized_diff`、`snapshot_hash`、`created_at`。
- `PreflightReport`：`snapshot_hash`、`included_items`、`excluded_items`、`context_sources`、`char_count`、
  `batch_plan`、`unreviewed_items`、`blocked`。
- `Finding`：对应 R36/R39/R40 的受控结构。
- `BatchReview`：批次 ID、覆盖项、decision、risk、reasoning、findings、解析警告、usage 和 latency。
- `ReviewResult`：对应 R58，并含 schema version 和历史持久化字段。

Pydantic 对所有进入 API 或来自 LLM 的结构执行枚举、长度和字段校验。原始 LLM 文本不进入普通成功响应；
必要诊断仅保存在受限日志中且不含完整敏感上下文。

### 4. Git 快照算法

1. 使用结构化 Git 命令获取 porcelain v2 状态，避免解析面向人的彩色输出。
2. 对 tracked 文件生成 `HEAD -> worktree` 的统一 diff，从而自然合并 staged 与 unstaged 状态。
3. 对未跟踪文件先验证常规文件、非符号链接、文本、大小和敏感规则，再构造 `/dev/null -> file` diff。
4. 规范化 diff 的路径分隔、换行和文件排序；保留 hunk 元信息，不保留时间戳等不稳定字段。
5. `snapshot_hash = sha256(schema_version + head_sha + scope + sorted normalized diffs)`。
6. API 发起审查前重新计算并与 `expected_snapshot_hash` 比较，形成乐观并发控制。

不通过 shell 拼接用户路径。所有 Git 调用使用参数数组，并设置仓库 cwd、超时和输出上限。

### 5. 上下文、分批与意见定位

上下文优先级固定为：diff > enclosing block/文件 > 相关测试 > 符号引用 > 项目约定。每类有独立字符预算，
总体预算按模型上下文限制折算。分批使用 first-fit 的稳定变体：先按文件排序，将完整文件审查单元放入当前
批次；放不下时开始新批；单文件过大时按 hunk 切为不可再分单元。最多保留前 4 批，其余进入覆盖缺口。

LLM 接收带稳定锚点的 hunk，例如 `FILE:path HUNK:H2 NEW_LINES:40-58`。返回 finding 后，解析器验证 path
和 line 是否属于对应批次的 changed-new-line 集合。重复意见按规范化 `(path, line, category, message)`
去重；汇总阶段只接收批结果，不负责发明新位置，因此汇总不得新增 finding，只能选择、去重和排序。

### 6. LLM Prompt 与汇总不变量

- `strict` system 指令继承 P5 的批判性 maintainer 角色和“先列 blocking 缺陷再判断”流程。
- `fast` 使用简短 P1 风格，但输出 schema 与 strict 完全相同。
- 用户消息明确这是本地 Git 修改，不提供平台最终状态，要求判断“是否具备进入提交/PR 的条件”。
- 单批 Prompt 同时要求 decision、risk、summary、reasoning、blocking defects 和 findings。
- 汇总优先由确定性代码执行 R47 不变量，再可调用 LLM 压缩 summary/reasoning 和选择前 5 条意见。
  LLM 汇总若违反不变量，以确定性结果为准。

### 7. ML 本地适配

ML 适配只使用 `pre-review` scaler 和模型。Python patch 直接复用实验二 `extract_code_features_for_pr`；
统计特征从本地 diff 构造；commit message 使用 `HEAD` 最近提交信息作为有限代理；PR title/body 留空，并记录
缺失。TF-IDF vectorizer 严格复用实验五的固定词表和实验二人类语料 IDF 恢复方式。

本地数据明显偏离训练分布，因此不设计四模型投票器、不调整阈值、不将概率转成风险等级，也不让 ML 失败
改变 HTTP 主审查状态。该区域的价值是证明实验二产物完成软件集成，并展示模型迁移限制。

### 8. 历史一致性

JSONL 每行一个带 `schema_version` 的完整记录。追加前获得进程内锁；写入后若超过 200 条，流式读取有效行、
保留最新记录并通过同目录临时文件原子替换。删除同样使用重写和原子替换。发现损坏行时查询跳过并报告
`storage_warning`，不得使整个历史页不可用。

历史详情的 stale 状态不持久化为永真字段，而是在读取时用当前对应 scope 哈希与记录哈希比较。仓库无修改或
记录涉及文件已消失时同样视为 stale。

### 9. 前端状态流

前端顶层状态为：`booting -> ready|degraded|fatal`。Review 子状态为：`idle -> preflighting -> ready_to_review
-> reviewing -> complete|incomplete|error`。任何 workspace 刷新都重新计算选择和 stale 状态。

浏览器通过 `fetch` 调用 API。非 2xx 响应统一读取 R59 错误结构并在相关区域显示，不使用原生 alert 承担
普通错误。删除/清空使用模态确认。结果中的文件与行号点击后切到 Review 页、选中文件并滚动到对应 hunk。

## Testing Decisions

### 测试原则

好测试验证用户可观察的行为和稳定契约，不断言内部辅助函数被调用多少次、具体类名、CSS 实现或 Prompt 的
逐字内容。除安全纯函数和 schema 不变量外，优先从最高的 FastAPI API 接缝测试，避免 Git、上下文、模型、
历史各自建立重复而脆弱的测试矩阵。真实 DeepSeek 调用只做显式付费冒烟，不进入默认自动测试。

### 主 API 接缝测试

使用 pytest、临时 Git 仓库、FastAPI `TestClient` 和确定性 LLM 桩覆盖：

- 健康仓库、无 API key degraded、缺失 ML 资产等 `/health` 外部状态。
- staged、unstaged、混合、未跟踪、新增、删除、重命名、无修改和二进制文件的 workspace/diff 行为。
- 当前文件与全部修改 preflight 的包含项、排除项、敏感阻止、批次数和快照哈希稳定性。
- 路径遍历、绝对路径、符号链接逃逸和未知文件均被拒绝且不泄露仓库外内容。
- 正常 MERGE、正常 REJECT、blocker 强制 REJECT、无法定位 finding 被丢弃、JSON 修复失败和部分批次失败。
- 大改动最多 4+1 调用、未覆盖项可见、单批不汇总、汇总不变量不被模型推翻。
- 预检后修改文件导致 `/reviews` 返回 409；历史详情随后能识别 stale。
- Python 返回四模型参考及限制说明；非 Python 不运行 ML；ML 失败不影响 LLM 主结果。
- JSONL 写入、200 条淘汰、筛选、分页、删除、清空、损坏行降级和 JSON/Markdown 导出。
- 默认历史不含 diff；显式 include 后仅保存预算内 diff。

### 少量边界测试

以下风险高且输入组合密集，可在 API 测试之外补纯函数参数化测试：

- 敏感路径与凭据模式的真阳性/常见假阳性，确保错误不回显秘密全文。
- LLM schema 枚举、长度、finding 行号范围和 R38/R47 不变量。
- 统一 diff hunk 行号解析，特别是纯删除 hunk 和 `\ No newline at end of file`。

### 浏览器测试

使用 Playwright 连接桩后端或确定性测试应用，仅覆盖核心用户旅程：

1. 加载工作台，选择修改文件并查看 diff。
2. 预检并发起当前文件审查，查看合并就绪度和定位意见。
3. 点击意见跳转到对应 hunk。
4. 查看历史、筛选、打开详情和导出。
5. 验证 no changes、degraded、sensitive blocked、incomplete 和 stale 状态。
6. 在桌面与移动 viewport 截图，检查三栏/页签切换、文本不溢出、无重叠和状态颜色可辨识。

### 真实环境冒烟

- 在一个小型临时/示例 Git 仓库中执行一次真实 `strict` 当前文件审查，确认 DeepSeek 认证、JSON 模式、
  usage、缓存和历史落盘。
- 在本项目自身修改上执行一次预检和一次可控规模审查，记录响应时间、批次数、token 和覆盖率，作为实验报告
  的运行效果与性能分析数据。
- 真实冒烟必须由显式命令触发，默认测试不访问网络、不消耗 API 额度。

### 测试先例

- 实验六 `Experiment6/src/tests.py` 已采用“外部行为 + 确定性桩”的离线测试风格，可复用其 LLM 桩和
  解析兼容思路。
- 实验四 `llm_client.py` 提供缓存、重试、usage 和 JSON 模式先例。
- 实验五 `ml_features.py` 提供 TF-IDF bit-level 重现与模型只 transform 的正确性先例。

## Out of Scope

- 不开发、打包或安装 VS Code Extension，不调用 VS Code Extension API。
- 不在浏览器中自动感知 VS Code 当前激活标签页；当前文件由 Web 修改列表选择定义。
- 不提供 Web 代码编辑、源码保存、自动修复、补丁应用、撤销或提交功能。
- 不以 GitHub PR URL/编号作为主输入，不抓取 Issue、PR 评论或在线维护流程状态。
- 不宣称“合并就绪度”等价于 GitHub 最终合并结果预测。
- 不训练、微调或重新校准任何模型；不修改实验二权重，不实现跳过的实验三模型。
- 不让传统 ML 参与最终投票、风险评级或自动决策。
- 不对非 Python 语言宣称经过本项目实验验证。
- 不实现向量数据库、语义 RAG、完整跨文件 AST 调用图或 Agent 多工具自治修改。
- 不支持远程多用户部署、身份认证、权限系统、云端历史同步或团队协作。
- 不无限制审查超大修改；默认最多 4 个代码批次和 1 个汇总调用。
- 不默认保存完整源码/diff 到历史，不提供敏感扫描绕过开关。
- 不把 LLM judge 引入在线产品结果；实验六 judge 是离线评价工具，不是用户裁决链的一部分。

## Further Notes

- 本设计有意保留“前端通过 HTTP 调用 Python 模型服务”的原指导书架构。未来若补做 VS Code 插件，
  可直接替换浏览器交互层并复用 `/api/v1`，因此本实验不是一次性演示脚本。
- “Merge Prediction”改称“合并就绪度”是必要的语义校正。报告仍可说明它继承了前序二分类任务和
  MERGE/REJECT 输出，但必须讨论本地输入缺少平台过程信息造成的效度边界。
- 实验六证明 P5 提高 non-merge recall 的同时会增加误拒，因此默认 strict 适合提交前风险分流，
  不适合自动合并。fast profile 作为低延迟对照保留，界面不宣称任一 profile 全面占优。
- 传统模型的本地迁移展示必须诚实呈现缺失特征和训练分布差异。软件集成的完成度不能靠隐藏模型限制来换取。
- 实验报告需将原思考题适配到 Web 场景：代码审查为何应靠近开发环境、HTTP 分离的优势、如何改善响应
  与体验、部署中遇到的 Git/模型/安全问题，以及未来如何替换为 IDE 插件或增加建议应用能力。
- 建议后续实施顺序：API 骨架与临时 Git 接缝 -> workspace/diff -> preflight/security -> LLM 单批审查
  -> 分批汇总 -> ML 参考 -> JSONL 历史/导出 -> 三栏前端 -> Playwright/真实冒烟 -> README 与实验报告。