# AI4SE-实验七

[English](README.md) | 简体中文

## 概述

实验七将课程代码审查流程落地为本地 Web 工作台。服务锁定一个可读的 Git 工作树，以 `HEAD -> worktree` 为边界，统一读取 staged、unstaged、删除、重命名和未跟踪文件，并通过浏览器界面和版本化 FastAPI 服务提供审查能力。DeepSeek 负责主要审查和本地合并就绪度判断，实验二的四个 `pre-review` 分类器提供明确标注的参考信号。

系统输出的核心概念是**本地合并就绪度**：判断当前修改是否已经达到进入提交或 Pull Request 的审查条件。由于本地修改不包含 PR 讨论、reviewer 决策和维护者最终结果，该指标不能解释为 GitHub 最终合并预测。下面的截图展示了实际交付的工作台。

![实验七本地 Web 代码审查工作台](results/image/example.png)

## 目录

- [核心功能](#核心功能)
- [安装](#安装)
- [运行要求](#运行要求)
- [使用方法](#使用方法)
  - [1. 准备环境](#1-准备环境)
  - [2. 启动工作台](#2-启动工作台)
  - [3. 执行审查](#3-执行审查)
  - [4. 运行离线测试](#4-运行离线测试)
  - [5. 运行显式付费冒烟测试](#5-运行显式付费冒烟测试)
- [API](#api)
- [故障排查](#故障排查)
- [限制](#限制)

## 核心功能

- 提供浏览器代码审查流程：桌面端使用三栏布局，移动端使用“文件、Diff、结果”页签。
- 将 staged 和 unstaged Git 状态归一为确定性快照，支持新增、修改、删除、重命名和未跟踪文件。
- 在外发审查前执行路径遍历、符号链接、二进制、敏感路径、凭据模式和文件大小检查。
- 通过快照哈希、可见预检、HTTP 409 乐观并发控制和动态 stale 检测，使审查结果绑定到实际代码状态。
- 支持 DeepSeek `strict` Self-Reflection 与 `fast` 模式，使用结构化 JSON、重试、一次格式修复、用量统计和内容缓存。
- 每次审查最多执行四个代码批次和一个结构化汇总；部分失败或存在未覆盖修改时强制输出 `INCOMPLETE`。
- 集成实验二的 SVM、Random Forest、XGBoost 和 LightGBM `pre-review` 模型，使用 scaler 的 93 列作为唯一特征顺序。
- 使用原子写入的 UTF-8 JSONL 保存历史，限制最近 200 条，支持筛选、分页、删除、清空以及 JSON/Markdown 导出。

## 安装

在仓库根目录同步 Python 3.12 环境，并安装界面测试所需的浏览器：

```bash
uv sync
uv run playwright install chromium
```

如需启用 DeepSeek 主审查，在当前 shell 中配置密钥：

```bash
export DEEPSEEK_API_KEY=<your_key>
```

没有密钥时服务仍可在 degraded 模式启动，工作区、Diff、历史和模型资产状态继续可用，但付费审查请求会被禁用。

## 运行要求

- Python >= 3.12
- `uv`，用于复现实验环境和依赖
- FastAPI、Uvicorn、Pydantic、HTTPX，用于本地服务和 API 测试
- 兼容 OpenAI 接口的客户端，用于调用 DeepSeek
- scikit-learn、XGBoost 和 LightGBM，用于实验二参考模型
- pytest 和 Playwright，用于离线 API 测试与浏览器测试
- 一个可读且至少包含有效 `HEAD` 提交的 Git 工作树
- 实验二的 `pre-review` 模型资产和 scaler，用于 ML 参考信号；资产缺失不会阻断 LLM 审查

系统不需要 Node.js 构建、前端包管理器、CDN 或远程浏览器服务。

## 使用方法

下面所有命令都建议从实验七目录运行：

```bash
cd /home/wzsyh/ai-software-engineer/Experiment7
```

### 1. 准备环境

首次运行时安装项目环境和浏览器：

```bash
uv sync
uv run playwright install chromium
```

### 2. 启动工作台

将服务指向需要审查的 Git 工作树根目录：

```bash
uv run python -m src.app serve --repo /absolute/path/to/git-worktree
```

打开 <http://127.0.0.1:8765>。`--port` 可修改端口；`--host` 可修改监听地址，但非回环地址会触发警告，因为服务能够读取目标仓库源码。

### 3. 执行审查

1. 选择修改文件并查看只读 Diff。
2. 选择 `Strict` 或 `Fast`，审查当前文件或全部修改。
3. 在发送请求前检查预检范围、排除项、字符数、上下文来源和批次计划。
4. 查看合并就绪度、风险、finding、覆盖缺口、四模型参考、token 用量、耗时和缓存状态。
5. 点击 finding 定位并高亮修改后的行，在 History 中筛选、打开、删除或导出记录。

默认不保存 Diff。若请求设置 `include_diff_in_history=true`，仅保存通过安全检查且不超过配置字符预算的 Diff 内容。

### 4. 运行离线测试

默认测试不会产生付费 API 请求：

```bash
uv run pytest tests -q
```

测试使用临时真实 Git 仓库、FastAPI `TestClient`、确定性 LLM 桩和 Python Playwright，覆盖 Git 状态归一、安全阻断、快照冲突、stale 结果、4+1 请求上限、模型变换、历史与导出、桌面/移动布局以及 finding 定位。

### 5. 运行显式付费冒烟测试

只有同时配置 API key 并显式传入确认参数，以下命令才会调用 DeepSeek：

```bash
uv run python -m src.paid_smoke \
  --repo /absolute/path/to/git-worktree \
  --path relative/modified_file.py \
  --confirm-paid
```

结果写入 `results/metrics/paid_smoke.json`，包含 token 用量、耗时、缓存、覆盖率和请求次数。该文件已被 Git 忽略。

## API

所有接口使用 `/api/v1` 前缀。主要接口包括 `/health`、`/workspace`、`/diff`、`/preflight`、`/reviews`、`/progress/{request_id}`、`/history`、`/reviews/{id}`、`/exports/{id}.{format}` 和 `/history/export`。错误统一使用 `{code, message, details, request_id}` 结构。

## 故障排查

- `No module named src`：必须从 `Experiment7/` 目录运行命令。
- `Invalid Git worktree`：将 `--repo` 指向工作树根目录，并确认 `HEAD` 可以解析为提交。
- 缺少 `DEEPSEEK_API_KEY`：设置变量并重启服务；不要将密钥放入前端、URL 或历史记录。
- 审查操作被禁用：检查 LLM 健康状态、是否存在可审查修改以及预检排除项。
- HTTP 409：预检后快照发生变化，刷新工作区并重新预检。
- 安全排除：移除凭据，或将二进制和生成文件移出审查范围；敏感检查不能绕过。
- Playwright 缺少浏览器：运行 `uv run playwright install chromium`。
- ML 区域报错：确认实验二 `pre-review` 模型、scaler、特征表以及实验一 PR 语料存在。ML 失败不会阻断 LLM 审查。

## 限制

- Python 是完整验证路径；其他文本语言只提供降级 LLM 审查，不提供实验二预测。
- 本地工作树缺少 PR 标题/正文、讨论、reviewer 决策和维护者最终结果，因此结果不能解释为平台合并预测。
- 相关测试和符号引用使用确定性词法检索，不构建语义索引或完整调用图。
- 服务面向本机单用户，不提供远程认证、代码编辑、补丁应用、提交操作或团队同步。
