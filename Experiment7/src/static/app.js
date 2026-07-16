const state = {
  health: null, workspace: null, selectedPath: null, selectedDiff: null,
  profile: "strict", review: null, historyPage: 1, historyPages: 1,
};

const byId = (id) => document.getElementById(id);
const escapeHtml = (value = "") => String(value).replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);

async function api(path, options = {}) {
  const response = await fetch(`/api/v1${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: response.statusText }));
    throw Object.assign(new Error(error.message || "请求失败"), { payload: error, status: response.status });
  }
  return response.json();
}

function message(text, type = "") {
  const node = byId("global-message");
  node.textContent = text;
  node.className = `message ${type}`;
  if (!text) node.classList.add("hidden");
}

function setMobilePanel(panelId) {
  document.querySelectorAll(".mobile-tabs button").forEach((button) => button.classList.toggle("active", button.dataset.panel === panelId));
  document.querySelectorAll(".review-grid .panel").forEach((panel) => panel.classList.toggle("mobile-active", panel.id === panelId));
}

function renderFiles() {
  const files = state.workspace?.files || [];
  const reviewable = files.filter((file) => file.reviewable);
  byId("reviewable-count").textContent = `${reviewable.length} 可审查`;
  if (!files.length) {
    byId("file-list").innerHTML = '<div class="empty-state">工作区没有相对 HEAD 的修改</div>';
    return;
  }
  const status = { added: "A", modified: "M", deleted: "D", renamed: "R" };
  byId("file-list").innerHTML = files.map((file) => `
    <button class="file-item ${file.path === state.selectedPath ? "active" : ""} ${file.reviewable ? "" : "excluded"}" data-path="${escapeHtml(file.path)}">
      <span class="file-status">${file.reviewable ? status[file.status] || "M" : "!"}</span>
      <span><span class="file-name">${escapeHtml(file.path)}</span><span class="file-meta">${file.reviewable ? escapeHtml(file.language) : escapeHtml(file.excluded_reason)}</span></span>
      <span class="file-meta"><span class="plus">+${file.additions}</span> <span class="minus">−${file.deletions}</span></span>
    </button>`).join("");
  document.querySelectorAll(".file-item").forEach((button) => button.addEventListener("click", () => selectFile(button.dataset.path)));
}

function renderDiff() {
  const file = state.selectedDiff?.files?.[0];
  byId("diff-title").textContent = file?.path || "Diff";
  byId("diff-stats").textContent = file ? `+${file.additions} −${file.deletions}` : "+0 −0";
  if (!file?.reviewable) {
    byId("diff-content").innerHTML = `<div class="diff-empty">该文件已被安全规则排除：${escapeHtml(file?.excluded_reason || "不可审查")}</div>`;
    return;
  }
  if (!file.diff) {
    byId("diff-content").innerHTML = '<div class="diff-empty">选择一个修改文件查看只读 Diff</div>';
    return;
  }
  let oldLine = 0, newLine = 0;
  const html = file.diff.split("\n").map((line) => {
    const match = line.match(/^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/);
    if (match) { oldLine = Number(match[1]); newLine = Number(match[2]); return `<div class="diff-line hunk">${escapeHtml(line)}</div>`; }
    if (line.startsWith("+++") || line.startsWith("---") || line.startsWith("diff ") || line.startsWith("index ")) return `<div class="diff-line hunk">${escapeHtml(line)}</div>`;
    let kind = "context", shownOld = oldLine, shownNew = newLine, marker = " ";
    if (line.startsWith("+")) { kind = "add"; shownOld = ""; marker = "+"; newLine += 1; }
    else if (line.startsWith("-")) { kind = "remove"; shownNew = ""; marker = "−"; oldLine += 1; }
    else if (!line.startsWith("\\")) { oldLine += 1; newLine += 1; }
    const lineId = kind === "add" ? `diff-line-${shownNew}` : "";
    return `<div class="diff-line ${kind}" ${lineId ? `id="${lineId}"` : ""}><span class="line-number">${shownOld}</span><span class="line-number">${shownNew}</span><span class="marker">${marker}</span><code>${escapeHtml(line.slice(kind === "context" ? 1 : 1))}</code></div>`;
  }).join("");
  byId("diff-content").innerHTML = html;
}

async function selectFile(path, switchPanel = false) {
  state.selectedPath = path;
  renderFiles();
  try { state.selectedDiff = await api(`/diff?path=${encodeURIComponent(path)}`); renderDiff(); }
  catch (error) { message(error.message, "error"); }
  if (switchPanel) setMobilePanel("diff-panel");
}

function emptyResult(text = "选择审查范围，预检确认后发起审查") {
  byId("result-content").innerHTML = `<div class="empty-state">${escapeHtml(text)}</div>`;
  byId("result-state").textContent = "待审查";
}

function renderPreflight(report) {
  byId("result-state").textContent = report.blocked ? "安全阻止" : "预检完成";
  byId("result-content").innerHTML = `
    <div class="result-hero ${report.blocked ? "reject" : ""}"><div class="decision"><strong>${report.blocked ? "存在敏感内容" : "准备送审"}</strong><span class="badge">${report.batch_plan.length} 批次</span></div><p>${report.char_count.toLocaleString()} 字符，${report.included_items.length} 个文件</p></div>
    <section class="result-section"><h3>发送范围</h3>${report.included_items.map((item) => `<p><code>${escapeHtml(item.path)}</code> · ${item.characters} 字符</p>`).join("") || "<p>无可发送文件</p>"}</section>
    <section class="result-section"><h3>排除与覆盖</h3>${report.excluded_items.map((item) => `<p><strong>${escapeHtml(item.path)}</strong> · ${escapeHtml(item.reason)}</p>`).join("") || "<p>无排除项</p>"}${report.unreviewed_items.length ? `<p>未覆盖：${report.unreviewed_items.map(escapeHtml).join(", ")}</p>` : ""}</section>`;
}

function renderReview(result) {
  state.review = result;
  const incomplete = result.status === "INCOMPLETE";
  const heroClass = incomplete ? "incomplete" : result.decision === "REJECT" ? "reject" : "";
  byId("result-state").textContent = result.stale ? "已过期" : result.status;
  const findings = result.findings.map((finding) => `
    <button class="finding ${finding.severity}" data-path="${escapeHtml(finding.path)}" data-line="${finding.line || ""}">
      <span class="finding-path">${escapeHtml(finding.path)}${finding.line ? `:${finding.line}` : ""} · ${escapeHtml(finding.category)}</span>
      <strong>${escapeHtml(finding.message)}</strong><small>${escapeHtml(finding.suggestion)}</small>
    </button>`).join("") || '<p>未发现可定位问题</p>';
  const models = result.ml_reference.models || [];
  byId("result-content").innerHTML = `
    <div class="result-hero ${heroClass}"><div class="decision"><strong>${incomplete ? "INCOMPLETE" : result.decision}</strong><span class="badge">${escapeHtml(result.risk_level)} risk</span></div><p>${escapeHtml(result.summary)}</p>${result.stale ? "<p><strong>结果已过期，请刷新后重新审查。</strong></p>" : ""}</div>
    <section class="result-section"><h3>审查意见</h3>${findings}</section>
    <details open><summary>覆盖与上下文</summary><div class="metric-grid"><div class="metric"><span>批次</span><strong>${result.coverage.reviewed_batches}/${result.coverage.planned_batches}</strong></div><div class="metric"><span>未审查</span><strong>${result.coverage.unreviewed_items.length}</strong></div></div>${result.coverage.warnings.map((item) => `<p>${escapeHtml(item)}</p>`).join("")}</details>
    <details><summary>传统模型参考</summary><p>${escapeHtml(result.ml_reference.disclaimer || result.ml_reference.reason || "")}</p><div class="metric-grid">${models.map((model) => `<div class="metric"><span>${escapeHtml(model.model)}</span><strong>${model.decision}${model.merge_probability == null ? "" : ` ${(model.merge_probability * 100).toFixed(0)}%`}</strong></div>`).join("")}</div></details>
    <details><summary>调用详情</summary><div class="metric-grid"><div class="metric"><span>Token</span><strong>${result.usage.total_tokens || 0}</strong></div><div class="metric"><span>模型耗时</span><strong>${result.latency.model_seconds.toFixed(2)}s</strong></div><div class="metric"><span>缓存</span><strong>${result.cached ? "命中" : "未命中"}</strong></div><div class="metric"><span>模型</span><strong>${escapeHtml(result.model)}</strong></div></div><p>${escapeHtml(result.reasoning)}</p></details>
    <section class="result-section"><a href="/api/v1/exports/${result.id}.md">Markdown</a> · <a href="/api/v1/exports/${result.id}.json">JSON</a></section>`;
  document.querySelectorAll(".finding").forEach((button) => button.addEventListener("click", () => locateFinding(button.dataset.path, Number(button.dataset.line))));
}

async function locateFinding(path, line) {
  await selectFile(path, true);
  const target = byId(`diff-line-${line}`);
  if (target) { target.scrollIntoView({ behavior: "smooth", block: "center" }); target.classList.remove("highlight"); requestAnimationFrame(() => target.classList.add("highlight")); }
}

async function refreshWorkspace() {
  message("");
  try {
    [state.health, state.workspace] = await Promise.all([api("/health"), api("/workspace")]);
    const service = byId("service-state");
    service.className = `service-state ${state.health.status === "ok" ? "ok" : ""}`;
    service.innerHTML = `<span class="state-dot"></span><span>${state.health.status === "ok" ? "服务正常" : "LLM 未配置"}</span>`;
    byId("repo-meta").textContent = state.workspace.repository.name;
    byId("branch-name").textContent = state.workspace.branch;
    byId("change-count").textContent = `${state.workspace.files.length} 个修改`;
    const paths = state.workspace.files.map((file) => file.path);
    if (!paths.includes(state.selectedPath)) state.selectedPath = state.workspace.files.find((file) => file.reviewable)?.path || paths[0] || null;
    renderFiles();
    if (state.selectedPath) await selectFile(state.selectedPath); else { state.selectedDiff = null; renderDiff(); emptyResult("工作区没有修改"); }
    const canReview = state.health.llm.ready && state.workspace.files.some((file) => file.reviewable);
    byId("review-file-button").disabled = !canReview || !state.selectedDiff?.files?.[0]?.reviewable;
    byId("review-all-button").disabled = !canReview;
    if (!state.health.llm.ready) message("DeepSeek 未配置：仍可浏览 Diff、传统资产状态与历史；主审查已禁用。", "");
  } catch (error) { byId("service-state").className = "service-state error"; message(error.message, "error"); }
}

async function runReview(scope) {
  const path = scope === "file" ? state.selectedPath : null;
  const observed = scope === "file" ? state.selectedDiff?.snapshot_hash : state.workspace?.snapshot_hash;
  const payload = { scope, path, profile: state.profile, include_diff_in_history: false, expected_snapshot_hash: observed };
  const buttons = [byId("review-file-button"), byId("review-all-button")];
  buttons.forEach((button) => button.disabled = true);
  try {
    byId("result-state").textContent = "预检中";
    const report = await api("/preflight", { method: "POST", body: JSON.stringify(payload) });
    renderPreflight(report); setMobilePanel("result-panel");
    if (report.blocked || !report.included_items.length) { message("安全预检阻止了审查，请移除敏感内容后刷新。", "error"); return; }
    byId("result-state").textContent = "审查中";
    message(`正在审查 ${report.batch_plan.length} 个批次，请勿重复提交。`);
    const result = await api("/reviews", { method: "POST", headers: { "X-Request-ID": crypto.randomUUID() }, body: JSON.stringify(payload) });
    renderReview(result); message(result.stale ? "审查期间工作区发生变化，结果已标记过期。" : "审查完成。", result.stale ? "" : "success");
  } catch (error) {
    if (error.status === 409) message("工作区已变化，请刷新并重新预检。", "error");
    else message(error.message, "error");
    byId("result-state").textContent = "失败";
  } finally { await refreshWorkspace(); }
}

async function loadHistory() {
  const path = byId("history-path").value.trim(); const status = byId("history-status").value;
  try {
    const data = await api(`/history?page=${state.historyPage}&page_size=20&path=${encodeURIComponent(path)}&status=${encodeURIComponent(status)}`);
    state.historyPages = Math.max(1, Math.ceil(data.total / data.page_size)); state.historyPage = Math.min(state.historyPage, state.historyPages);
    byId("history-total").textContent = `${data.total} 条记录`; byId("history-page").textContent = `${state.historyPage} / ${state.historyPages}`;
    byId("history-prev").disabled = state.historyPage <= 1; byId("history-next").disabled = state.historyPage >= state.historyPages;
    byId("history-list").innerHTML = data.items.length ? data.items.map((item) => `
      <article class="history-row"><time>${new Date(item.created_at).toLocaleString()}</time><span class="badge">${item.status === "INCOMPLETE" ? item.status : item.decision}</span><button data-open-review="${item.id}">${escapeHtml(item.summary)}</button><small class="history-files">${escapeHtml((item.files || []).join(", "))}</small><span class="row-actions"><a href="/api/v1/exports/${item.id}.md">MD</a><button data-delete-review="${item.id}">删除</button></span></article>`).join("") : '<div class="empty-state">暂无审查历史</div>';
    document.querySelectorAll("[data-open-review]").forEach((button) => button.addEventListener("click", () => openHistory(button.dataset.openReview)));
    document.querySelectorAll("[data-delete-review]").forEach((button) => button.addEventListener("click", () => confirmAction("删除记录", "该操作无法撤销。", async () => { await api(`/history/${button.dataset.deleteReview}`, { method: "DELETE" }); loadHistory(); })));
  } catch (error) { message(error.message, "error"); }
}

async function openHistory(id) {
  try { const record = await api(`/reviews/${id}`); document.querySelector('[data-view="review"]').click(); renderReview(record); setMobilePanel("result-panel"); }
  catch (error) { message(error.message, "error"); }
}

function confirmAction(title, text, action) {
  const dialog = byId("confirm-dialog"); byId("confirm-title").textContent = title; byId("confirm-message").textContent = text; dialog.showModal();
  dialog.addEventListener("close", async function handler() { dialog.removeEventListener("close", handler); if (dialog.returnValue === "confirm") await action(); });
}

document.querySelectorAll(".nav-item").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("active", item === button));
  document.querySelectorAll(".view").forEach((view) => view.classList.toggle("active", view.id === `${button.dataset.view}-view`));
  if (button.dataset.view === "history") loadHistory();
}));
document.querySelectorAll(".profile-control button").forEach((button) => button.addEventListener("click", () => { state.profile = button.dataset.profile; document.querySelectorAll(".profile-control button").forEach((item) => item.classList.toggle("active", item === button)); }));
document.querySelectorAll(".mobile-tabs button").forEach((button) => button.addEventListener("click", () => setMobilePanel(button.dataset.panel)));
byId("refresh-button").addEventListener("click", refreshWorkspace);
byId("review-file-button").addEventListener("click", () => runReview("file"));
byId("review-all-button").addEventListener("click", () => runReview("workspace"));
byId("history-path").addEventListener("input", () => { state.historyPage = 1; loadHistory(); });
byId("history-status").addEventListener("change", () => { state.historyPage = 1; loadHistory(); });
byId("history-prev").addEventListener("click", () => { state.historyPage -= 1; loadHistory(); });
byId("history-next").addEventListener("click", () => { state.historyPage += 1; loadHistory(); });
byId("export-history").addEventListener("click", () => { location.href = "/api/v1/history/export"; });
byId("clear-history").addEventListener("click", () => confirmAction("清空历史", "将永久删除全部本地审查记录。", async () => { await api("/history", { method: "DELETE" }); loadHistory(); }));

emptyResult();
refreshWorkspace();