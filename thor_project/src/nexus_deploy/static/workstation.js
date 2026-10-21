// The backend owns admission and navigation state; viewing never starts a task.
import { configureAlignmentLink } from "./alignment-link.js";
import { freshCsrfHeaders, sessionErrorMessage } from "./ui-session.js?v=20261008-csrf-refresh";
const panel = document.getElementById("workstation-panel");
const handoff = document.getElementById("workstation-handoff");
const admissionAction = document.getElementById("workstation-admission-action");
const status = document.getElementById("workstation-status");
const help = document.getElementById("workstation-help");
const robotState = document.getElementById("workstation-robot-state");
const resetLink = document.getElementById("workstation-reset-link");
const currentTask = document.getElementById("workstation-current-task");
const switchStatus = document.getElementById("workstation-switch-status");
const records = document.getElementById("workstation-records");
const recordsTasks = document.getElementById("records-tasks");
const recordsStatus = document.getElementById("records-status");
let bootstrap = null;
let station = null;
let updating = false;
let pendingRefresh = null;
let updateError = "";

async function request(path, options = {}) {
  const headers = await freshCsrfHeaders(options);
  headers.set("Content-Type", "application/json");
  const response = await fetch(`/api${path}`, {
    credentials: "same-origin", cache: "no-store", ...options,
    headers,
  });
  const value = await response.json();
  if (!response.ok) throw new Error(sessionErrorMessage(response, value) || `${value.reason_code || "REQUEST_FAILED"}: ${value.error || response.status}`);
  return value;
}

function showError(message) {
  const target = status || recordsStatus;
  target.textContent = message;
  target.classList.add("is-error");
}

function currentDeploymentReason(current) {
  if (current?.reason) return current.reason;
  if (current?.status === "model_stopped" || current?.model_status === "stopped") {
    return "当前模型已关闭；请重新加载模型并保存应用配置";
  }
  return current ? "当前已应用配置尚未待命，请核实模型与机器人状态"
    : station.active_deployment_id ? "当前已应用配置不可用；请打开运行配置核实模型状态"
      : "尚无当前已应用配置，请先保存并应用配置";
}

function profileReadinessDetails() {
  const fallback = "调度 profile 尚未匹配当前已应用配置，或包裹检测、模型健康未通过。";
  if (!station.profiles.length) {
    return "先在运行配置保存并应用含包裹检测的配置，再配置匹配当前部署的调度 profile。";
  }
  const fieldLabels = {
    server_mode: "模型服务模式（server_mode）",
    "safety.safety_profile_id": "安全配置（safety.safety_profile_id）",
  };
  const valueText = (field, value) => {
    if (value === undefined || value === null) return "未设置";
    if (field === "server_mode") {
      const modes = { real: "加载真实模型（real）", reuse: "复用已加载服务（reuse）", mock: "模拟服务（mock）" };
      if (Object.hasOwn(modes, value)) return modes[value];
    }
    return typeof value === "object" ? JSON.stringify(value) : String(value);
  };
  return station.profiles.filter((item) => !item.ready).map((item) => {
    const reasons = [...new Set([item.reason, item.error].filter((value) => typeof value === "string" && value.trim()))];
    const mismatches = Array.isArray(item.config_mismatches) ? item.config_mismatches : [];
    const differences = mismatches.filter((item) => item && typeof item.field === "string").map((item) =>
      `${Object.hasOwn(fieldLabels, item.field) ? fieldLabels[item.field] : item.field}：当前部署=${valueText(item.field, item.deployment_value)}，调度 profile=${valueText(item.field, item.profile_value)}`);
    const details = [...reasons, ...differences];
    return `调度 profile ${item.profile || "（未命名）"} 未就绪：${details.length ? details.join("；") : fallback}`;
  }).join(" ");
}

function syncControls() {
  if (!panel || !station) return;
  const ready = station.profiles.some((item) => item.ready);
  const current = station.console_deployments.find((item) => item.id === station.active_deployment_id);
  const standby = current?.phase === "standby";
  const resetRequired = station.robot_gate?.required === true;
  handoff.textContent = station.mode === "automatic" ? "人工接管" : "交还调度";
  handoff.disabled = updating || station.persistence_error || Boolean(station.switching_deployment_id) ||
    station.active_tasks.length > 0 || !standby || (station.mode === "debug" && (!ready || resetRequired));
  const handoffReasons = [];
  if (updating) handoffReasons.push("正在更新工作站状态，请稍候");
  if (station.persistence_error) handoffReasons.push(`工作站状态持久化失败：${typeof station.persistence_error === "string" ? station.persistence_error : station.reason || "请检查持久化存储后重试"}`);
  if (station.switching_deployment_id) handoffReasons.push("正在应用配置，请等待完成后再交接");
  if (station.active_tasks.length) handoffReasons.push("先结束当前任务并完成必要复位，再交接新任务启动权");
  if (!standby) handoffReasons.push(currentDeploymentReason(current));
  if (station.mode === "debug" && resetRequired) handoffReasons.push(`机器人必须成功复位后才能交还调度：${station.robot_gate.reason || "请先完成机器人复位"}`);
  if (station.mode === "debug" && !ready) handoffReasons.push(profileReadinessDetails());
  handoff.title = handoffReasons.length ? handoffReasons.join("；")
    : station.mode === "automatic" ? "切换到人工启动；不会中断当前机器人动作"
      : "通过预检后允许调度 API 启动新任务；不会立即启动任务";
  const readyCount = station.profiles.filter((item) => item.ready).length;
  const profileDetails = profileReadinessDetails();
  help.textContent = "此按钮只控制下一次任务能否启动，不会暂停、恢复或结束当前任务。" +
    (handoffReasons.length ? ` 交接不可用：${handoffReasons.join("；")}` :
      `配置匹配模板 ${readyCount}/${station.profiles.length}；交接时仍检查模型实时健康；交接前请检测机器人并确认空闲。交接只改变新任务启动权，不会执行先前被拒绝的请求。`) +
    (profileDetails && !handoffReasons.includes(profileDetails) ? ` ${profileDetails}` : "");
  admissionAction.disabled = updating || station.persistence_error || Boolean(station.switching_deployment_id) ||
    (!station.accepting && resetRequired) ||
    (!station.accepting && station.active_tasks.length > 0) ||
    (!station.accepting && !(station.mode === "automatic" ? ready : standby));
}

function renderTasks(items) {
  const activeId = station.current_task_id;
  const active = items.find((item) => item.task_id === activeId);
  currentTask.replaceChildren();
  if (active) {
    const link = document.createElement("a");
    link.href = `./monitor?id=${encodeURIComponent(active.task_id)}`;
    link.textContent = `当前任务：${active.source === "scheduler" ? "自动调度" : "人工调试"} · ${active.status} · 查看监控`;
    currentTask.append(link);
  } else currentTask.textContent = "当前无活动任务";
}

function renderHistory(taskItems) {
  recordsTasks.replaceChildren();
  const pastTasks = taskItems.filter((item) => ["completed", "simulated", "cancelled", "failed"].includes(item.status));
  for (const item of pastTasks) {
    const link = document.createElement("a");
    link.className = "workstation-task";
    link.href = `./monitor?id=${encodeURIComponent(item.task_id)}`;
    const source = item.source === "scheduler" ? "调度批次" : "人工调试";
    const delivery = item.callback?.enabled ? ` · 通知${item.callback.status || "未生成"}` : "";
    link.textContent = `${source} ${item.request_id || item.task_id.slice(0, 8)} · ${item.status}${delivery}`;
    recordsTasks.append(link);
  }
  if (!pastTasks.length) recordsTasks.textContent = "暂无已结束任务";
  recordsStatus.textContent = `已结束任务 ${pastTasks.length} 条`;
}

function refresh() {
  if (pendingRefresh) return pendingRefresh;
  pendingRefresh = (async () => {
    const [next, taskList] = await Promise.all([
      request("/workstation"), panel || records ? request("/tasks") : Promise.resolve({ tasks: [] }),
    ]);
    station = next;
    station.console_deployments ||= [];
    if (panel) {
      admissionAction.textContent = station.accepting ? "暂停新任务" : "核实后允许新任务";
      admissionAction.title = "仅控制下一次任务能否启动；不会暂停、恢复或结束正在运行的任务";
      if (!updateError) status.classList.remove("is-error");
      const ready = station.profiles.filter((item) => item.ready).length;
      const active = station.active_tasks.length;
      const resetRequired = station.robot_gate?.required === true;
      robotState.textContent = resetRequired
        ? `机器人：${station.robot_gate.status === "resetting" ? "复位中" : station.robot_gate.status === "failed" ? "复位失败" : "待复位"}；${station.robot_gate.reason || "成功复位前不能启动下一任务"}`
        : active ? "机器人：任务执行中" : "机器人：任务名额空闲；启动前仍需检测机器人";
      resetLink.hidden = !resetRequired;
      const entry = station.mode === "automatic"
        ? ready ? active ? "当前批次执行中" : "等待调度 API" : "任务模板检查未通过，无法接收新任务"
        : "可从控制台人工启动任务";
      status.textContent = `${station.mode === "automatic" ? "调度控制" : "人工控制"} · ${station.accepting ? entry : `新任务入口已关闭：${station.reason || "等待人工核实"}`} · 活动任务 ${active}` +
        (resetRequired ? ` · 机器人未复位：${station.robot_gate.reason || "请先完成机器人复位"}` : "");
      switchStatus.textContent = station.switching_deployment_id ? "正在应用配置；新任务入口已关闭" :
        station.switch_error ? `应用失败：${station.switch_error}；新任务入口保持关闭` : "";
      if (updateError) showError(updateError);
      renderTasks(taskList.tasks || []);
      syncControls();
    }
    if (records) renderHistory(taskList.tasks || []);
    window.dispatchEvent(new CustomEvent("nexus-workstation-change", { detail: station }));
  })().finally(() => { pendingRefresh = null; });
  return pendingRefresh;
}

async function update(changes) {
  if (updating) return;
  updating = true;
  updateError = "";
  syncControls();
  try {
    if (pendingRefresh) await pendingRefresh;
    await request("/workstation", { method: "POST", headers: { "X-CSRF-Token": bootstrap.csrf_token }, body: JSON.stringify(changes) });
    await refresh();
  } catch (error) {
    updateError = error.message;
    showError(error.message);
  } finally {
    updating = false;
    syncControls();
  }
}

if (panel || records) {
  if (panel) {
    handoff.addEventListener("click", () => update({
      mode: station.mode === "automatic" ? "debug" : "automatic",
      accepting: station.mode === "automatic" ? !station.robot_gate?.required : true,
    }));
    admissionAction.addEventListener("click", () => update({ accepting: !station.accepting }));
  }
  (async () => {
    try {
      bootstrap = await request("/bootstrap");
      if (records) configureAlignmentLink(bootstrap);
      await refresh();
    } catch (error) { showError(error.message); }
  })();
  window.setInterval(() => {
    if (bootstrap && !document.hidden && !updating) refresh().catch((error) => showError(error.message));
  }, 1500);
}
