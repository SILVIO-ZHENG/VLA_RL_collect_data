import { configureAlignmentLink } from "./alignment-link.js";
import { freshCsrfHeaders, sessionErrorMessage } from "./ui-session.js?v=20261008-csrf-refresh";
import { inferenceModeLabel, modelVersionForConfig, modelVersionLabel, rtcInferenceMode, trainedRtcSupported, inferenceSelectionValue, parseInferenceSelection } from "./rtc-labels.js?v=20261016-rtc-selector";
import { RTC_MODEL_MODES } from "./rtc-model-metadata.js?v=20261016-rtc-selector";

const API_ROOT = "/api";
const MODEL_STAGES = ["检查服务", "准备环境", "加载配置", "加载权重", "搬入 GPU", "模型预热", "健康检查"];
const PHASE_INDEX = { checking: 0, environment: 1, config: 2, weights: 3, gpu: 4, warmup: 5, health: 6, healthy: 6, simulated: 6, stopped: 6 };
const SERVER_LABELS = { reuse: "使用已运行服务", mock: "Mock 测试服务", real: "真实模型服务" };
const HARDWARE_LABELS = { "read-only": "只读观测", execute: "真机执行" };
const TWIN_VIEW_STORAGE_KEY = "nexus-platform-twin-view";
const SAFETY_NOTICE_DURATION_MS = { collision: 2000, warning: 1200 };
const monitorView = window.location.pathname === "/monitor";
document.body.classList.add(monitorView ? "monitor-page" : "settings-page");
document.title = monitorView ? "机器人控制台 · Nexus" : "运行配置 · Nexus";

function savedTwinView() {
  try {
    const view = window.localStorage.getItem(TWIN_VIEW_STORAGE_KEY);
    return ["near", "overview"].includes(view) ? view : "overview";
  } catch {
    return "overview";
  }
}

function rememberTwinView(view) {
  try {
    window.localStorage.setItem(TWIN_VIEW_STORAGE_KEY, view);
  } catch {
    // Browser storage is optional; persisted server presets still work.
  }
}

const state = {
  bootstrap: null,
  csrfToken: "",
  job: null,
  jobId: new URLSearchParams(window.location.search).get("id") || "",
  pollTimer: null,
  formInitialized: false,
  formDirty: false,
  busy: false,
  lastApplyError: "",
  pollFailures: 0,
  pollErrorShown: false,
  destroyed: false,
  startArmed: false,
  startArmTimer: null,
  modelElapsedSnapshot: null,
  modelElapsedUpdatedAt: 0,
  modelElapsedTimer: null,
  resetArmed: false,
  resetArmTimer: null,
  closeModelArmed: false,
  cameraShellKey: "",
  renderedCameraFrameKey: "",
  pendingCameraFrameKey: "",
  cameraSwapGeneration: 0,
  cameraBatch: null,
  latestCameras: [],
  camerasVisible: monitorView,
  twinVisible: monitorView,
  twinSoftwareRenderer: false,
  twinTimer: null,
  twinTelemetry: null,
  safetyNotice: null,
  safetyNoticeSequence: null,
  safetyHazardKey: null,
  twinViewerLoaded: false,
  twinRendererMode: "",
  twinRenderedSequence: null,
  twinRenderedStale: null,
  twinSyncError: "",
  twinPollIntervalMs: 125,
  twinView: savedTwinView(),
  collisionOverlayVisible: true,
  twinViewerOrigin: "",
  twinPresets: {},
  twinCaptureRequests: new Map(),
  usableSafetyProfiles: [],
};

const ids = [
  "connection-state", "session-error", "session-error-message", "session-content", "route-label", "job-id",
  "model-status", "model-name", "model-endpoint", "model-progress", "model-stages", "model-strategy",
  "model-phase", "model-elapsed", "model-health", "model-effector-profile", "model-normalization", "model-log", "telemetry-state", "telemetry-message",
  "camera-grid", "request-metrics", "presence-gate-card", "presence-gate-state", "presence-gate-message",
  "presence-gate-observation", "presence-gate-mode", "presence-gate-window", "presence-gate-empty-elapsed",
  "presence-gate-action", "presence-gate-freshness", "presence-gate-boundary", "runtime-form", "robot-id", "robot-host", "robot-port", "probe-robot",
  "probe-result", "console-probe-robot", "console-probe-result", "task", "presence-gate-enabled", "inference-mode", "hardware-mode", "duration-s", "duration-unit", "execution-fps", "servo-fps",
  "chunk-smoothing-mode", "action-interpolation", "arm-tracking-mode", "arm-tracking-omega", "rtc-prefix-mode", "arm-limit-profile", "max-arm-velocity", "max-arm-acceleration",
  "prediction-horizon", "quality-horizon", "lead-margin", "rtc-blend", "request-timeout", "hitl",
  "workspace-safety", "safety-profile", "safety-availability", "hardware-mode-help",
  "low-pass-filter", "effector-low-pass-filter", "hold-effectors",
  "config-lock", "config-message", "save-config", "task-status", "control-model", "control-model-version", "control-robot", "control-mode",
  "control-hardware", "control-note", "start", "pause", "resume", "stop",
  "reset-robot", "robot-reset-status", "close-model", "ownership-note", "events", "toast-region", "session-main", "safe-hold", "execute-gate",
  "twin-state", "robot-twin", "twin-viewport", "twin-placeholder", "collision-banner",
  "collision-title", "collision-detail", "twin-metrics", "twin-save-view", "twin-geometry-note", "collision-geometry-toggle",
  "safety-distance-panel", "inter-arm-distance-row", "inter-arm-distance-current",
  "inter-arm-warning-distance", "inter-arm-hard-distance", "table-distance-row",
  "table-distance-current", "table-distance-side", "table-warning-distance", "table-hard-distance",
];
const dom = Object.fromEntries(ids.map((id) => [id.replaceAll("-", "_"), document.getElementById(id)]));

async function api(path, options = {}) {
  const headers = await freshCsrfHeaders(options);
  headers.set("Accept", "application/json");
  if (options.body !== undefined) headers.set("Content-Type", "application/json");
  const response = await fetch(`${API_ROOT}${path}`, { ...options, headers, cache: "no-store", credentials: "same-origin" });
  let payload = null;
  try { payload = await response.json(); } catch { payload = null; }
  if (!response.ok) throw new Error(sessionErrorMessage(response, payload) || payload?.error || `请求失败（HTTP ${response.status}）`);
  return payload;
}

function csrfHeaders() {
  return { "X-CSRF-Token": state.csrfToken };
}

function toast(message, tone = "info") {
  if (isAbortLike(message)) return;
  const item = document.createElement("div");
  item.className = `toast ${tone === "info" ? "" : `is-${tone}`}`.trim();
  item.textContent = message;
  dom.toast_region.append(item);
  window.setTimeout(() => item.remove(), 4500);
}

function isAbortLike(error) {
  const message = String(error?.message || error || "");
  return error?.name === "AbortError" || /abort(?:ed|error)?/i.test(message);
}

function setConnection(kind, label) {
  dom.connection_state.className = `connection-state is-${kind}`;
  dom.connection_state.lastElementChild.textContent = label;
}

function modelById(id) {
  return state.bootstrap?.models?.find((item) => String(item.id) === String(id));
}

function robotById(id) {
  return state.bootstrap?.robots?.find((item) => String(item.id) === String(id));
}

function formatDuration(value) {
  const seconds = Math.max(0, Math.round(Number(value) || 0));
  if (seconds < 60) return `${seconds} 秒`;
  return `${Math.floor(seconds / 60)}分 ${String(seconds % 60).padStart(2, "0")}秒`;
}

function formatTime(value) {
  const normalized = typeof value === "number" && value < 10_000_000_000 ? value * 1000 : value;
  const date = new Date(normalized);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleTimeString("zh-CN", { hour12: false });
}

function modelProgress(service) {
  if (["ready", "simulated", "stopped"].includes(service.status)) return 100;
  const index = PHASE_INDEX[service.phase] ?? 0;
  if (service.shard_current && service.shard_total) return 38 + 30 * service.shard_current / service.shard_total;
  return [7, 17, 29, 50, 69, 84, 94][index];
}

function modelElapsedText(service, ageMs = 0) {
  if (!Number.isFinite(Number(service.started_at)) || service.started_at == null) {
    return service.status === "ready" ? "已复用服务" : "未开始";
  }
  const elapsed = Math.max(0, Number(service.elapsed_s) || 0);
  const running = service.status === "starting" && !service.finished_at;
  return formatDuration(elapsed + (running ? Math.max(0, ageMs) / 1000 : 0));
}

function renderModelElapsed() {
  const service = state.modelElapsedSnapshot;
  if (!service) return;
  const ageMs = performance.now() - state.modelElapsedUpdatedAt;
  dom.model_elapsed.textContent = `${modelElapsedText(service, ageMs)} / ${formatDuration(service.timeout_s)}`;
}

function startModelElapsedTimer() {
  window.clearInterval(state.modelElapsedTimer);
  state.modelElapsedTimer = window.setInterval(() => {
    if (state.modelElapsedSnapshot?.status === "starting" && !state.destroyed) renderModelElapsed();
  }, 250);
}

function renderModel(job) {
  const service = job.model_service || {};
  const model = modelById(job.config.model_id);
  const variant = model?.effector_variants?.[job.config.effector_profile];
  const ready = service.status === "ready";
  const labels = { pending: "等待检查", starting: "模型加载中", ready: "服务健康", simulated: "模拟完成", candidate: "待应用", stopped: "已关闭" };
  dom.model_status.textContent = labels[service.status] || service.status || "等待检查";
  dom.model_status.className = `runtime-status ${ready ? "is-ready" : service.status === "starting" ? "is-active" : service.status === "stopped" ? "is-failed" : ""}`;
  dom.model_name.textContent = model?.label || service.model_id || job.config.model_id;
  dom.model_endpoint.textContent = `GPU ${service.gpu ?? "—"} · :${service.port ?? "—"}`;
  dom.model_progress.value = modelProgress(service);
  dom.model_strategy.textContent = SERVER_LABELS[service.strategy] || service.strategy || "—";
  const shard = service.shard_current && service.shard_total ? ` · 分片 ${service.shard_current}/${service.shard_total}` : "";
  dom.model_phase.textContent = `${service.phase_label || "等待检查"}${shard}`;
  state.modelElapsedSnapshot = service;
  state.modelElapsedUpdatedAt = performance.now();
  renderModelElapsed();
  dom.model_health.textContent = service.healthy ? "健康" : service.status === "simulated" ? "未实际探测" : "未就绪";
  dom.model_effector_profile.textContent = variant?.label || model?.effector_profile_label || "—";
  dom.model_normalization.textContent = variant?.normalization?.label || model?.normalization?.label || "—";
  dom.model_log.textContent = service.recent_log || job.message || "等待模型输出";
  dom.model_log.title = dom.model_log.textContent;
  const active = PHASE_INDEX[service.phase] ?? 0;
  const complete = ["ready", "simulated", "stopped"].includes(service.status);
  const stagesKey = `${active}:${complete}`;
  if (dom.model_stages.dataset.renderKey === stagesKey) return;
  dom.model_stages.dataset.renderKey = stagesKey;
  dom.model_stages.replaceChildren(...MODEL_STAGES.map((label, index) => {
    const li = document.createElement("li");
    li.textContent = label;
    if (complete || index < active) li.className = "is-complete";
    else if (index === active) li.className = "is-active";
    return li;
  }));
}

function cameraStatusLabel(status) {
  return ({
    online: "最近一次推理输入",
    last_frame: "最后一次推理输入",
    waiting: "等待首次推理",
  })[status] || "尚无推理画面";
}

function buildCameraShell(cameras) {
  const shellKey = cameras.map((camera) => String(camera.id)).join("|");
  if (shellKey === state.cameraShellKey) {
    cameras.forEach((camera) => {
      const card = [...dom.camera_grid.querySelectorAll("[data-camera-id]")]
        .find((item) => item.dataset.cameraId === String(camera.id));
      const status = card?.querySelector("[data-camera-status]");
      if (status) status.textContent = cameraStatusLabel(camera.status);
    });
    return;
  }
  state.cameraShellKey = shellKey;
  cancelInferenceImages();
  state.renderedCameraFrameKey = "";
  state.pendingCameraFrameKey = "";
  state.cameraSwapGeneration += 1;
  dom.camera_grid.replaceChildren(...cameras.map((camera) => {
    const card = document.createElement("article");
    card.className = "camera-view";
    card.dataset.cameraId = String(camera.id);
    const frame = document.createElement("div");
    frame.className = "camera-frame";
    frame.dataset.cameraFrame = String(camera.id);
    const placeholder = document.createElement("div");
    placeholder.className = "camera-placeholder";
    placeholder.innerHTML = '<svg viewBox="0 0 24 24"><path d="M4 7h4l1.5-2h5L16 7h4v11H4zM12 10a3 3 0 1 0 0 6 3 3 0 0 0 0-6Z"/></svg><span>等待首次推理</span>';
    frame.append(placeholder);
    const caption = document.createElement("div");
    caption.className = "camera-caption";
    const name = document.createElement("strong");
    name.textContent = camera.label || camera.id;
    const status = document.createElement("small");
    status.dataset.cameraStatus = String(camera.id);
    status.textContent = cameraStatusLabel(camera.status);
    caption.append(name, status);
    card.append(frame, caption);
    return card;
  }));
}

function preloadInferenceImage(camera) {
  const batch = state.cameraBatch;
  return new Promise((resolve, reject) => {
    const image = new Image();
    let settled = false;
    const finish = (error) => {
      if (settled) return;
      settled = true;
      image.onload = image.onerror = null;
      batch.controller.signal.removeEventListener("abort", abort);
      if (error) { image.removeAttribute("src"); reject(error); }
      else resolve({ camera, image });
    };
    const abort = () => finish(new DOMException("Image load cancelled", "AbortError"));
    batch.controller.signal.addEventListener("abort", abort, { once: true });
    image.alt = `${camera.label || camera.id}最近一次推理输入`;
    image.decoding = "async";
    image.onload = async () => {
      try {
        if (typeof image.decode === "function") await image.decode();
      } catch {
        // The load event already proves that the image is displayable. Some
        // browsers reject a redundant decode() after completing the decode.
      }
      finish();
    };
    image.onerror = () => finish(new Error(`无法加载${camera.label || camera.id}推理画面`));
    image.src = camera.image_url;
  });
}

function cancelInferenceImages() {
  const batch = state.cameraBatch;
  state.cameraBatch = null;
  state.cameraSwapGeneration += 1;
  state.pendingCameraFrameKey = "";
  if (batch) {
    window.clearTimeout(batch.timer);
    batch.controller.abort();
  }
}

function scheduleInferenceFrameSwap(cameras) {
  state.latestCameras = cameras;
  if (state.destroyed || document.hidden || !state.camerasVisible || state.cameraBatch) return;
  const readyCameras = cameras.filter((camera) => (
    typeof camera.image_url === "string" && camera.image_url.startsWith("/api/")
  ));
  if (!readyCameras.length || readyCameras.length !== cameras.length) return;
  const frameKey = readyCameras
    .map((camera) => `${camera.id}:${camera.frame_id || camera.image_url}`)
    .join("|");
  if (frameKey === state.renderedCameraFrameKey || frameKey === state.pendingCameraFrameKey) return;

  state.pendingCameraFrameKey = frameKey;
  const generation = ++state.cameraSwapGeneration;
  const batch = { controller: new AbortController(), timer: null, succeeded: false };
  state.cameraBatch = batch;
  batch.timer = window.setTimeout(() => batch.controller.abort(), 6000);
  Promise.all(readyCameras.map(preloadInferenceImage)).then((frames) => {
    if (
      state.destroyed || document.hidden || !state.camerasVisible
      || generation !== state.cameraSwapGeneration
      || frameKey !== state.pendingCameraFrameKey
    ) return;
    // All images have loaded and decoded. Replacing all three synchronously in
    // one task lets the browser paint them as one frame and keeps the previous
    // inference input visible throughout network/decode time.
    frames.forEach(({ camera, image }) => {
      const frame = [...dom.camera_grid.querySelectorAll("[data-camera-frame]")]
        .find((item) => item.dataset.cameraFrame === String(camera.id));
      if (frame) frame.replaceChildren(image);
    });
    state.renderedCameraFrameKey = frameKey;
    state.pendingCameraFrameKey = "";
    batch.succeeded = true;
  }).catch(() => {
    if (generation === state.cameraSwapGeneration) state.pendingCameraFrameKey = "";
    // Keep the last complete inference frame. A later poll will retry without
    // flashing a placeholder or exposing a partially updated camera set.
  }).finally(() => {
    window.clearTimeout(batch.timer);
    batch.controller.abort();
    if (state.cameraBatch === batch) {
      state.cameraBatch = null;
      if (batch.succeeded) scheduleInferenceFrameSwap(state.latestCameras);
    }
  });
}

function renderMetricCells(container, metrics, className) {
  const key = JSON.stringify(metrics.map(([label]) => label));
  if (container.dataset.labels !== key) {
    container.dataset.labels = key;
    container.replaceChildren(...metrics.map(([label]) => {
      const item = document.createElement("div");
      const name = document.createElement("span"); name.textContent = label;
      item.append(name, document.createElement("strong"));
      return item;
    }));
  }
  metrics.forEach(([, value, danger, warning], index) => {
    const item = container.children[index];
    const classes = `${className}${danger ? " is-danger" : warning ? " is-warning" : ""}`;
    if (item.className !== classes) item.className = classes;
    const text = String(value ?? "等待数据");
    if (item.lastElementChild.textContent !== text) {
      item.lastElementChild.textContent = text;
      item.lastElementChild.title = text;
    }
  });
}

function renderTelemetry(job) {
  const telemetry = job.telemetry || {};
  const transportLabels = {
    online: "遥测在线", connecting: "连接中", stalled: "遥测停滞",
    paused: "已暂停 · 相机预览", ended: "已结束 · 最后一帧",
    cancelled: "已取消", failed: "遥测失败", not_connected: "等待遥测",
  };
  dom.telemetry_state.textContent = transportLabels[telemetry.transport]
    || (telemetry.available ? "遥测在线" : "等待遥测");
  dom.telemetry_message.textContent = telemetry.message || "产生推理请求后显示最近一次实际输入画面。";
  const cameras = Array.isArray(telemetry.cameras) ? telemetry.cameras : [];
  buildCameraShell(cameras);
  scheduleInferenceFrameSwap(cameras);
  const request = telemetry.request || {};
  const metrics = [
    ["观测状态", ({ received: "真机已采集", synthetic: "合成观测", waiting: "等待首帧", not_received: "尚未采集" })[telemetry.observation_status]],
    ["推理状态", ({ running: "实际推理中", paused: "已暂停推理 · 相机预览中", waiting_first_request: "等待首个请求", not_started: "尚未运行" })[telemetry.inference_status]],
    ["Request ID", request.request_id], ["Session ID", request.session_id], ["RTT", request.rtt_ms == null ? null : `${request.rtt_ms} ms`],
    ["模型推理", request.inference_ms == null ? null : `${request.inference_ms} ms`], ["动作队列", request.queue_depth],
    ["Chunk ID", request.chunk_id], ["RTC 融合", request.rtc_blend_steps == null ? null : `${request.rtc_blend_steps} 步`], ["遥测链路", transportLabels[telemetry.transport] || telemetry.transport],
    ["累计请求", Number.isFinite(Number(telemetry.request_count)) ? telemetry.request_count : null],
    ["响应新鲜度", telemetry.response_age_s == null ? null : `${telemetry.response_age_s} 秒前`],
    ["推理画面时间", telemetry.frame_age_s == null ? null : `${telemetry.frame_age_s} 秒前`],
  ];
  renderMetricCells(dom.request_metrics, metrics, "request-metric");
}

function renderPresenceGate(job) {
  // The browser remains display-only: it never turns a visual state into a
  // control request.  The server-owned gate independently validates every
  // observation before it may call an existing guarded robot operation.
  const configured = job.config?.presence_gate || {};
  const gate = job.presence_gate || {};
  const enabled = gate.enabled === true || configured.enabled === true;
  const readOnly = job.config?.hardware_mode === "read-only";
  const clearClasses = () => dom.presence_gate_card.classList.remove(
    "is-present", "is-empty", "is-stale",
  );
  const setChip = (label, kind = "") => {
    dom.presence_gate_state.textContent = label;
    dom.presence_gate_state.className = `telemetry-state${kind ? ` ${kind}` : ""}`;
  };
  if (!enabled) {
    clearClasses();
    setChip("未启用");
    dom.presence_gate_message.textContent = "本任务未启用包裹检测；视觉状态不会参与任务控制。";
    dom.presence_gate_observation.textContent = "未启用";
    dom.presence_gate_mode.textContent = "不记录包裹检测";
    dom.presence_gate_window.textContent = "—";
    dom.presence_gate_empty_elapsed.textContent = "—";
    dom.presence_gate_action.textContent = "无";
    dom.presence_gate_freshness.textContent = "未启用";
    dom.presence_gate_boundary.textContent = job.config?.human_in_loop && job.config?.hardware_mode === "execute"
      ? "已获准的人工任务可由页面或平板新录制启动；视觉状态不能创建、替代或恢复任务。"
      : "只有操作员点击“开始任务”才会授权一个会话；视觉状态永远不能创建、替代或恢复任务。";
    dom.presence_gate_boundary.removeAttribute("title");
    return;
  }

  const transport = String(gate.transport || "awaiting_task");
  const stateLabel = {
    PACKAGE_PRESENT: "已确认有包裹",
    NO_PACKAGE: "已确认无包裹",
    UNKNOWN: "不确定",
  };
  const shortState = {
    PACKAGE_PRESENT: "有包裹",
    NO_PACKAGE: "无包裹",
    UNKNOWN: "不确定",
  };
  const actionLabel = {
    none: "无待执行动作",
    pause: "自动软暂停",
    reset: readOnly ? "只读复位验证" : "自动复位",
    resume: "自动恢复同一任务",
  };
  const currentState = stateLabel[gate.state] ? String(gate.state) : "UNKNOWN";
  const age = Number(gate.age_s);
  const stale = transport === "stale";
  clearClasses();
  if (currentState === "PACKAGE_PRESENT" && !stale) dom.presence_gate_card.classList.add("is-present");
  else if (currentState === "NO_PACKAGE" && !stale) dom.presence_gate_card.classList.add("is-empty");
  else if (stale) dom.presence_gate_card.classList.add("is-stale");

  if (transport === "online") setChip("包裹检测在线", "is-online");
  else if (stale) setChip("视觉数据过期", "is-failed");
  else if (transport === "failed") setChip("检测服务不可用", "is-failed");
  else if (transport === "recovery_required") setChip("服务重启后需新建任务", "is-failed");
  else if (transport === "awaiting_runtime_config") setChip("待保存真机安全配置", "is-active");
  else if (transport === "ended") setChip("任务已结束");
  else setChip("等待包裹检测", "is-active");

  dom.presence_gate_message.textContent = gate.message
    || (readOnly
      ? "视觉模型使用四帧一致结果确认包裹状态；只读客户端不会发送机械臂命令。"
      : "视觉模型使用四帧一致结果确认包裹状态；自动动作仅由控制服务执行。");
  dom.presence_gate_observation.textContent = stateLabel[currentState];
  dom.presence_gate_mode.textContent = gate.mode === "automatic" && readOnly
    ? "只读观测 · 自动门控验证 · 不发送机械臂命令"
    : gate.mode === "automatic"
    ? "自动安全门控 · 由控制服务执行"
    : "未启用自动门控";
  const windowStates = Array.isArray(gate.window_states) ? gate.window_states : [];
  dom.presence_gate_window.textContent = windowStates.length
    ? windowStates.map((item) => shortState[item] || "不确定").join(" · ")
    : "等待四帧结果";
  const emptyElapsed = Number(gate.no_package_elapsed_s);
  dom.presence_gate_empty_elapsed.textContent = Number.isFinite(emptyElapsed)
    ? `${emptyElapsed.toFixed(1)} 秒`
    : "—";
  const pendingAction = String(gate.pending_action || gate.proposed_action || gate.last_action || "none");
  const actionStatus = String(gate.action_status || "idle");
  const statusSuffix = {
    queued: "（等待执行）",
    applying: "（执行中）",
    resetting: "（复位中）",
    applied: "（已完成）",
    failed: "（失败，已安全停用）",
    disabled: "（已停用）",
    disabled_after_restart: "（服务重启后已停用）",
  }[actionStatus] || "";
  dom.presence_gate_action.textContent = `${actionLabel[pendingAction] || "无待执行动作"}${statusSuffix}`;
  dom.presence_gate_freshness.textContent = Number.isFinite(age)
    ? stale ? `${age.toFixed(1)} 秒前 · 过期` : `${age.toFixed(1)} 秒前 · 新鲜`
    : transport === "awaiting_observation" ? "等待视觉结果"
      : transport === "awaiting_runtime_config" ? "待运行配置" : "—";
  const reason = String(gate.reason || "").trim();
  dom.presence_gate_boundary.textContent = readOnly
    ? "只有操作员点击“开始任务”才会授权一个只读会话；视觉状态不能创建、替代或复活任务。暂停／恢复只控制只读推理会话，5 秒复位只作验证记录；不会发送机械臂命令。UNKNOWN、混合窗口、过期或失联数据均不会触发动作。"
    : job.config?.human_in_loop
      ? "已获准的人工任务可由页面或平板新录制启动；视觉状态不能创建、替代或复活任务。UNKNOWN、混合窗口、过期或失联数据均不会触发动作。"
      : "只有操作员点击“开始任务”才会授权一个会话；视觉状态不能创建、替代或复活任务。UNKNOWN、混合窗口、过期或失联数据均不会触发动作。";
  if (reason) dom.presence_gate_boundary.title = reason;
  else dom.presence_gate_boundary.removeAttribute("title");
}

function initializeTwinViewer() {
  const simulator = state.bootstrap?.simulator || {};
  if (!simulator.enabled) {
    state.twinSyncError = "完整三维渲染服务未配置";
    return;
  }
  const updateHz = Math.max(1, Math.min(10, Number(simulator.update_hz) || 8));
  state.twinPollIntervalMs = Math.round(1000 / updateHz);
  const viewerPort = Number(simulator.viewer_port) || 8767;
  const controlPort = Number(simulator.control_port) || 8766;
  // Only the documented 18765/18766/18767 SSH tunnel needs translation.
  // Direct Thor :21865 uses the configured viewer :8766/:8767.
  const pagePort = Number(window.location.port);
  const tunnelOffset = pagePort === 18765 ? 10000 : 0;
  const browserControlPort = controlPort + tunnelOffset;
  const browserViewerPort = viewerPort + tunnelOffset;
  state.twinViewerOrigin = `${window.location.protocol}//${window.location.hostname}:${browserControlPort}`;
  const viewerParams = new URLSearchParams({
    meshcat_port: String(browserViewerPort),
    platform_origin: window.location.origin,
  });
  dom.robot_twin.src = `${state.twinViewerOrigin}/viewer/?${viewerParams}`;
  dom.robot_twin.addEventListener("load", () => {
    state.twinSyncError = "";
  }, { once: true });
}

function applyTwinCamera(camera) {
  if (!state.twinViewerLoaded || !camera || typeof camera !== "object") return;
  dom.robot_twin.contentWindow?.postMessage(
    { type: "nexus-camera-apply", camera, view: state.twinView }, state.twinViewerOrigin,
  );
}

function stabilizeInitialTwinCamera(camera) {
  // MeshCat may continue replaying the scene's default camera briefly after its
  // WebSocket reports open. Reapply only during initial viewer startup so the
  // persisted preset wins without fighting later user mouse interaction.
  [0, 180, 500].forEach((delay) => {
    window.setTimeout(() => applyTwinCamera(camera), delay);
  });
}

function captureTwinCamera() {
  if (!state.twinViewerLoaded) return Promise.reject(new Error("三维画面尚未就绪"));
  const requestId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => {
      state.twinCaptureRequests.delete(requestId);
      reject(new Error("读取当前视角超时"));
    }, 1800);
    state.twinCaptureRequests.set(requestId, { resolve, reject, timer });
    dom.robot_twin.contentWindow?.postMessage(
      { type: "nexus-camera-capture", requestId }, state.twinViewerOrigin,
    );
  });
}

window.addEventListener("message", (event) => {
  if (event.origin !== state.twinViewerOrigin || event.source !== dom.robot_twin.contentWindow) return;
  const message = event.data;
  if (!message || typeof message !== "object") return;
  if (message.type === "nexus-viewer-ready") {
    state.twinViewerLoaded = true;
    state.twinRendererMode = String(message.renderer || "unknown");
    state.twinSoftwareRenderer = message.softwareRenderer === true;
    syncPresentationVisibility();
    state.twinSyncError = "";
    dom.twin_placeholder.classList.remove("is-loading");
    dom.twin_placeholder.hidden = true;
    dom.robot_twin.contentWindow?.postMessage(
      { type: "nexus-collision-overlay", visible: state.collisionOverlayVisible },
      state.twinViewerOrigin,
    );
    window.setTimeout(
      () => setTwinView(state.twinView, { force: true, stabilize: true }),
      100,
    );
    if (state.twinTelemetry) renderTwin(state.twinTelemetry);
    return;
  }
  if (message.type === "nexus-viewer-error") {
    state.twinSyncError = String(message.error || "三维渲染资源加载失败");
    dom.twin_placeholder.classList.remove("is-loading");
    dom.twin_placeholder.hidden = false;
    dom.twin_placeholder.textContent = state.twinSyncError;
    return;
  }
  if (!["nexus-camera-state", "nexus-camera-error"].includes(message.type)) return;
  const pending = state.twinCaptureRequests.get(String(message.requestId || ""));
  if (!pending) return;
  window.clearTimeout(pending.timer);
  state.twinCaptureRequests.delete(String(message.requestId || ""));
  if (message.type === "nexus-camera-state") pending.resolve(message.camera);
  else pending.reject(new Error(String(message.error || "读取当前视角失败")));
});

async function setTwinView(view, { force = false, stabilize = false } = {}) {
  if (!["near", "overview"].includes(view)
      || (!force && view === state.twinView) || state.busy) return;
  const buttons = [...document.querySelectorAll("[data-twin-view]")];
  buttons.forEach((button) => { button.disabled = true; });
  try {
    const result = await api(`/deployments/${encodeURIComponent(state.jobId)}/simulator/view`, {
      method: "POST", headers: csrfHeaders(), body: JSON.stringify({ view }),
    });
    state.twinView = view;
    rememberTwinView(view);
    if (result?.preset) state.twinPresets[view] = result.preset;
    buttons.forEach((button) => {
      const active = button.dataset.twinView === view;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    if (stabilize) stabilizeInitialTwinCamera(state.twinPresets[view]);
    else {
      applyTwinCamera(state.twinPresets[view]);
      window.setTimeout(() => applyTwinCamera(state.twinPresets[view]), 160);
    }
  } catch (error) {
    if (!isAbortLike(error)) toast(error.message, "error");
  } finally {
    buttons.forEach((button) => { button.disabled = false; });
  }
}

async function saveTwinView() {
  if (state.busy || !state.twinViewerLoaded) return;
  dom.twin_save_view.disabled = true;
  try {
    const camera = await captureTwinCamera();
    const result = await api(`/deployments/${encodeURIComponent(state.jobId)}/simulator/preset`, {
      method: "POST",
      headers: csrfHeaders(),
      body: JSON.stringify({ view: state.twinView, camera }),
    });
    state.twinPresets[state.twinView] = result.preset || camera;
    const label = state.twinView === "overview" ? "整机远景" : "双臂近景";
    toast(`${label}已保存，之后进入该视角会恢复到当前位置。`, "success");
  } catch (error) {
    if (!isAbortLike(error)) toast(`保存视角失败：${error.message}`, "error");
  } finally {
    dom.twin_save_view.disabled = false;
  }
}

function updateSafetyNotice({
  sequence,
  safety,
  collision,
  proximityWarning,
  intervention,
  candidateOnly,
  distanceMm,
  distanceLabel,
  hardMarginMm,
  warningMarginMm,
}) {
  const now = Date.now();
  const incomingKind = collision ? "collision" : proximityWarning ? "warning" : null;
  const reason = String(safety.reason || "");
  const workSurfaceHazard = Boolean(
    safety.work_surface?.collision || safety.work_surface?.warning,
  );
  const interArmHazard = Boolean(safety.inter_arm?.collision || safety.inter_arm?.warning);
  const candidateInterArmHazard = Boolean(
    safety.candidate_inter_arm?.collision || safety.candidate_inter_arm?.warning,
  );
  const hazardScope = workSurfaceHazard && interArmHazard
    ? "multiple"
    : interArmHazard
      || safety.limiting_surface === "inter_arm"
      || safety.candidate_reason === "inter_arm_collision_hold_previous_safe_action"
      || (candidateOnly && candidateInterArmHazard)
      ? "inter_arm"
      : "work_surface";
  const interArm = hazardScope === "inter_arm";
  const tableMm = Number(safety.work_surface?.distance_m) * 1000;
  const armsMm = Number(safety.inter_arm?.minimum_separation_m) * 1000;
  const combinedDistanceLabel = hazardScope === "multiple"
    ? `桌面几何净距 ${Number.isFinite(tableMm) ? tableMm.toFixed(2) : "—"} mm · 双臂实际净距 ${Number.isFinite(armsMm) ? armsMm.toFixed(2) : "—"} mm`
    : distanceLabel;
  const incomingHazardKey = incomingKind
    ? [incomingKind, hazardScope, candidateOnly ? "candidate" : "current", reason].join(":")
    : null;
  const preserveActiveCollision = incomingKind === "warning"
    && state.safetyNotice?.kind === "collision"
    && now < state.safetyNotice.expiresAt;
  // A hazard is latched for a minimum readable interval. Every fresh sample of
  // the same hazard renews the deadline, so a continuously unsafe pose remains
  // visible. Once the hazard clears, the final sample stays on screen until
  // its minimum duration has elapsed instead of flickering between polls.
  if (incomingKind && !preserveActiveCollision && incomingHazardKey !== state.safetyHazardKey) {
    const kind = incomingKind;
    let title;
    let detail;
    if (proximityWarning) {
      title = hazardScope === "multiple"
        ? "桌面与双臂距离预警"
        : interArm ? "左右机械臂距离预警" : "夹爪接近桌面";
      detail = `${combinedDistanceLabel} · 当前仅预警，动作保持允许`;
    } else if (reason === "safety_guard_deadline_hold") {
      title = "安全检查超时，已保持姿态";
      detail = `${distanceLabel} · 本周期未向机器人放行候选动作`;
    } else {
      title = hazardScope === "multiple"
        ? "多项碰撞限制已触发"
        : interArm
          ? candidateOnly ? "双臂碰撞候选动作已拦截" : "当前姿态触发双臂碰撞限制"
          : candidateOnly ? "模型候选动作已拦截" : "当前姿态触发桌面硬限制";
      detail = `${combinedDistanceLabel}${hazardScope === "multiple" ? "" : ` · 硬限制阈值 ${hardMarginMm?.toFixed(2) ?? "—"} mm（阈值不是当前距离）`} · ${
        intervention
          ? candidateOnly
            ? `保持上一放行目标，持续检查后续动作 · ${safety.response || "碰撞拦截"}`
            : `拦截已生效 · ${safety.response || "保持安全动作"}`
          : "已命中硬限制条件；当前为只读观测，不接管遥操作"
      }`;
    }
    state.safetyNotice = {
      kind,
      title,
      detail,
      expiresAt: now + SAFETY_NOTICE_DURATION_MS[kind],
    };
    state.safetyNoticeSequence = sequence;
    state.safetyHazardKey = incomingHazardKey;
  } else if (incomingKind && !preserveActiveCollision && state.safetyNotice) {
    state.safetyNotice.expiresAt = now + SAFETY_NOTICE_DURATION_MS[state.safetyNotice.kind];
    state.safetyNoticeSequence = sequence;
  } else if (!incomingKind) {
    state.safetyHazardKey = null;
  }

  const notice = state.safetyNotice;
  if (!notice || now >= notice.expiresAt) {
    state.safetyNotice = null;
    dom.collision_banner.hidden = true;
    return;
  }
  const recovered = !collision && !proximityWarning;
  dom.collision_banner.hidden = false;
  dom.collision_banner.classList.toggle("is-warning", notice.kind === "warning");
  dom.collision_title.textContent = notice.title;
  dom.collision_detail.textContent = `${notice.detail}${recovered ? " · 当前状态已恢复" : ""}`;
}

function distanceMillimeters(value) {
  return value != null && Number.isFinite(Number(value)) ? Number(value) * 1000 : null;
}

function formatDistance(value) {
  return value == null ? "—" : `${value.toFixed(2)} mm`;
}

function distanceAtHardLimit(distance, hardMargin) {
  if (distance == null || hardMargin == null) return false;
  const value = Number(distance);
  const threshold = Number(hardMargin);
  return Number.isFinite(value) && Number.isFinite(threshold)
    && value <= threshold + 1e-9;
}

function workSurfaceSide(point) {
  const value = String(point || "").toLowerCase();
  if (value.includes("left")) return "左夹爪";
  if (value.includes("right")) return "右夹爪";
  return value ? "最接近点" : "等待实测姿态";
}

function renderSafetyDistanceRow(row, current, warning, hard, collision, proximityWarning) {
  row.classList.toggle("is-danger", Boolean(collision));
  row.classList.toggle("is-warning", !collision && Boolean(proximityWarning));
  row.classList.toggle("is-unavailable", current == null);
  return {
    current: formatDistance(current),
    warning: formatDistance(warning),
    hard: formatDistance(hard),
  };
}

function renderTwin(control) {
  state.twinTelemetry = control;
  const available = Boolean(control?.available && Array.isArray(control.measured_state));
  const safety = control?.safety || {};
  const stale = Number(control?.age_s) > 1.0;
  const readOnly = String(control?.hardware_mode || state.job?.config?.hardware_mode) === "read-only";
  const currentCollision = Boolean(safety.collision_detected);
  const candidateCollision = Boolean(safety.candidate_collision_detected);
  // Read-only observation validates the pose actually reported by G2. Model
  // candidates remain visible as diagnostics but must never masquerade as a
  // collision of the tele-operated robot. Execute mode still surfaces a
  // rejected candidate because it is an active control intervention.
  const collision = !stale && (currentCollision || (!readOnly && candidateCollision));
  const proximityWarning = !stale && Boolean(safety.proximity_warning) && !collision;
  const intervention = Boolean(safety.intervention_active);
  const bridgeError = String(control?.visualization?.error || "");
  const renderError = state.twinSyncError || bridgeError;
  const renderFailed = Boolean(renderError);
  const fallback = state.twinRendererMode === "meshcat-websocket-fallback";
  dom.twin_state.textContent = renderFailed ? "三维服务未连接" : !available ? "等待机器人状态" : stale ? "状态已延迟" : fallback ? "兼容渲染在线" : state.twinSoftwareRenderer ? "节能渲染在线 · 8 FPS" : "缓存模型在线";
  dom.twin_state.className = `telemetry-state${collision || renderFailed ? " is-failed" : proximityWarning || stale ? " is-active" : available ? " is-online" : ""}`;
  dom.twin_placeholder.textContent = renderFailed ? renderError : "";
  dom.twin_placeholder.classList.toggle("is-loading", !state.twinViewerLoaded && !renderFailed);
  dom.twin_placeholder.hidden = state.twinViewerLoaded && !renderFailed;
  const sequence = Number(control?.sequence);
  if (available && state.twinVisible && !document.hidden && state.twinViewerLoaded && Number.isInteger(sequence)
      && (sequence !== state.twinRenderedSequence || stale !== state.twinRenderedStale)) {
    state.twinRenderedSequence = sequence;
    state.twinRenderedStale = stale;
    dom.robot_twin.contentWindow?.postMessage(
      {
        type: "nexus-measured-joints",
        sequence,
        joints: control.measured_state,
        effectorProfile: state.job?.config?.effector_profile || "dahuan",
        safety,
        stale,
      },
      state.twinViewerOrigin,
    );
  }
  const workSurface = safety.work_surface || null;
  const legacyTableDistance = safety.limiting_surface !== "inter_arm"
    ? safety.distance_to_surface : null;
  const tableDistanceValue = workSurface?.distance_m ?? legacyTableDistance;
  const tableDistanceMm = tableDistanceValue != null && Number.isFinite(Number(tableDistanceValue))
    ? Number(tableDistanceValue) * 1000 : null;
  const interArm = safety.inter_arm || null;
  const candidateInterArm = safety.candidate_inter_arm || null;
  const interArmDistanceMm = interArm?.minimum_separation_m != null
    ? Number(interArm.minimum_separation_m) * 1000
    : null;
  const tableCollision = Boolean(workSurface?.collision)
    || distanceAtHardLimit(workSurface?.distance_m, workSurface?.hard_margin_m)
    || (currentCollision && safety.limiting_surface !== "inter_arm");
  const tableWarning = !tableCollision && (Boolean(workSurface?.warning)
    || (Boolean(safety.proximity_warning) && safety.limiting_surface !== "inter_arm"));
  const interArmCollision = Boolean(interArm?.collision)
    || distanceAtHardLimit(interArm?.minimum_separation_m, interArm?.hard_margin_m)
    || (currentCollision && safety.limiting_surface === "inter_arm");
  const interArmWarning = !interArmCollision && (Boolean(interArm?.warning)
    || (Boolean(safety.proximity_warning) && safety.limiting_surface === "inter_arm"));
  const tableHardMm = distanceMillimeters(workSurface?.hard_margin_m
    ?? (safety.limiting_surface !== "inter_arm" ? safety.hard_margin : null));
  const tableWarningMm = distanceMillimeters(workSurface?.warning_margin_m
    ?? (safety.limiting_surface !== "inter_arm" ? safety.warning_margin : null));
  const interArmHardMm = distanceMillimeters(interArm?.hard_margin_m);
  const interArmWarningMm = distanceMillimeters(interArm?.warning_margin_m);
  const tableValues = renderSafetyDistanceRow(
    dom.table_distance_row,
    tableDistanceMm,
    tableWarningMm,
    tableHardMm,
    tableCollision,
    tableWarning,
  );
  const interArmValues = renderSafetyDistanceRow(
    dom.inter_arm_distance_row,
    interArmDistanceMm,
    interArmWarningMm,
    interArmHardMm,
    interArmCollision,
    interArmWarning,
  );
  dom.safety_distance_panel.classList.toggle("is-stale", stale || !available);
  dom.table_distance_current.textContent = tableValues.current;
  dom.table_warning_distance.textContent = tableValues.warning;
  dom.table_hard_distance.textContent = tableValues.hard;
  dom.table_distance_side.textContent = workSurfaceSide(workSurface?.limiting_point);
  dom.inter_arm_distance_current.textContent = interArmValues.current;
  dom.inter_arm_warning_distance.textContent = interArmValues.warning;
  dom.inter_arm_hard_distance.textContent = interArmValues.hard;
  const interArmLimiting = interArmCollision || interArmWarning
    || safety.limiting_surface === "inter_arm"
    || (!readOnly && safety.candidate_reason === "inter_arm_collision_hold_previous_safe_action")
    || (!readOnly && Boolean(candidateInterArm?.warning || candidateInterArm?.collision));
  const hardMarginValue = interArmLimiting
    ? interArm?.hard_margin_m
    : workSurface?.hard_margin_m ?? safety.hard_margin;
  const warningMarginValue = interArmLimiting
    ? interArm?.warning_margin_m
    : workSurface?.warning_margin_m ?? safety.warning_margin;
  const hardMarginMm = hardMarginValue != null && Number.isFinite(Number(hardMarginValue))
    ? Number(hardMarginValue) * 1000
    : null;
  const warningMarginMm = warningMarginValue != null && Number.isFinite(Number(warningMarginValue))
    ? Number(warningMarginValue) * 1000
    : null;
  // A measured pose at the exact hard boundary is a live hard-limit event,
  // not a candidate-only interception—even if the model candidate is unsafe
  // at the same time.
  const candidateOnly = !readOnly && candidateCollision
    && !tableCollision && !interArmCollision;
  const distanceMm = safety.distance_to_surface != null && Number.isFinite(Number(safety.distance_to_surface))
    ? Number(safety.distance_to_surface) * 1000 : null;
  const distanceLabel = interArmLimiting
    ? candidateOnly && distanceMm != null
      ? `预测路径最小间距 ${distanceMm.toFixed(2)} mm${interArmDistanceMm == null ? "" : ` · 当前双臂间距 ${interArmDistanceMm.toFixed(2)} mm`} · ${safety.limiting_point || `${interArm?.left_geometry || "左侧"} ↔ ${interArm?.right_geometry || "右侧"}`}`
      : interArmDistanceMm == null
        ? "等待双臂间距"
        : `双臂最小间距 ${interArmDistanceMm.toFixed(2)} mm · ${interArm.left_geometry || "左侧"} ↔ ${interArm.right_geometry || "右侧"}`
    : (safety.geometry_mode === "measured-opening" ? "实测开度几何 · " : "") + (tableDistanceMm == null
    ? "等待精确距离"
    : tableDistanceMm >= 0
      ? `桌面几何净距 ${tableDistanceMm.toFixed(2)} mm`
      : `低于桌面 ${Math.abs(tableDistanceMm).toFixed(2)} mm`);
  const noticeCollision = !stale && (
    tableCollision || interArmCollision || (!readOnly && candidateCollision)
  );
  const noticeWarning = !noticeCollision && !stale && (
    tableWarning || interArmWarning || Boolean(safety.proximity_warning)
    || (!readOnly && Boolean(safety.candidate_proximity_warning))
  );
  updateSafetyNotice({
    sequence,
    safety,
    collision: noticeCollision,
    proximityWarning: noticeWarning,
    intervention,
    candidateOnly,
    distanceMm: interArmLimiting ? interArmDistanceMm : tableDistanceMm,
    distanceLabel,
    hardMarginMm,
    warningMarginMm,
  });

  const milliseconds = safety.guard_latency_ms == null ? null : `${Number(safety.guard_latency_ms).toFixed(2)} ms`;
  const responseLabels = {
    hold_previous_successful_action: "保持上一安全动作",
    hold_measured_pose: "保持实测姿态",
    hold_measured_pose_deadline: "超时保持实测姿态",
    monitor_only_allow_candidate: "仅告警，候选动作未改写",
    allow_candidate: "候选动作通过",
    background_projection_revalidated: "后台投影复核通过",
    checked_segment_continue: "执行已校验后半段",
    checked_segment_hold: "已校验保持动作",
    async_checked_motion: "独立检测已通过",
    async_checked_hold: "等待检测，保持已校验目标",
    async_collision_hold: "保持上一放行目标，等待后续动作通过",
  };
  renderMetricCells(dom.twin_metrics, [
    ["数据新鲜度", control?.age_s == null ? null : `${Number(control.age_s).toFixed(2)} s`, stale],
    ["桌面几何净距", !safety.monitoring_enabled ? "未配置" : tableDistanceMm == null
      ? "等待精确距离"
      : `${tableDistanceMm.toFixed(2)} mm · 硬限制阈值 ${Number.isFinite(tableHardMm) ? tableHardMm.toFixed(2) : "—"} mm`, tableCollision, tableWarning],
    ["双臂实际净距", !safety.monitoring_enabled ? "未配置" : interArmDistanceMm == null
      ? "等待双臂间距"
      : `${interArmDistanceMm.toFixed(2)} mm · 硬限制阈值 ${Number.isFinite(interArmHardMm) ? interArmHardMm.toFixed(2) : "—"} mm · ${interArm?.left_geometry || "左侧"} ↔ ${interArm?.right_geometry || "右侧"}`, interArmCollision, interArmWarning],
    [readOnly ? "实测姿态碰撞" : "碰撞检测", stale ? "遥测中断，无法判定" : !safety.monitoring_enabled ? "未配置" : collision ? "硬限制" : proximityWarning ? "接近预警" : "安全", collision, proximityWarning || stale],
    ["模型候选预测", stale ? "遥测中断，结果已过期" : !safety.decision_available ? "等待模型结果" : candidateCollision ? "预测路径存在风险（不代表当前姿态碰撞）" : "预测路径安全", false, !stale && candidateCollision],
    ["处理结果", responseLabels[safety.response] || safety.response || "等待决策", collision],
    ["Guard 耗时", milliseconds, Number(safety.guard_latency_ms) > Number(safety.guard_budget_ms)],
    ["碰撞检测配置", safety.check_fps == null ? null : `${Number(safety.check_fps)} Hz`],
  ], "twin-metric");
}

async function pollTwin() {
  if (state.destroyed) return;
  if (document.hidden || !state.twinVisible || !state.jobId) {
    state.twinTimer = window.setTimeout(pollTwin, 500);
    return;
  }
  try {
    const control = await api(`/deployments/${encodeURIComponent(state.jobId)}/control-telemetry`);
    if (!state.destroyed && !document.hidden) renderTwin(control);
  } catch (error) {
    if (!isAbortLike(error) && state.twinTelemetry?.available) {
      renderTwin({ ...state.twinTelemetry, age_s: 99 });
    }
  } finally {
    if (!state.destroyed) state.twinTimer = window.setTimeout(pollTwin, state.twinPollIntervalMs);
  }
}

function initializeForm(job) {
  const config = job.config;
  const useDefaultSafety = job.task_session?.runtime_configured === false;
  const proposeExecute = job.task_session?.runtime_configured === false
    && config.hardware_mode === "read-only"
    && config.server_mode !== "mock"
    && ["ready_to_start", "waiting_approval"].includes(job.status)
    && state.bootstrap?.constraints?.physical_execute_available === true
    && robotById(config.robot_id)?.execution_supported === true;
  dom.robot_id.replaceChildren(...(state.bootstrap.robots || []).map((robot) => {
    const option = document.createElement("option"); option.value = robot.id; option.textContent = robot.label || robot.id; return option;
  }));
  dom.robot_id.value = config.robot_id;
  dom.robot_host.value = config.robot_host;
  dom.robot_port.value = config.robot_ssh_port;
  dom.task.value = config.task;
  dom.presence_gate_enabled.value = config.presence_gate?.enabled === true ? "enabled" : "disabled";
  renderModeOptions(config);
  dom.hardware_mode.value = proposeExecute ? "execute" : config.hardware_mode;
  const robot = robotById(config.robot_id);
  const effectorProfile = state.bootstrap?.effector_profiles?.find(
    (profile) => String(profile.id) === String(config.effector_profile)
  );
  dom.twin_geometry_note.textContent = effectorProfile?.collision_geometry_status === "provisional"
    ? "G2_dahuan 末端几何已接入 · 真机测试已授权"
    : "碰撞几何已匹配";
  dom.twin_geometry_note.className = effectorProfile?.collision_geometry_status === "provisional"
    ? "is-warning"
    : "";
  state.usableSafetyProfiles = robot?.safety_profiles || [];
  dom.workspace_safety.checked = useDefaultSafety || Boolean(config.safety?.workspace_safety_enabled);
  dom.safety_profile.dataset.selectedProfile = config.safety?.safety_profile_id || "";
  dom.low_pass_filter.checked = config.safety?.low_pass_filter_enabled !== false;
  dom.effector_low_pass_filter.checked = config.safety?.effector_low_pass_filter_enabled !== false;
  dom.hold_effectors.checked = Boolean(config.safety?.hold_effectors);
  const alignmentSurface = state.bootstrap?.tools?.alignment_surface;
  dom.safety_availability.hidden = false;
  dom.safety_availability.className = "safety-availability";
  if (alignmentSurface?.available) {
    const freshness = alignmentSurface.fresh
      ? `标定于 ${Math.max(0, Number(alignmentSurface.age_s || 0)).toFixed(0)} 秒前`
      : "标定已超过 5 分钟，执行前需要重新识别";
    dom.safety_availability.textContent = `AprilTag 桌面高度 ${Number(alignmentSurface.height_m).toFixed(3)} m（相对 base_link，${freshness}）。有效快照可用于真机测试，启动前仍需通过预检和二次确认。`;
  } else {
    dom.safety_availability.textContent = "只读验证可使用 0.750 m 默认桌面高度；真机执行前必须完成 AprilTag 标定与大寰末端几何验收。";
  }
  dom.duration_s.value = config.duration_s;
  updateDurationDisplay();
  dom.execution_fps.value = config.execution_fps;
  dom.servo_fps.value = config.servo_fps;
  dom.chunk_smoothing_mode.value = config.chunk_smoothing_mode || "savgol";
  dom.action_interpolation.value = config.action_interpolation || "linear";
  dom.arm_tracking_mode.value = config.arm_tracking_mode || "legacy";
  dom.arm_tracking_omega.value = config.arm_tracking_omega ?? 12;
  dom.rtc_prefix_mode.value = config.rtc_prefix_mode || "model";
  dom.arm_limit_profile.value = config.arm_limit_profile;
  dom.max_arm_velocity.value = config.max_arm_velocity;
  dom.max_arm_acceleration.value = config.max_arm_acceleration;
  dom.prediction_horizon.value = config.prediction_horizon;
  dom.quality_horizon.value = config.quality_horizon;
  dom.lead_margin.value = config.request_lead_margin_steps;
  dom.rtc_blend.value = config.min_rtc_blend_steps;
  dom.request_timeout.value = config.request_timeout_s;
  dom.hitl.checked = Boolean(config.human_in_loop);
  updateModeFields();
  updateHardwareFields();
  state.formInitialized = true;
  state.formDirty = proposeExecute || useDefaultSafety;
  if (useDefaultSafety && !dom.workspace_safety.checked) {
    dom.config_message.textContent = "当前机器人没有可用的碰撞安全配置，请先完成配置发布。";
  } else if (proposeExecute) {
    dom.config_message.textContent = `已预选${inferenceModeLabel(config.inference_mode, modelVersionForConfig(state.bootstrap, config, RTC_MODEL_MODES), config.rtc_inference_mode)} ${config.execution_fps} Hz 真机及碰撞安全配置；应用并通过预检后才能启动。`;
  } else if (useDefaultSafety) {
    dom.config_message.textContent = "已预选碰撞安全配置；应用并通过预检后才能启动。";
  }
}

function executeAvailable() {
  const robot = robotById(dom.robot_id.value || state.job?.config?.robot_id);
  return state.bootstrap?.constraints?.physical_execute_available === true && robot?.execution_supported === true;
}

function safetyProfileEligible(profile, execute) {
  if (!profile?.deployable) return false;
  return execute ? profile.execute_approved === true : profile.read_only_ready === true;
}

function preferredSafetyProfile(profiles, execute) {
  const eligible = profiles.filter((profile) => safetyProfileEligible(profile, execute));
  return eligible.find((profile) =>
    profile.inter_arm_collision?.geometry === "cross-chain-capsules-v2"
    && profile.inter_arm_collision.validation_status === "execute-approved"
  ) || eligible.find((profile) =>
    profile.inter_arm_collision?.geometry === "cross-chain-capsules-v2"
  ) || eligible[0];
}

function renderSafetyProfileOptions(execute) {
  const profiles = state.usableSafetyProfiles || [];
  const previous = dom.safety_profile.value
    || dom.safety_profile.dataset.selectedProfile
    || state.job?.config?.safety?.safety_profile_id
    || "";
  const placeholder = new Option("请选择安全配置", "");
  const options = profiles.map((profile) => {
    const eligible = safetyProfileEligible(profile, execute);
    const status = profile.execute_approved
      ? "真机已验收"
      : profile.read_only_ready
        ? "只读验证"
        : profile.calibrated
          ? "已标定 · 未发布"
          : "未就绪";
    const height = Number.isFinite(Number(profile.height_m))
      ? ` · 桌高 ${Number(profile.height_m).toFixed(3)} m`
      : "";
    const freshness = profile.generated
      ? profile.fresh ? " · 最新测量" : " · 历史快照"
      : "";
    const interArm = profile.inter_arm_collision?.enabled
      ? profile.inter_arm_collision.validation_status === "execute-approved"
        ? " · 双臂已验收"
        : " · 双臂只读候选"
      : "";
    const option = new Option(
      `${profile.label || profile.id}（${status}${height}${freshness}${interArm}）`,
      profile.id,
    );
    option.disabled = !eligible;
    return option;
  });
  dom.safety_profile.replaceChildren(placeholder, ...options);
  const selected = [...dom.safety_profile.options].find(
    (option) => option.value === previous && !option.disabled,
  );
  if (selected) dom.safety_profile.value = previous;
  else if (dom.workspace_safety.checked) {
    dom.safety_profile.value = preferredSafetyProfile(profiles, execute)?.id || "";
  }
  dom.safety_profile.dataset.selectedProfile = dom.safety_profile.value;
  return profiles.filter((profile) => safetyProfileEligible(profile, execute));
}

function updateHardwareFields({ userChange = false } = {}) {
  const executeOption = dom.hardware_mode.querySelector('option[value="execute"]');
  const available = executeAvailable();
  if (executeOption) executeOption.disabled = !available;
  const robot = robotById(dom.robot_id.value || state.job?.config?.robot_id);
  dom.hardware_mode_help.textContent = robot?.effectors?.mode === "absent_fixed"
    ? "该机器人未安装夹爪；双末端状态以训练集参考固定值补齐，仅允许只读观测，真机执行永久锁定。"
    : available
      ? ""
      : "真机执行保持锁定：需要站点 execute_enabled 和控制服务 --enable-execute 双重放行。";
  const execute = dom.hardware_mode.value === "execute";
  const profiles = renderSafetyProfileOptions(execute);
  const editable = ["ready_to_start", "waiting_approval", "succeeded"].includes(state.job?.status)
    && !state.busy;
  dom.workspace_safety.disabled = !editable || !profiles.length;
  dom.safety_profile.disabled = !editable || !dom.workspace_safety.checked || !profiles.length;
  if (!profiles.length) {
    dom.workspace_safety.checked = false;
    dom.safety_availability.textContent = execute
      ? "大寰安全几何尚未完成真机验收，碰撞拦截仅允许在只读观测中验证。"
      : "当前机器人没有可用的只读碰撞安全配置，请先完成配置发布。";
  } else if (!execute && profiles.length && !state.bootstrap?.tools?.alignment_surface?.available) {
    dom.safety_availability.textContent = "可开启只读碰撞拦截验证；当前使用 0.750 m 默认桌面高度，不会向机器人发送动作。";
  }
}

function renderModeOptions(config) {
  const model = modelById(config.model_id);
  const version = modelVersionForConfig(state.bootstrap, config, RTC_MODEL_MODES);
  const modes = (model?.supported_modes || ["sync", "async"]).flatMap((mode) =>
    mode === "rtc" ? ["rtc-trained", "rtc-guided"] : [mode]);
  dom.inference_mode.replaceChildren(...modes.map((mode) => {
    const option = document.createElement("option");
    option.value = mode;
    option.textContent = inferenceModeLabel(mode, version);
    option.disabled = mode === "rtc-trained" && !trainedRtcSupported(version);
    if (option.disabled) option.title = "所选权重未使用 Training RTC 训练";
    return option;
  }));
  dom.inference_mode.value = inferenceSelectionValue(config, version);
}

function updateModeFields() {
  const mode = parseInferenceSelection(dom.inference_mode.value).inference_mode;
  document.querySelectorAll("[data-runtime-modes]").forEach((field) => {
    field.hidden = !field.dataset.runtimeModes.split(" ").includes(mode);
  });
}

function updateDurationDisplay() {
  const continuous = Number(dom.duration_s.value) === 0;
  dom.duration_unit.textContent = continuous ? "（一直执行）" : "秒";
  dom.duration_unit.classList.toggle("is-continuous", continuous);
  dom.duration_unit.parentElement.classList.toggle("is-continuous", continuous);
}

function setFormEnabled(job) {
  const applying = job.workstation?.switching_deployment_id;
  const priority = job.workstation?.priority_apply;
  const priorityHere = priority?.target_id === job.id;
  const editable = ["ready_to_start", "waiting_approval", "succeeded"].includes(job.status) && !state.busy && !applying;
  [...dom.runtime_form.elements].forEach((element) => { element.disabled = !editable; });
  dom.runtime_form.querySelectorAll("[data-instance-locked]").forEach((element) => { element.disabled = true; });
  updateHardwareFields();
  dom.probe_robot.disabled = state.busy;
  dom.console_probe_robot.disabled = state.busy || !monitorView;
  dom.config_lock.textContent = editable ? "可编辑" : applying ? "正在安全接管" : job.status === "deploying" ? "模型加载中" : "配置已冻结";
  dom.config_message.textContent = state.lastApplyError || (priorityHere && priority?.status === "failed"
    ? `真机配置未应用：${job.workstation?.switch_error || priority.error || "请核实机器人与模型状态"}。新任务仍被禁止。`
    : priorityHere && priority?.status === "applied"
      ? "真机配置已应用；请核实机器人状态后恢复新任务。"
    : editable
    ? "真机配置确认后会优先暂停旧任务、结束旧会话、安全复位，再应用配置；任何结果不明都会停止接管。"
    : applying ? `正在${({ reserved: "保留工作站", pausing: "确认旧任务安全保持", stopping: "结束旧会话", resetting: "安全复位", activating: "应用模型配置" })[job.workstation?.priority_apply?.status] || "应用配置"}；请等待结果。`
    : "模型加载完成后可以配置；任务开始后配置保持冻结。");
}

function statusLabel(status) {
  return ({ queued: "准备中", validating: "配置校验", preflight: "冻结配置", deploying: "模型加载中", model_ready: "模型已就绪",
    waiting_approval: "迁移旧会话", ready_to_start: "等待开始", task_starting: "任务启动中", cancelling: "正在取消", running: "任务运行中",
    pausing: "正在暂停", paused: "已暂停", stopping: "正在结束", succeeded: "任务已结束", cancelled: "启动已取消",
    recovery_required: "需要恢复确认", failed: "作业失败", rejected: "任务已拒绝", model_stopped: "模型已关闭" })[status] || status;
}

function renderControls(job) {
  const automatic = job.workstation?.mode === "automatic";
  const status = job.status;
  const config = job.config;
  const model = modelById(config.model_id);
  const robot = robotById(config.robot_id);
  const candidateOnly = job.model_service?.candidate_only === true;
  const isCurrent = job.workstation?.active_deployment_id === (job.model_service?.deployment_id || job.id);
  dom.task_status.textContent = candidateOnly ? "待应用配置" : statusLabel(status);
  dom.task_status.className = `runtime-status ${status === "running" ? "is-ready" : ["ready_to_start", "waiting_approval", "paused"].includes(status) ? "is-waiting" : ["failed", "rejected", "model_stopped"].includes(status) ? "is-failed" : "is-active"}`;
  dom.control_model.textContent = `${model?.label || config.model_id} · ${job.model_service?.healthy ? "健康" : "未就绪"}`;
  dom.control_model_version.textContent = modelVersionLabel(state.bootstrap, config);
  dom.control_model_version.title = dom.control_model_version.textContent;
  dom.control_robot.textContent = `${robot?.label || config.robot_id} · ${config.robot_host}`;
  dom.control_mode.textContent = inferenceModeLabel(config.inference_mode, modelVersionForConfig(state.bootstrap, config, RTC_MODEL_MODES), config.rtc_inference_mode);
  dom.control_hardware.textContent = HARDWARE_LABELS[config.hardware_mode] || config.hardware_mode;
  const execute = config.hardware_mode === "execute";
  dom.safe_hold.textContent = job.task_session?.safe_hold_available
    ? execute ? "已接入 · 100 Hz 保持" : "软暂停可用"
    : "等待客户端确认";
  const notes = {
    deploying: "模型正在加载。任务参数将在模型就绪后开放。",
    ready_to_start: execute
      ? "真机配置已通过预检；首次点击启动会进入防误触确认，再次点击才会启动。"
      : "模型服务已就绪，任务尚未开始，也没有发送机器人动作。",
    task_starting: "机器人客户端正在初始化和握手；可点击“取消启动”。",
    running: execute
      ? config.human_in_loop
        ? "人在环路会话运行中；平板每次采集对应一个包裹，录制绑定后先保持 HOLD，按 Y 才开启 Policy。急停应保持可达。"
        : "真机动作执行中；急停应保持可达。暂停会清空动作计划并持续保持当前位置。"
      : "只读机器人客户端正在读取真机状态与相机，但不会下发动作。",
    paused: execute
      ? "真机软暂停中；客户端持续保持当前位置，继续时会从当前姿态建立新计划。"
      : "只读软暂停中；GDK 与模型连接保留，继续时从当前观测建立新计划。",
    succeeded: "任务会话已结束。模型服务保持运行，可继续复用或按所有权关闭。",
    failed: job.error || job.message,
  };
  const effectorNote = robot?.effectors?.mode === "absent_fixed"
    ? " 当前机器人无夹爪，模型输入中的双末端状态来自固定训练参考值。"
    : "";
  const task = job.task_view || {};
  const source = task.source === "scheduler" ? `调度批次 ${task.request_id || ""}` : "人工调试";
  const callback = task.callback?.enabled ? ` · 通知 ${task.callback.status}` : "";
  dom.control_note.textContent = candidateOnly ? "配置已保存但尚未应用。确认参数后点击“保存并应用到工作站”；应用前不能启动任务或复位机器人。" :
    `${source} · ${task.status || status}${callback}。` + (notes[status] || job.message || "正在准备作业。") + effectorNote;
  const startableState = ["ready_to_start", "succeeded"].includes(status);
  if (execute && config.human_in_loop && !candidateOnly) {
    const tabletStatus = job.task_session?.tablet_start?.status;
    const tabletNotes = {
      listening: "正在连接平板录制监听。",
      ready: "已获准启动；现在可在平板点击开始采集。检测到新录制后自动进入 HOLD，按 Y 才开启 Policy。",
      awaiting_clear: "检测到已有录制；请先在平板结束并保存，再开始一份新录制。",
      ambiguous: "检测到多份活动录制；请先结束多余录制。",
      detected: "检测到新录制，正在按网页相同的准入流程启动并绑定 HOLD。",
      rejected: "平板触发的任务启动未通过最新准入检查；请核对页面状态。",
      unavailable: "平板录制监听不可用；仍可从页面启动。",
    };
    if (job.task_session?.start_source === "tablet" && ["task_starting", "running", "paused"].includes(status)) {
      dom.control_note.textContent += " 本次任务由平板新录制启动，初始为 HOLD；按 Y 开启 Policy。";
    } else if (startableState && tabletNotes[tabletStatus]) {
      dom.control_note.textContent += ` ${tabletNotes[tabletStatus]}`;
    }
  }
  dom.execute_gate.hidden = automatic || !(execute && startableState);
  if (!startableState || !execute) state.startArmed = false;
  if (state.formDirty) state.startArmed = false;
  const canStart = startableState && !state.formDirty && job.task_view?.allowed_actions?.includes("start");
  const startBlockers = {
    NOT_ACCEPTING: "新任务入口已关闭；核实后点击“核实后允许新任务”",
    MODE_CONFLICT: "当前是自动模式，只能由调度 API 启动",
    ROBOT_BUSY: "机器人已有任务占用或需要人工核实",
    RESET_REQUIRED: "机器人尚未成功复位；请先检测并完成复位",
    SCHEME_NOT_ACTIVE: "这份配置尚未应用到工作站",
    NOT_READY: "模型或运行配置尚未就绪",
  };
  const startBlocked = candidateOnly ? "请先保存并应用配置" : state.formDirty ? "请先应用未保存的运行参数" :
    startBlockers[job.task_view?.unavailable_reasons?.start] || "当前状态不可开始";
  if (startableState && !canStart) dom.control_note.textContent += ` 无法启动：${startBlocked}。`;
  const canPause = job.task_view?.allowed_actions?.includes("pause");
  const canResume = job.task_view?.allowed_actions?.includes("resume");
  const canStop = job.task_view?.allowed_actions?.includes("stop");
  dom.start.hidden = automatic;
  dom.start.disabled = state.busy || automatic || !canStart;
  dom.start.textContent = execute
    ? state.startArmed ? "再次点击确认启动真机" : status === "succeeded" ? "再次启动真机" : "启动真机"
    : status === "succeeded" ? "再次开始观测" : "开始观测";
  dom.start.title = canStart
    ? execute ? "需要连续点击两次以防误触" : "开始观测任务"
    : startBlocked;
  dom.pause.disabled = state.busy || !canPause;
  dom.pause.title = canPause ? "清空动作计划并进入软暂停" : "仅任务运行时可暂停";
  dom.resume.disabled = state.busy || !canResume;
  dom.resume.title = canResume ? "重新读取当前姿态并等待首个新动作计划" : "仅任务暂停后可继续";
  dom.stop.disabled = state.busy || !canStop;
  dom.stop.textContent = status === "task_starting" ? "取消启动" : execute ? "结束真机任务" : "结束任务";
  dom.stop.title = canStop ? dom.stop.textContent : "当前没有可结束的任务";
  const reset = job.robot_reset || {};
  const resetRunning = ["preparing", "running"].includes(reset.status);
  const resettableState = ["ready_to_start", "succeeded", "cancelled", "failed", "rejected", "model_stopped", "recovery_required"].includes(status);
  const resetSupported = robot?.reset_supported === true;
  const canReset = isCurrent && !candidateOnly && resetSupported && resettableState && !job.task_session?.remote_started && !resetRunning;
  dom.reset_robot.disabled = state.busy || !canReset;
  if (!canReset) state.resetArmed = false;
  dom.reset_robot.textContent = resetRunning
    ? "机器人复位中…"
    : state.resetArmed ? "再次点击确认复位" : reset.status === "failed" ? "重试机器人复位" : "机器人复位";
  dom.reset_robot.title = resetRunning
    ? "正在复位双臂、头部和腰部"
    : !resetSupported
      ? "控制服务或机器人配置未开放复位"
      : !resettableState || job.task_session?.remote_started
        ? "请先结束当前任务"
        : "需要连续点击两次；复位包含大寰夹爪松开、夹紧、松开和身体关节复位";
  dom.robot_reset_status.className = `robot-reset-status ${resetRunning ? "is-running" : reset.status === "succeeded" ? "is-success" : reset.status === "failed" ? "is-failed" : ""}`.trim();
  dom.robot_reset_status.textContent = job.workstation?.robot_gate?.required
    ? `下一任务已禁止：${job.workstation.robot_gate.reason}。${reset.message || "请检测机器人后重试复位。"}`
    : reset.message || "复位会先让大寰夹爪松开、夹紧并最终松开，再复位双臂、头部和腰部。";
  const modelReady = job.model_service?.status === "ready" || job.model_service?.healthy === true;
  const canCloseModel = isCurrent && !candidateOnly && modelReady && ["ready_to_start", "succeeded", "failed", "rejected"].includes(status);
  dom.close_model.disabled = state.busy || !canCloseModel;
  dom.close_model.title = canCloseModel ? "关闭当前模型服务并释放 GPU" : modelReady ? "请先结束当前任务" : "当前没有可关闭的模型服务";
  if (!state.closeModelArmed) dom.close_model.textContent = "关闭模型服务";
  dom.ownership_note.textContent = modelReady
    ? "结束任务后可关闭当前受管模型服务；不受模型启动任务归属限制。"
    : "当前模型服务未运行或已经关闭。";
}

function renderEvents(events) {
  const key = JSON.stringify((events || []).slice(-40));
  if (dom.events.dataset.renderKey === key) return;
  dom.events.dataset.renderKey = key;
  dom.events.replaceChildren(...(events || []).slice(-40).reverse().map((event) => {
    const item = document.createElement("li");
    const time = document.createElement("time"); time.textContent = formatTime(event.time);
    const message = document.createElement("span"); message.textContent = event.message || event.status;
    item.append(time, message); return item;
  }));
}

function render(job) {
  state.job = job;
  if (state.destroyed || document.hidden) return;
  const priority = job.workstation?.priority_apply;
  if (priority?.target_id === job.id && ["applied", "failed"].includes(priority.status) &&
      state.priorityToastStatus !== priority.status) {
    state.priorityToastStatus = priority.status;
    toast(priority.status === "applied" ? "真机配置已应用；请核实状态后恢复新任务"
      : `真机配置接管失败：${job.workstation?.switch_error || priority.error || "请核实机器人状态"}`,
    priority.status === "applied" ? "success" : "error");
  }
  const model = modelById(job.config.model_id);
  const robot = robotById(job.config.robot_id);
  dom.route_label.textContent = `${robot?.label || job.config.robot_id} → ${model?.label || job.config.model_id}`;
  dom.job_id.textContent = job.id;
  const switchView = document.getElementById("view-switch");
  const monitorId = job.workstation?.current_task_id || job.workstation?.active_deployment_id;
  switchView.href = monitorView ? "./session.html" :
    monitorId ? `./monitor?id=${encodeURIComponent(monitorId)}` : "./monitor";
  switchView.textContent = monitorView ? "运行配置" : "机器人控制台";
  const active = job.workstation?.active_tasks?.includes(job.id);
  const standby = ["ready_to_start", "succeeded"].includes(job.status);
  document.getElementById("console-title").textContent = monitorView
    ? active ? "机器人控制台 · 任务执行中" : standby ? "机器人控制台 · 待命" : "机器人控制台 · 状态核实"
    : "运行配置";
  document.getElementById("console-description").textContent = monitorView
    ? "三维状态、碰撞信息和相机画面来自当前会话；无实时数据时会显示断开或过期状态。"
    : "保存并应用模型运行参数、安全配置与包裹完成条件；控制权在机器人控制台交接。";
  dom.session_main.dataset.phase = ["running", "paused", "pausing", "stopping", "succeeded", "cancelled"].includes(job.status)
    ? "runtime"
    : ["ready_to_start", "waiting_approval"].includes(job.status)
      ? "configuration"
      : "loading";
  if (!state.formInitialized || (!state.formDirty && ["ready_to_start", "waiting_approval"].includes(job.status))) initializeForm(job);
  renderModel(job);
  renderTelemetry(job);
  renderPresenceGate(job);
  renderTwin(job.telemetry?.control || {});
  setFormEnabled(job);
  renderControls(job);
  renderEvents(job.events);
}

function runtimePayload() {
  const current = state.job.config;
  const selection = parseInferenceSelection(dom.inference_mode.value);
  const version = modelVersionForConfig(state.bootstrap, current, RTC_MODEL_MODES);
  return {
    ...current,
    // Timing and reset policy remain server-owned; only the completion method
    // is selected on the runtime page after the model is ready.
    presence_gate: {
      enabled: dom.presence_gate_enabled.value === "enabled",
    },
    robot_id: dom.robot_id.value,
    robot_host: dom.robot_host.value.trim(),
    robot_ssh_port: Number(dom.robot_port.value),
    task: dom.task.value.trim(),
    inference_mode: selection.inference_mode,
    rtc_inference_mode: selection.rtc_inference_mode || rtcInferenceMode(current, version),
    hardware_mode: dom.hardware_mode.value,
    duration_s: Number(dom.duration_s.value),
    execution_fps: Number(dom.execution_fps.value),
    servo_fps: Number(dom.servo_fps.value),
    chunk_smoothing_mode: dom.chunk_smoothing_mode.value,
    action_interpolation: dom.action_interpolation.value,
    arm_tracking_mode: dom.arm_tracking_mode.value,
    arm_tracking_omega: Number(dom.arm_tracking_omega.value),
    rtc_prefix_mode: dom.rtc_prefix_mode.value,
    arm_limit_profile: dom.arm_limit_profile.value,
    max_arm_velocity: Number(dom.max_arm_velocity.value),
    max_arm_acceleration: Number(dom.max_arm_acceleration.value),
    prediction_horizon: Number(dom.prediction_horizon.value),
    quality_horizon: Number(dom.quality_horizon.value),
    request_lead_margin_steps: Number(dom.lead_margin.value),
    min_rtc_blend_steps: Number(dom.rtc_blend.value),
    request_timeout_s: Number(dom.request_timeout.value),
    human_in_loop: dom.hitl.checked,
    safety: {
      ...current.safety,
      low_pass_filter_enabled: dom.low_pass_filter.checked,
      effector_low_pass_filter_enabled: dom.effector_low_pass_filter.checked,
      hold_effectors: dom.hold_effectors.checked,
      workspace_safety_enabled: dom.workspace_safety.checked,
      safety_profile_id: dom.workspace_safety.checked ? dom.safety_profile.value : "",
    },
  };
}

function armForSecondClick(kind) {
  const armedKey = `${kind}Armed`;
  const timerKey = `${kind}ArmTimer`;
  state[armedKey] = true;
  window.clearTimeout(state[timerKey]);
  state[timerKey] = window.setTimeout(() => {
    state[armedKey] = false;
    state[timerKey] = null;
    if (state.job) renderControls(state.job);
  }, 5000);
  if (state.job) renderControls(state.job);
}

function disarm(kind) {
  const armedKey = `${kind}Armed`;
  const timerKey = `${kind}ArmTimer`;
  state[armedKey] = false;
  window.clearTimeout(state[timerKey]);
  state[timerKey] = null;
}

function startTask() {
  if (dom.start.disabled || !state.job) return;
  if (state.job.config?.hardware_mode !== "execute") {
    postAction("start");
    return;
  }
  if (!state.startArmed) {
    armForSecondClick("start");
    toast("请确认现场已清场且急停可用，然后再次点击启动真机");
    return;
  }
  disarm("start");
  postAction("start");
}

function resetRobot() {
  if (dom.reset_robot.disabled || !state.job) return;
  if (!state.resetArmed) {
    armForSecondClick("reset");
    toast("请清空工作区和夹爪内部、确认急停可用，然后再次点击机器人复位");
    return;
  }
  disarm("reset");
  postAction("reset-robot");
}

async function postAction(action, body = {}) {
  if (state.busy) return false;
  state.busy = true;
  if (state.job) {
    setFormEnabled(state.job);
    renderControls(state.job);
  }
  try {
    const result = await api(`/deployments/${encodeURIComponent(state.jobId)}/${action}`, {
      method: "POST", headers: csrfHeaders(), body: JSON.stringify(body),
    });
    render(result);
    toast(({ start: "任务正在启动",
      pause: "机器人客户端已暂停", resume: "正在重新启动机器人客户端", stop: "任务已结束，模型保持运行",
      "reset-robot": "机器人复位已开始；将执行大寰夹爪松开—夹紧—松开和身体关节复位", "close-model": "模型服务已关闭" })[action] || "操作已完成");
    return true;
  } catch (error) {
    if (isAbortLike(error)) return false;
    toast(error.message, "error");
    return false;
  } finally {
    state.busy = false;
    if (state.job) {
      setFormEnabled(state.job);
      renderControls(state.job);
    }
  }
}

async function applyConfiguration() {
  if (state.busy || !state.job) return false;
  state.busy = true;
  state.lastApplyError = "";
  let targetId = state.jobId;
  let failure = "";
  try {
    const runtime = runtimePayload();
    const physical = runtime.hardware_mode === "execute";
    if (!window.confirm(physical
      ? "确认将此真机配置优先应用到工作站？系统会先暂停并确认旧任务安全保持，再结束旧会话、自动复位，复位成功后应用配置。任何步骤结果不明都将停止接管并禁止新任务。"
      : "应用此配置到工作站？运行参数会先经过预检；替换模型时会暂停接收新任务，完成后仍需人工核实并恢复。")) return false;
    setFormEnabled(state.job);
    renderControls(state.job);
    if (physical) {
      const result = await api("/workstation/apply-priority", {
        method: "POST", headers: csrfHeaders(),
        body: JSON.stringify({ deployment_id: state.jobId, runtime }),
      });
      targetId = result.priority_apply?.target_id;
      if (!targetId) throw new Error("服务器未返回真机配置接管编号，请检查工作站状态");
      state.formDirty = false;
      window.location.assign(`./session.html?id=${encodeURIComponent(targetId)}`);
      return true;
    }
    const station = await api("/workstation");
    if (station.active_tasks?.length || station.switching_deployment_id) {
      throw new Error("机器人正被任务占用或正在应用配置；本次请求已拒绝，不会排队");
    }
    const configured = await api(`/deployments/${encodeURIComponent(state.jobId)}/configure`, {
      method: "POST", headers: csrfHeaders(), body: JSON.stringify(runtime),
    });
    targetId = configured.id;
    const afterSave = await api("/workstation");
    if (afterSave.active_deployment_id !== targetId) {
      await api("/workstation/activate", {
        method: "POST", headers: csrfHeaders(), body: JSON.stringify({ deployment_id: targetId }),
      });
    }
    state.formDirty = false;
    if (targetId !== state.jobId) {
      window.location.assign(`./session.html?id=${encodeURIComponent(targetId)}`);
    } else {
      render(await api(`/deployments/${encodeURIComponent(targetId)}`));
      toast("配置已应用；请核实状态后恢复新任务");
    }
    return true;
  } catch (error) {
    if (isAbortLike(error)) return false;
    failure = error.message;
    state.lastApplyError = failure;
    if (targetId !== state.jobId) {
      state.jobId = targetId;
      window.history.replaceState(null, "", `./session.html?id=${encodeURIComponent(targetId)}`);
      try {
        render(await api(`/deployments/${encodeURIComponent(targetId)}`));
        state.formDirty = false;
      } catch {
        // The saved configuration remains available by its URL for later inspection.
      }
    }
    toast(failure, "error");
    return false;
  } finally {
    state.busy = false;
    if (state.job) {
      setFormEnabled(state.job);
      renderControls(state.job);
    }
    if (failure) showRuntimeValidationErrors(failure);
  }
}

function showRuntimeValidationErrors(message) {
  let errors = [];
  try {
    const parsed = JSON.parse(message);
    if (Array.isArray(parsed)) errors = parsed;
  } catch {
    errors = [];
  }
  dom.runtime_form.querySelectorAll("input, textarea, select").forEach((element) => element.setCustomValidity(""));
  if (!errors.length) {
    dom.config_message.textContent = message;
    return;
  }
  const aliases = {
    prediction_horizon: "prediction-horizon", quality_horizon: "quality-horizon",
    execution_fps: "execution-fps", servo_fps: "servo-fps", request_timeout_s: "request-timeout",
    chunk_smoothing_mode: "chunk-smoothing-mode",
    action_interpolation: "action-interpolation",
    request_lead_margin_steps: "lead-margin", min_rtc_blend_steps: "rtc-blend",
    inference_mode: "inference-mode", rtc_inference_mode: "inference-mode", human_in_loop: "hitl", task: "task",
    "safety.safety_profile_id": "safety-profile",
    "safety.workspace_safety_enabled": "workspace-safety",
    "presence_gate.enabled": "presence-gate-enabled",
  };
  errors.forEach((item) => {
    const element = document.getElementById(aliases[item.field] || String(item.field || "").replaceAll("_", "-"));
    if (element) element.setCustomValidity(String(item.message || "参数无效"));
  });
  const first = dom.runtime_form.querySelector(":invalid");
  first?.reportValidity();
  first?.focus();
  dom.config_message.textContent = `预检未通过：${errors.map((item) => item.message).join("；")}`;
}

async function poll() {
  if (state.destroyed || !state.jobId) return;
  if (document.hidden) {
    state.pollTimer = window.setTimeout(poll, 1500);
    return;
  }
  try {
    const job = await api(`/deployments/${encodeURIComponent(state.jobId)}`);
    render(job);
    setConnection("online", "控制面在线");
    if (state.pollErrorShown) toast("控制面连接已恢复", "success");
    state.pollFailures = 0;
    state.pollErrorShown = false;
  } catch (error) {
    if (!isAbortLike(error)) {
      state.pollFailures += 1;
      setConnection("offline", state.pollFailures < 3 ? "正在重连" : "连接失败");
      if (state.pollFailures >= 3 && !state.pollErrorShown) {
        toast(`会话连接连续失败：${error.message}。页面会自动重试。`, "error");
        state.pollErrorShown = true;
      }
    }
  } finally {
    const active = ["task_starting", "running", "pausing", "paused", "stopping"].includes(state.job?.status);
    if (!state.destroyed) state.pollTimer = window.setTimeout(poll, active ? 500 : 1200);
  }
}

async function bootstrap() {
  state.bootstrap = await api("/bootstrap");
  configureAlignmentLink(state.bootstrap, () => ({
    robotHost: state.job?.config?.robot_host || "",
    robotId: state.job?.config?.robot_id || "",
  }));
  state.csrfToken = state.bootstrap.csrf_token || "";
  state.bootstrap.models = Array.isArray(state.bootstrap.models) ? state.bootstrap.models : [];
  state.bootstrap.robots = Array.isArray(state.bootstrap.robots) ? state.bootstrap.robots : [];
  if (!state.jobId) {
    const station = await api("/workstation");
    const pending = station.console_deployments?.find((item) => ["candidate", "preparing"].includes(item.phase));
    const id = monitorView ? station.current_task_id || station.active_deployment_id :
      station.console_deployments?.find((item) => item.id === station.active_deployment_id)?.id || pending?.id || station.console_deployments?.find((item) => item.phase === "standby")?.id;
    if (id) {
      window.location.replace(`${monitorView ? "./monitor" : "./session.html"}?id=${encodeURIComponent(id)}`);
      return;
    }
    renderOverview(station);
    setConnection("online", "控制面在线");
    return;
  }
  const station = await api("/workstation");
  const requested = station.console_deployments?.find((item) => item.id === state.jobId);
  if (monitorView && (requested?.phase === "candidate" || requested?.phase === "preparing")) {
    window.location.replace(station.landing_url || "./monitor");
    return;
  }
  if (!monitorView && !requested) {
    window.location.replace("./session.html");
    return;
  }
  if (monitorView) initializeTwinViewer();
  const job = await api(`/deployments/${encodeURIComponent(state.jobId)}`);
  dom.session_content.hidden = false;
  render(job);
  setConnection("online", "控制面在线");
  state.pollTimer = window.setTimeout(poll, 700);
  if (monitorView) state.twinTimer = window.setTimeout(pollTwin, state.twinPollIntervalMs);
}

function renderOverview(station) {
  const entries = station.console_deployments || [];
  const switchView = document.getElementById("view-switch");
  switchView.href = monitorView ? "./session.html" : "./monitor";
  switchView.textContent = monitorView ? "运行配置" : "机器人控制台";
  document.getElementById("console-title").textContent = monitorView ? "机器人控制台" : "运行配置";
  document.getElementById("console-description").textContent = entries.length
    ? monitorView ? "当前没有活动任务。运行参数请到运行设置查看。" : "确认运行参数后应用到工作站。"
    : "当前没有可用部署。请先准备模型并保存运行配置。";
  document.getElementById("console-empty").hidden = monitorView || entries.length > 0;
  document.getElementById("monitor-empty").hidden = !monitorView;
  dom.route_label.textContent = "工作站总览";
}

window.addEventListener("nexus-workstation-change", (event) => {
  const station = event.detail;
  if (!state.jobId) {
    renderOverview(station);
    return;
  }
  if (!state.job) return;
  if (monitorView && !state.job.task_session?.accepted && !station.active_tasks?.includes(state.jobId) &&
      station.active_deployment_id && station.active_deployment_id !== state.jobId) {
    window.location.replace(`./monitor?id=${encodeURIComponent(station.active_deployment_id)}`);
    return;
  }
  // Follow a new scheduler batch only from its idle deployment console.
  // Historical task pages and unsaved operator edits stay where they are.
  if (station.mode === "automatic" && !state.formDirty && !state.job.task_session?.accepted &&
      state.job.status === "ready_to_start") {
    const active = station.console_deployments?.find((item) =>
      item.phase === "active" && item.source === "scheduler" && item.deployment_id === state.jobId && item.id !== state.jobId);
    if (active) {
      window.location.replace(`./monitor?id=${encodeURIComponent(active.id)}`);
      return;
    }
  }
  if (state.job.workstation?.mode !== station.mode) state.startArmed = false;
  state.job.workstation = station;
  renderControls(state.job);
});

dom.runtime_form.addEventListener("input", () => {
  dom.runtime_form.querySelectorAll("input, textarea, select").forEach((element) => element.setCustomValidity(""));
  state.formDirty = true;
  state.lastApplyError = "";
  updateDurationDisplay();
  if (state.job) renderControls(state.job);
});
dom.runtime_form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await applyConfiguration();
});
dom.inference_mode.addEventListener("change", updateModeFields);
dom.hardware_mode.addEventListener("change", () => updateHardwareFields({ userChange: true }));
dom.workspace_safety.addEventListener("change", updateHardwareFields);
document.querySelectorAll("[data-twin-view]").forEach((button) => {
  button.addEventListener("click", () => setTwinView(button.dataset.twinView));
});
dom.twin_save_view.addEventListener("click", saveTwinView);
dom.collision_geometry_toggle.addEventListener("click", () => {
  state.collisionOverlayVisible = !state.collisionOverlayVisible;
  dom.collision_geometry_toggle.classList.toggle("is-active", state.collisionOverlayVisible);
  dom.collision_geometry_toggle.setAttribute("aria-pressed", String(state.collisionOverlayVisible));
  dom.collision_geometry_toggle.textContent = state.collisionOverlayVisible ? "隐藏碰撞体" : "显示碰撞体";
  if (state.twinViewerLoaded) dom.robot_twin.contentWindow?.postMessage(
    { type: "nexus-collision-overlay", visible: state.collisionOverlayVisible },
    state.twinViewerOrigin,
  );
});
dom.robot_id.addEventListener("change", () => {
  const robot = robotById(dom.robot_id.value);
  if (robot?.connection) { dom.robot_host.value = robot.connection.host || ""; dom.robot_port.value = robot.connection.ssh_port || 22; }
});
async function probeRobot() {
  const showProbe = (message) => {
    dom.probe_result.textContent = message;
    dom.console_probe_result.textContent = message;
  };
  showProbe("检测中…");
  try {
    const result = await api("/robots/probe", { method: "POST", headers: csrfHeaders(), body: JSON.stringify({ robot_id: dom.robot_id.value, robot_host: dom.robot_host.value.trim(), robot_ssh_port: Number(dom.robot_port.value) }) });
    const client = result.client || {};
    if (!result.reachable) showProbe("网络不可达");
    else if (!result.ssh_authenticated) showProbe(`仅网络可达 · ${client.reason || "客户端状态未知"}`);
    else if (client.status === "not_deployed") showProbe("SSH 正常 · 客户端未部署");
    else if (client.status === "invalid_client") showProbe("客户端文件、协议或机器人类型不匹配");
    else if (client.status === "running") showProbe(`客户端运行中 · PID ${client.processes?.[0]?.pid ?? "—"}`);
    else if (client.status === "idle") {
      const observation = result.observation || {};
      showProbe(observation.ready
        ? (observation.ptp_ready === false
          ? "客户端空闲 · GDK 服务就绪 · PTP 警告将运行时验证"
          : "客户端空闲 · PTP / GDK 基础检查通过")
        : `客户端空闲 · 观测被阻止：${observation.reason || "运行环境未就绪"}`);
    }
    else showProbe(client.reason || "客户端状态未知");
  } catch (error) {
    if (!isAbortLike(error)) showProbe(error.message);
    else showProbe("等待刷新");
  }
}
dom.probe_robot.addEventListener("click", probeRobot);
dom.console_probe_robot.addEventListener("click", probeRobot);
dom.start.addEventListener("click", startTask);
dom.pause.addEventListener("click", () => postAction("pause"));
dom.resume.addEventListener("click", () => postAction("resume"));
dom.stop.addEventListener("click", () => postAction("stop"));
dom.reset_robot.addEventListener("click", resetRobot);
dom.close_model.addEventListener("click", () => {
  if (!state.closeModelArmed) {
    state.closeModelArmed = true;
    dom.close_model.textContent = "再次点击确认关闭模型";
    toast("再次点击关闭按钮以确认释放 GPU");
    window.setTimeout(() => {
      state.closeModelArmed = false;
      if (state.job) renderControls(state.job);
    }, 5000);
    return;
  }
  state.closeModelArmed = false;
  postAction("close-model");
});

function syncPresentationVisibility() {
  const visible = !state.destroyed && !document.hidden;
  if (state.twinViewerOrigin) dom.robot_twin.contentWindow?.postMessage(
    { type: "nexus-viewer-visibility", visible: visible && state.twinVisible }, state.twinViewerOrigin,
  );
  if (!visible || !state.camerasVisible) cancelInferenceImages();
  else scheduleInferenceFrameSwap(state.latestCameras);
  if (visible && state.twinVisible && state.twinTelemetry) renderTwin(state.twinTelemetry);
}

const presentationObserver = typeof IntersectionObserver === "function" ? new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (entry.target === dom.twin_viewport) state.twinVisible = entry.isIntersecting;
    if (entry.target === dom.camera_grid) state.camerasVisible = entry.isIntersecting;
  }
  syncPresentationVisibility();
}) : null;
presentationObserver?.observe(dom.twin_viewport);
presentationObserver?.observe(dom.camera_grid);
document.addEventListener("visibilitychange", syncPresentationVisibility);

startModelElapsedTimer();
window.addEventListener("pagehide", () => {
  state.destroyed = true;
  window.clearInterval(state.modelElapsedTimer);
  syncPresentationVisibility();
  presentationObserver?.disconnect();
  if (state.pollTimer) window.clearTimeout(state.pollTimer);
  if (state.twinTimer) window.clearTimeout(state.twinTimer);
});
window.addEventListener("pageshow", (event) => {
  if (!event.persisted) return;
  state.destroyed = false;
  presentationObserver?.observe(dom.twin_viewport);
  presentationObserver?.observe(dom.camera_grid);
  syncPresentationVisibility();
  window.clearTimeout(state.pollTimer); window.clearTimeout(state.twinTimer);
  state.pollTimer = window.setTimeout(poll, 0);
  if (monitorView) state.twinTimer = window.setTimeout(pollTwin, 0);
  startModelElapsedTimer();
});

window.addEventListener("unhandledrejection", (event) => {
  if (isAbortLike(event.reason)) event.preventDefault();
});
window.addEventListener("error", (event) => {
  if (isAbortLike(event.error || event.message)) event.preventDefault();
});

function handleBootstrapFailure(error) {
  if (isAbortLike(error)) {
    if (!state.destroyed) window.setTimeout(() => bootstrap().catch(handleBootstrapFailure), 700);
    return;
  }
  dom.session_error.hidden = false;
  dom.session_error_message.textContent = error.message;
  setConnection("offline", "载入失败");
}

bootstrap().catch(handleBootstrapFailure);
