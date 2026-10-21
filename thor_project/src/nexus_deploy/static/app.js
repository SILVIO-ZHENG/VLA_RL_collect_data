import { configureAlignmentLink } from "./alignment-link.js";
import { freshCsrfHeaders, sessionErrorMessage } from "./ui-session.js?v=20261008-csrf-refresh";
import { inferenceModeLabel, modelVersionForConfig, rtcInferenceMode, trainedRtcSupported, inferenceSelectionValue, parseInferenceSelection } from "./rtc-labels.js?v=20261016-rtc-selector";
import { RTC_MODEL_MODES } from "./rtc-model-metadata.js?v=20261016-rtc-selector";

const API_ROOT = "/api";

const FALLBACK_DEFAULTS = Object.freeze({
  robot_host: "",
  robot_ssh_port: 22,
  effector_profile: "dahuan",
  model_version: "",
  duration_s: 30,
  execution_fps: 30,
  servo_fps: 100,
  prediction_horizon: 50,
  quality_horizon: 50,
  request_lead_margin_steps: 8,
  min_replan_after_steps: 1,
  latency_window_size: 20,
  request_timeout_s: 30,
  min_rtc_blend_steps: 8,
  warmup_requests: 1,
  execution_preflight_requests: 5,
  watchdog_timeout_s: 3,
  approval_timeout_s: 300,
  inference_mode: "rtc",
  hardware_mode: "execute",
  server_mode: "real",
  gpu_id: "auto",
  human_in_loop: false,
  presence_gate: {
    enabled: false,
    mode: "automatic",
    pause_after_s: 2,
    reset_after_s: 5,
    auto_resume: true,
  },
  action_low_pass_cutoff_hz: 4,
  effector_action_low_pass_cutoff_hz: 2,
  effector_control_mode: "device-position",
  arm_limit_profile: "urdf-velocity-uniform-acceleration",
  chunk_smoothing_mode: "savgol",
  action_interpolation: "linear",
  arm_tracking_mode: "damped",
  arm_tracking_omega: 16,
  rtc_prefix_mode: "model",
  max_arm_velocity: 3.76,
  max_arm_acceleration: 8,
  max_effector_velocity: 0.05,
  max_effector_acceleration: 0.2,
  max_head_waist_velocity: 0.3,
  max_head_waist_acceleration: 1,
  max_arm_tracking_error: 0.4,
  max_effector_tracking_error: 0.01,
  max_head_waist_tracking_error: 0.2,
  safety: {
    watchdog_enabled: true,
    kinematic_limits_enabled: true,
    tracking_error_enabled: true,
    low_pass_filter_enabled: false,
    effector_low_pass_filter_enabled: false,
    hold_effectors: false,
    workspace_safety_enabled: false,
    safety_profile_id: "",
  },
});

const PARAMETER_LABELS = {
  robot_host: "机器人 IP",
  robot_ssh_port: "SSH 端口",
  effector_profile: "夹爪类型",
  model_version: "模型版本",
  duration_s: "运行时长",
  execution_fps: "动作执行频率",
  servo_fps: "伺服频率",
  prediction_horizon: "预测步长",
  quality_horizon: "可信执行步长",
  request_lead_margin_steps: "请求提前余量",
  min_replan_after_steps: "最小重规划间隔",
  latency_window_size: "延迟统计窗口",
  request_timeout_s: "单次请求超时",
  min_rtc_blend_steps: "RTC 最小融合步数",
  warmup_requests: "模型预热请求",
  execution_preflight_requests: "执行预检请求",
  watchdog_timeout_s: "看门狗超时",
  approval_timeout_s: "审批超时",
  action_low_pass_cutoff_hz: "手臂低通截止（可选）",
  effector_action_low_pass_cutoff_hz: "末端低通截止（可选）",
  effector_control_mode: "夹爪目标处理",
  arm_limit_profile: "手臂动态限制配置",
  chunk_smoothing_mode: "轨迹平滑方案",
  action_interpolation: "100 Hz 插值方式",
  arm_tracking_mode: "手臂跟踪方式",
  arm_tracking_omega: "手臂阻尼参数",
  rtc_prefix_mode: "RTC 前缀测试",
  max_arm_velocity: "手臂最大速度",
  max_arm_acceleration: "手臂最大加速度",
  max_effector_velocity: "末端最大速度",
  max_effector_acceleration: "末端最大加速度",
  max_head_waist_velocity: "头腰最大速度",
  max_head_waist_acceleration: "头腰最大加速度",
  max_arm_tracking_error: "手臂误差阈值",
  max_effector_tracking_error: "末端误差阈值",
  max_head_waist_tracking_error: "头腰误差阈值",
};

const FIELD_SELECTORS = {
  robot_id: "#robot-options",
  effector_profile: "#effector-profile-options",
  robot_host: "#robot-host",
  robot_ssh_port: "#robot-ssh-port",
  model_id: "#model-options",
  model_version: "#model-version",
  task: "#task-input",
  inference_mode: "#inference-mode-options",
  rtc_inference_mode: "#inference-mode-options",
  "safety.safety_profile_id": "#safety-profile",
};

const HARDWARE_LABELS = { "read-only": "只读观测", execute: "真机执行" };
const SERVER_LABELS = { reuse: "使用已运行服务", mock: "Mock 测试服务", real: "启动真实模型服务" };

const JOB_STATUS = {
  queued: { label: "部署准备中", stage: 0, tone: "is-active", progress: 5 },
  pending: { label: "部署准备中", stage: 0, tone: "is-active", progress: 5 },
  validating: { label: "正在校验", stage: 0, tone: "is-active", progress: 12 },
  starting: { label: "准备环境", stage: 1, tone: "is-active", progress: 25 },
  starting_server: { label: "启动模型", stage: 1, tone: "is-active", progress: 30 },
  server_ready: { label: "模型已就绪", stage: 2, tone: "is-active", progress: 45 },
  deploying: { label: "发布客户端", stage: 2, tone: "is-active", progress: 48 },
  deploying_client: { label: "发布客户端", stage: 2, tone: "is-active", progress: 50 },
  connecting: { label: "连接机器人", stage: 3, tone: "is-active", progress: 60 },
  warmup: { label: "模型预热", stage: 3, tone: "is-active", progress: 65 },
  preflight: { label: "生成计划", stage: 3, tone: "is-active", progress: 72 },
  model_ready: { label: "模型已就绪", stage: 4, tone: "is-active", progress: 78 },
  waiting_approval: { label: "迁移旧会话", stage: 4, tone: "is-waiting", progress: 80 },
  approved: { label: "已批准", stage: 5, tone: "is-active", progress: 84 },
  ready_to_start: { label: "等待开始任务", stage: 5, tone: "is-waiting", progress: 86 },
  task_starting: { label: "正在启动任务", stage: 5, tone: "is-active", progress: 92 },
  running: { label: "任务运行中", stage: 5, tone: "is-active", progress: 90 },
  paused: { label: "已暂停", stage: 5, tone: "is-waiting", progress: 96 },
  completed: { label: "部署完成", stage: 6, tone: "is-success", progress: 100, terminal: true },
  succeeded: { label: "部署完成", stage: 6, tone: "is-success", progress: 100, terminal: true },
  success: { label: "部署完成", stage: 6, tone: "is-success", progress: 100, terminal: true },
  failed: { label: "部署失败", stage: 6, tone: "is-failed", progress: 100, terminal: true },
  rejected: { label: "任务已拒绝", stage: 6, tone: "is-failed", progress: 100, terminal: true },
  cancelled: { label: "任务已取消", stage: 6, tone: "is-failed", progress: 100, terminal: true },
};

const MODEL_STAGES = ["检查服务", "准备环境", "加载配置", "加载权重", "搬入 GPU", "模型预热", "健康检查"];
const MODEL_PHASE_INDEX = {
  checking: 0,
  environment: 1,
  config: 2,
  weights: 3,
  gpu: 4,
  warmup: 5,
  health: 6,
  healthy: 6,
  simulated: 6,
};

const state = {
  bootstrap: null,
  defaults: structuredClone(FALLBACK_DEFAULTS),
  csrfToken: "",
  bootstrapping: false,
  loaded: false,
  validating: false,
  deploying: false,
  dirty: true,
  validation: null,
  validationSequence: 0,
  validationTimer: null,
  activeJob: null,
  pollTimer: null,
  bootstrapRetryTimer: null,
  pollFailures: 0,
  pollErrorShown: false,
  destroyed: false,
  endpointRobotId: "",
  robotProbe: null,
  cleanupPromptedSessionId: "",
  cleanupTarget: null,
  cleanupChallenge: null,
  cleanupBusy: false,
};

const dom = {
  form: document.querySelector("#deployment-form"),
  projectProtocol: document.querySelector("#project-protocol"),
  connectionState: document.querySelector("#connection-state"),
  loadError: document.querySelector("#load-error"),
  loadErrorMessage: document.querySelector("#load-error-message"),
  simulationBanner: document.querySelector("#simulation-banner"),
  retryBootstrap: document.querySelector("#retry-bootstrap"),
  effectorProfileOptions: document.querySelector("#effector-profile-options"),
  robotOptions: document.querySelector("#robot-options"),
  robotHost: document.querySelector("#robot-host"),
  robotSshPort: document.querySelector("#robot-ssh-port"),
  robotEndpointStatus: document.querySelector("#robot-endpoint-status"),
  robotEndpointNote: document.querySelector("#robot-endpoint-note"),
  checkRobotEndpoint: document.querySelector("#check-robot-endpoint"),
  occupiedSessionNotice: document.querySelector("#occupied-session-notice"),
  occupiedSessionSummary: document.querySelector("#occupied-session-summary"),
  openSessionCleanup: document.querySelector("#open-session-cleanup"),
  cleanupDialog: document.querySelector("#session-cleanup-dialog"),
  cleanupClose: document.querySelector("#session-cleanup-close"),
  cleanupCancelOne: document.querySelector("#cleanup-cancel-one"),
  cleanupReview: document.querySelector("#cleanup-review"),
  cleanupBack: document.querySelector("#cleanup-back"),
  cleanupConfirm: document.querySelector("#cleanup-confirm"),
  cleanupStepOne: document.querySelector("#cleanup-step-one"),
  cleanupStepTwo: document.querySelector("#cleanup-step-two"),
  cleanupRobot: document.querySelector("#cleanup-robot"),
  cleanupEndpoint: document.querySelector("#cleanup-endpoint"),
  cleanupMode: document.querySelector("#cleanup-mode"),
  cleanupPid: document.querySelector("#cleanup-pid"),
  cleanupSession: document.querySelector("#cleanup-session"),
  cleanupPhrase: document.querySelector("#cleanup-phrase"),
  cleanupConfirmationInput: document.querySelector("#cleanup-confirmation-input"),
  modelOptions: document.querySelector("#model-options"),
  modelVersion: document.querySelector("#model-version"),
  modelVersionDescription: document.querySelector("#model-version-description"),
  gpuField: document.querySelector("#gpu-field"),
  gpuSelect: document.querySelector("#gpu-select"),
  gpuGrid: document.querySelector("#gpu-grid"),
  gpuControls: document.querySelector("#gpu-controls"),
  gpuSelectionHelp: document.querySelector("#gpu-selection-help"),
  refreshGpus: document.querySelector("#refresh-gpus"),
  thorFixedGpu: document.querySelector("#thor-fixed-gpu"),
  computeResourceNote: document.querySelector("#compute-resource-note"),
  resourceModeSummary: document.querySelector("#resource-mode-summary"),
  resourceModeSummaryTitle: document.querySelector("#resource-mode-summary-title"),
  resourceModeSummaryNote: document.querySelector("#resource-mode-summary-note"),
  taskInput: document.querySelector("#task-input"),
  taskCount: document.querySelector("#task-count"),
  taskLimit: document.querySelector("#task-limit"),
  durationInput: document.querySelector("#duration-s"),
  durationUnit: document.querySelector("#duration-unit"),
  modeSupportHint: document.querySelector("#mode-support-hint"),
  hitl: document.querySelector("#human-in-loop"),
  hitlCard: document.querySelector("#hitl-card"),
  workspaceSafety: document.querySelector("#workspace-safety-enabled"),
  workspaceSafetyHelp: document.querySelector("#workspace-safety-help"),
  safetyProfile: document.querySelector("#safety-profile"),
  profileSelectWrap: document.querySelector("#profile-select-wrap"),
  resetParameters: document.querySelector("#reset-parameters"),
  advancedParameters: document.querySelector("#advanced-parameters"),
  parameterErrors: document.querySelector("#parameter-errors"),
  validateButton: document.querySelector("#validate-button"),
  submitButtons: [...document.querySelectorAll("[data-submit-deployment]")],
  actionScopeNote: document.querySelector("#action-scope-note"),
  summaryStatus: document.querySelector("#summary-status"),
  summaryRobot: document.querySelector("#summary-robot"),
  summaryEffector: document.querySelector("#summary-effector"),
  summaryEndpoint: document.querySelector("#summary-endpoint"),
  summaryModel: document.querySelector("#summary-model"),
  summaryVersion: document.querySelector("#summary-version"),
  summaryNormalization: document.querySelector("#summary-normalization"),
  summaryTask: document.querySelector("#summary-task"),
  summaryInference: document.querySelector("#summary-inference"),
  summaryHardware: document.querySelector("#summary-hardware"),
  summaryServer: document.querySelector("#summary-server"),
  summaryGpu: document.querySelector("#summary-gpu"),
  summaryHitl: document.querySelector("#summary-hitl"),
  summaryDuration: document.querySelector("#summary-duration"),
  summaryFps: document.querySelector("#summary-fps"),
  summaryHorizon: document.querySelector("#summary-horizon"),
  summarySafety: document.querySelector("#summary-safety"),
  changedCount: document.querySelector("#changed-count"),
  changesList: document.querySelector("#changes-list"),
  validationCard: document.querySelector("#validation-card"),
  validationIcon: document.querySelector("#validation-icon"),
  validationTitle: document.querySelector("#validation-title"),
  validationSubtitle: document.querySelector("#validation-subtitle"),
  validationMessages: document.querySelector("#validation-messages"),
  serverSummary: document.querySelector("#server-summary"),
  jobCard: document.querySelector("#job-card"),
  jobStatus: document.querySelector("#job-status"),
  jobId: document.querySelector("#job-id"),
  jobProgress: document.querySelector("#job-progress"),
  jobStages: document.querySelector("#job-stages"),
  jobMessage: document.querySelector("#job-message"),
  modelServiceStatus: document.querySelector("#model-service-status"),
  modelServiceStrategy: document.querySelector("#model-service-strategy"),
  modelServicePhase: document.querySelector("#model-service-phase"),
  modelServiceElapsed: document.querySelector("#model-service-elapsed"),
  modelServiceTimeout: document.querySelector("#model-service-timeout"),
  modelServiceTarget: document.querySelector("#model-service-target"),
  modelLastLog: document.querySelector("#model-last-log"),
  taskSessionStatus: document.querySelector("#task-session-status"),
  taskSessionNote: document.querySelector("#task-session-note"),
  jobConsoleLink: document.querySelector("#job-console-link"),
  currentModelCard: document.querySelector("#current-model-card"),
  currentModelSummary: document.querySelector("#current-model-summary"),
  currentModelConfig: document.querySelector("#current-model-config"),
  currentModelClose: document.querySelector("#current-model-close"),
  activeModePill: document.querySelector("#active-mode-pill"),
  modeParameterTitle: document.querySelector("#mode-parameter-title"),
  modeParameterDescription: document.querySelector("#mode-parameter-description"),
  jobEventsWrap: document.querySelector("#job-events-wrap"),
  jobEvents: document.querySelector("#job-events"),
  toastRegion: document.querySelector("#toast-region"),
};

function cloneDefaults(defaults = {}) {
  return {
    ...structuredClone(FALLBACK_DEFAULTS),
    ...defaults,
    safety: {
      ...structuredClone(FALLBACK_DEFAULTS.safety),
      ...(defaults.safety || {}),
    },
    presence_gate: {
      ...structuredClone(FALLBACK_DEFAULTS.presence_gate),
      ...(defaults.presence_gate || {}),
    },
  };
}

async function apiFetch(path, options = {}) {
  const headers = await freshCsrfHeaders(options);
  headers.set("Accept", "application/json");
  if (options.body !== undefined && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(`${API_ROOT}${path}`, {
    credentials: "same-origin",
    cache: "no-store",
    ...options,
    headers,
  });

  let payload = null;
  const contentType = response.headers.get("content-type") || "";
  try {
    payload = contentType.includes("json") ? await response.json() : await response.text();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    const message = sessionErrorMessage(response, payload) || payload?.detail || payload?.message || payload?.error || (typeof payload === "string" && payload) || `请求失败（HTTP ${response.status}）`;
    const error = new Error(message);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

function setConnection(tone, label) {
  dom.connectionState.className = `connection-state ${tone ? `is-${tone}` : ""}`.trim();
  dom.connectionState.lastElementChild.textContent = label;
}

function setButtonLoading(button, loading) {
  if (!button) return;
  button.classList.toggle("is-loading", loading);
  button.setAttribute("aria-busy", String(loading));
}

function toast(message, tone = "info", duration = 4200) {
  if (isAbortLike(message)) return;
  const item = document.createElement("div");
  item.className = `toast ${tone !== "info" ? `is-${tone}` : ""}`.trim();
  item.setAttribute("role", tone === "error" ? "alert" : "status");
  item.textContent = message;
  dom.toastRegion.append(item);
  window.setTimeout(() => item.remove(), duration);
}

function isAbortLike(error) {
  const message = String(error?.message || error || "");
  return error?.name === "AbortError" || /abort(?:ed|error)?/i.test(message);
}

function radioValue(name) {
  return dom.form.querySelector(`input[name="${name}"]:checked`)?.value || "";
}

function setRadioValue(name, value) {
  const input = dom.form.querySelector(`input[name="${name}"][value="${CSS.escape(String(value))}"]`);
  if (input && !input.disabled) input.checked = true;
}

function numberValue(key) {
  const input = dom.form.querySelector(`[data-config-key="${key}"]`);
  if (!input) return Number.NaN;
  const raw = input.value.trim();
  return raw === "" ? Number.NaN : Number(raw);
}

function selectedRobot() {
  const id = radioValue("robot_id");
  return state.bootstrap?.robots?.find((item) => String(item.id) === id) || null;
}

function selectedModel() {
  const id = radioValue("model_id");
  return state.bootstrap?.models?.find((item) => String(item.id) === id) || null;
}

function selectedEffectorProfile() {
  const id = radioValue("effector_profile");
  return state.bootstrap?.effector_profiles?.find((item) => String(item.id) === id) || null;
}

function selectedModelVariant(model = selectedModel()) {
  const profile = radioValue("effector_profile");
  return model?.effector_variants?.[profile] || null;
}

function selectedModelVersion() {
  return modelVersionForConfig(state.bootstrap, {
    model_id: radioValue("model_id"),
    effector_profile: radioValue("effector_profile"),
    model_version: dom.modelVersion?.value || "",
  }, RTC_MODEL_MODES);
}

function modeLabel(mode) {
  return inferenceModeLabel(mode, selectedModelVersion(), parseInferenceSelection(radioValue("inference_mode")).rtc_inference_mode);
}

function selectedCheckpointKey() {
  return JSON.stringify([radioValue("model_id"), radioValue("effector_profile"), dom.modelVersion?.value || ""]);
}

function modelHasHealthyService(model) {
  const selectedProfile = radioValue("effector_profile");
  const selectedVersion = dom.modelVersion?.value || "";
  return Boolean(
    model?.status?.running
    && model?.status?.healthy
    && model?.running_effector_profile === selectedProfile
    && model?.running_model_version === selectedVersion
    && (!(model?.supported_modes || []).includes("rtc")
      || model.running_rtc_inference_mode === (parseInferenceSelection(radioValue("inference_mode")).rtc_inference_mode
        || rtcInferenceMode({}, selectedModelVersion())))
  );
}

function recommendedServerMode(model = selectedModel()) {
  return modelHasHealthyService(model) ? "reuse" : "real";
}

function applySmartServerMode() {
  const model = selectedModel();
  if (model) setRadioValue("server_mode", recommendedServerMode(model));
}

function updateServerModeAvailability() {
  const reuseInput = dom.form.querySelector('input[name="server_mode"][value="reuse"]');
  const reuseOption = reuseInput?.closest("label");
  const reusable = modelHasHealthyService(selectedModel());
  if (!reuseInput || !reuseOption) return;
  reuseInput.disabled = !reusable;
  reuseOption.classList.toggle("is-disabled", !reusable);
  reuseOption.title = reusable ? "复用当前平台受管的健康服务" : "所选模型当前没有可复用的健康实例";
  if (!reusable && reuseInput.checked) setRadioValue("server_mode", "real");
}

function taskMaxLength() {
  const configured = Number(state.bootstrap?.constraints?.task_max_length);
  return Number.isInteger(configured) && configured > 0 ? configured : 4096;
}

function updateActionScope() {
  const actionsEnabled = state.bootstrap?.constraints?.actions_enabled === true;
  dom.simulationBanner.hidden = actionsEnabled;
  updatePrimaryActionLabel();
  const note = dom.actionScopeNote?.querySelector("span");
  if (note) {
    note.textContent = actionsEnabled
      ? "这里只准备模型服务；任务启动与控制在机器人控制台。"
      : "安全预览不会实际启动组件，也不会连接机器人。";
  }
}

function updatePrimaryActionLabel() {
  const actionsEnabled = state.bootstrap?.constraints?.actions_enabled === true;
  const model = selectedModel();
  const serverMode = radioValue("server_mode") || recommendedServerMode(model);
  let text = "准备模型服务";
  if (!actionsEnabled) {
    text = "预览模型部署";
  } else if (!model) {
    text = "请先选择模型";
  } else if (serverMode === "reuse") {
    text = `连接 ${model.label || model.id}`;
  } else if (serverMode === "real") {
    text = `加载 ${model.label || model.id}`;
  } else if (serverMode === "mock") {
    text = "启动 Mock 测试";
  }
  dom.submitButtons.forEach((button) => {
    const label = button.querySelector(".button-label");
    if (label) label.textContent = text;
  });
}

function getConfiguration() {
  // Workspace collision handling is configured after the model is ready. The
  // preparation page submits a neutral read-only baseline so hidden runtime
  // controls can never make model preparation fail.
  const safety = {
    watchdog_enabled: true,
    kinematic_limits_enabled: true,
    tracking_error_enabled: true,
    low_pass_filter_enabled: false,
    effector_low_pass_filter_enabled: false,
    hold_effectors: false,
    workspace_safety_enabled: false,
    safety_profile_id: null,
  };

  const configuration = {
    robot_id: radioValue("robot_id") || null,
    effector_profile: radioValue("effector_profile") || null,
    robot_host: dom.robotHost.value.trim(),
    robot_ssh_port: Number(dom.robotSshPort.value),
    model_id: radioValue("model_id") || null,
    model_version: dom.modelVersion.value || null,
    gpu_id: dom.gpuSelect.value || "auto",
    task: dom.taskInput.value.trim(),
    inference_mode: parseInferenceSelection(radioValue("inference_mode")).inference_mode,
    rtc_inference_mode: parseInferenceSelection(radioValue("inference_mode")).rtc_inference_mode
      || rtcInferenceMode({}, selectedModelVersion()),
    // Model loading is read-only staging. The runtime page defaults to
    // execute but requires an explicit, validated save before any start.
    hardware_mode: "read-only",
    server_mode: radioValue("server_mode") || "reuse",
    // Completion rules are selected after the model is ready.
    presence_gate: { enabled: false },
    duration_s: numberValue("duration_s"),
    execution_fps: numberValue("execution_fps"),
    servo_fps: numberValue("servo_fps"),
    prediction_horizon: numberValue("prediction_horizon"),
    quality_horizon: numberValue("quality_horizon"),
    request_lead_margin_steps: numberValue("request_lead_margin_steps"),
    min_replan_after_steps: numberValue("min_replan_after_steps"),
    latency_window_size: numberValue("latency_window_size"),
    request_timeout_s: numberValue("request_timeout_s"),
    min_rtc_blend_steps: numberValue("min_rtc_blend_steps"),
    warmup_requests: numberValue("warmup_requests"),
    execution_preflight_requests: numberValue("execution_preflight_requests"),
    watchdog_timeout_s: numberValue("watchdog_timeout_s"),
    human_in_loop: dom.hitl.checked,
    action_low_pass_cutoff_hz: numberValue("action_low_pass_cutoff_hz"),
    effector_action_low_pass_cutoff_hz: numberValue("effector_action_low_pass_cutoff_hz"),
    effector_control_mode: document.querySelector('[data-config-key="effector_control_mode"]').value,
    arm_limit_profile: document.querySelector('[data-config-key="arm_limit_profile"]').value,
    chunk_smoothing_mode: document.querySelector('[data-config-key="chunk_smoothing_mode"]').value,
    action_interpolation: document.querySelector('[data-config-key="action_interpolation"]').value,
    arm_tracking_mode: document.querySelector('[data-config-key="arm_tracking_mode"]').value,
    arm_tracking_omega: numberValue("arm_tracking_omega"),
    rtc_prefix_mode: document.querySelector('[data-config-key="rtc_prefix_mode"]').value,
    max_arm_velocity: numberValue("max_arm_velocity"),
    max_arm_acceleration: numberValue("max_arm_acceleration"),
    max_effector_velocity: numberValue("max_effector_velocity"),
    max_effector_acceleration: numberValue("max_effector_acceleration"),
    max_head_waist_velocity: numberValue("max_head_waist_velocity"),
    max_head_waist_acceleration: numberValue("max_head_waist_acceleration"),
    max_arm_tracking_error: numberValue("max_arm_tracking_error"),
    max_effector_tracking_error: numberValue("max_effector_tracking_error"),
    max_head_waist_tracking_error: numberValue("max_head_waist_tracking_error"),
    safety,
  };
  return configuration;
}

function applyDefaults({ preserveTargets = false } = {}) {
  const defaults = state.defaults;
  document.querySelectorAll("[data-config-key]").forEach((input) => {
    const key = input.dataset.configKey;
    const value = defaults[key];
    if (value !== undefined && value !== null) input.value = String(value);
  });
  dom.taskInput.value = defaults.task || "";
  dom.hitl.checked = Boolean(defaults.human_in_loop);
  setRadioValue("effector_profile", defaults.effector_profile || "dahuan");
  setRadioValue("hardware_mode", "read-only");
  setRadioValue("server_mode", defaults.server_mode || "reuse");
  dom.gpuSelect.value = String(defaults.gpu_id || "auto");
  document.querySelectorAll("[data-safety-key]").forEach((input) => {
    const key = input.dataset.safetyKey;
    input.checked = ["watchdog_enabled", "kinematic_limits_enabled", "tracking_error_enabled"].includes(key)
      ? true
      : Boolean(defaults.safety?.[key]);
  });

  if (!preserveTargets) {
    const defaultRobot = defaults.robot_id || state.bootstrap?.robots?.[0]?.id;
    const defaultModel = defaults.model_id || state.bootstrap?.models?.[0]?.id;
    if (defaultRobot !== undefined) setRadioValue("robot_id", defaultRobot);
    if (defaultModel !== undefined) setRadioValue("model_id", defaultModel);
    applySelectedRobotConnection({ force: true });
  }
  renderModelVersions({ preferred: defaults.model_version || "" });
  state.rtcSelectionKey = selectedCheckpointKey();
  setRadioValue("inference_mode", inferenceSelectionValue(defaults, selectedModelVersion()));
  applySmartServerMode();
  updateDurationDisplay();
  updateDependentControls();
  updateSummary();
  markDirty(false);
}

function createSvg(kind) {
  if (kind === "robot") {
    return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 6V4m8 2V4M6 9h12v9H6zM4 12H2m20 0h-2M9 13h.01M15 13h.01M9 17h6"/></svg>';
  }
  return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 3h8v4H8zM5 9h14v11H5zM9 13h2m2 0h2m-6 4h6"/></svg>';
}

function makeTag(text, className = "mini-tag") {
  const tag = document.createElement("span");
  tag.className = className;
  tag.textContent = text;
  return tag;
}

function renderRobots(robots) {
  dom.robotOptions.replaceChildren();
  if (!robots.length) {
    const empty = document.createElement("div");
    empty.className = "empty-options";
    empty.textContent = "当前项目没有可用机器人配置。";
    dom.robotOptions.append(empty);
    return;
  }

  robots.forEach((robot) => {
    const card = document.createElement("label");
    card.className = "selection-card";
    card.dataset.robotId = String(robot.id);
    const isOffline = robot.online === false;
    if (isOffline) card.classList.add("is-disabled");
    card.title = isOffline ? "机器人当前离线，选择后将无法通过部署校验" : robot.online == null ? "在线状态待运行时探测" : "";

    const input = document.createElement("input");
    input.type = "radio";
    input.name = "robot_id";
    input.value = String(robot.id);

    const icon = document.createElement("span");
    icon.className = "selection-card__icon";
    icon.innerHTML = createSvg("robot");

    const body = document.createElement("span");
    body.className = "selection-card__body";
    const title = document.createElement("span");
    title.className = "selection-card__title";
    const strong = document.createElement("strong");
    strong.textContent = robot.label || robot.id;
    const onlineLabel = robot.online === true ? "在线" : isOffline ? "离线" : "未检测";
    const onlineClass = robot.online === true ? "is-good" : isOffline ? "is-offline" : "is-warn";
    const health = makeTag(onlineLabel, `health-tag robot-health ${onlineClass}`);
    title.append(strong, health);
    const description = document.createElement("p");
    description.textContent = robot.description || "未提供机器人描述";
    const meta = document.createElement("span");
    meta.className = "selection-card__meta";
    meta.append(makeTag(robot.configured ? "已配置" : "未配置"));
    const availableGrippers = robot.effectors?.available_profiles?.length || 0;
    if (availableGrippers) meta.append(makeTag(`支持 ${availableGrippers} 种夹爪配置`));
    (robot.tags || []).slice(0, 3).forEach((tag) => meta.append(makeTag(String(tag))));
    const calibratedCount = (robot.safety_profiles || []).filter((profile) => profile.calibrated).length;
    meta.append(makeTag(`${calibratedCount} 个已标定安全配置`));
    body.append(title, description, meta);

    const radio = document.createElement("span");
    radio.className = "selection-card__radio";
    radio.setAttribute("aria-hidden", "true");
    card.append(input, icon, body, radio);
    dom.robotOptions.append(card);
  });
}

function renderEffectorProfiles(profiles) {
  dom.effectorProfileOptions.replaceChildren();
  if (!profiles.length) {
    const empty = document.createElement("div");
    empty.className = "empty-options";
    empty.textContent = "当前项目没有可用夹爪配置。";
    dom.effectorProfileOptions.append(empty);
    return;
  }
  profiles.forEach((profile) => {
    const card = document.createElement("label");
    card.className = "selection-card selection-card--compact";
    card.dataset.effectorProfile = String(profile.id);
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "effector_profile";
    input.value = String(profile.id);
    const body = document.createElement("span");
    body.className = "selection-card__body";
    const title = document.createElement("span");
    title.className = "selection-card__title";
    const strong = document.createElement("strong");
    strong.textContent = profile.label || profile.id;
    title.append(strong);
    const description = document.createElement("p");
    description.textContent = `${profile.state_semantics}；${profile.action_semantics}`;
    const meta = document.createElement("span");
    meta.className = "selection-card__meta";
    meta.append(makeTag(`行程 ${profile.hardware_state_range?.join(" – ")} ${profile.unit}`));
    meta.append(makeTag(profile.action_mapping === "identity" ? "模型/GDK 同向" : "自动反向映射"));
    body.append(title, description, meta);
    const radio = document.createElement("span");
    radio.className = "selection-card__radio";
    radio.setAttribute("aria-hidden", "true");
    card.append(input, body, radio);
    dom.effectorProfileOptions.append(card);
  });
}

function applyEffectorProfileDefaults({ updateTask = false } = {}) {
  const profile = selectedEffectorProfile();
  if (!profile?.limits) return;
  if (updateTask && profile.default_task) dom.taskInput.value = String(profile.default_task);
  const values = {
    effector_control_mode: profile.id === "dahuan" ? "device-position" : "software-trajectory",
    max_effector_velocity: profile.limits.max_velocity,
    max_effector_acceleration: profile.limits.max_acceleration,
    max_effector_tracking_error: profile.limits.tracking_error,
  };
  Object.entries(values).forEach(([key, value]) => {
    const input = dom.form.querySelector(`[data-config-key="${key}"]`);
    if (input && value !== undefined && value !== null) input.value = String(value);
  });
}

function setEndpointStatus(stateName, label, note = null) {
  dom.robotEndpointStatus.className = `endpoint-status is-${stateName}`;
  dom.robotEndpointStatus.lastChild.textContent = label;
  if (note) dom.robotEndpointNote.textContent = note;
}

function updateRobotCardProbe(robotId, status) {
  const card = [...dom.robotOptions.querySelectorAll("[data-robot-id]")]
    .find((item) => item.dataset.robotId === String(robotId));
  const health = card?.querySelector(".robot-health");
  if (!health) return;
  const healthy = status === "已部署 · 空闲";
  const failed = status === "连接失败" || status === "客户端未部署";
  const tone = status == null ? "is-warn" : healthy ? "is-good" : failed ? "is-offline" : "is-warn";
  health.className = `health-tag robot-health ${tone}`;
  health.lastChild.nodeValue = status || "未检测";
}

function resetRobotProbe() {
  state.robotProbe = null;
  state.cleanupPromptedSessionId = "";
  dom.occupiedSessionNotice.hidden = true;
  const robot = selectedRobot();
  if (robot) updateRobotCardProbe(robot.id, null);
  const endpoints = Array.isArray(robot?.connection?.endpoints) ? robot.connection.endpoints : [];
  const route = endpoints.length
    ? endpoints.map((item, index) => `${index === 0 ? "首选" : "备用"}${item.label ? ` ${item.label}` : ""} ${item.host}`).join("；")
    : "未声明备用地址";
  setEndpointStatus("idle", "未检测", `${route}。依次检测网络、SSH 身份、客户端部署状态和客户端进程。`);
}

function occupiedSessionTarget(result) {
  if (result?.client?.status !== "running") return null;
  const process = (Array.isArray(result.client.processes) ? result.client.processes : [])
    .find((item) => item?.managed === true && /^[0-9a-f]{1,64}$/.test(String(item.session_id || "")) && Number(item.pid) > 1);
  if (!process) return null;
  return {
    robot_id: String(result.robot_id),
    robot_host: String(result.host),
    robot_ssh_port: Number(result.port),
    session_id: String(process.session_id),
    pid: Number(process.pid),
    hardware_mode: String(process.hardware_mode || "unknown"),
  };
}

function syncOccupiedSessionNotice(result) {
  const target = occupiedSessionTarget(result);
  dom.occupiedSessionNotice.hidden = !target;
  if (!target) return;
  dom.occupiedSessionSummary.textContent = `${cleanupModeLabel(target.hardware_mode)} · PID ${target.pid} · Session ${target.session_id.slice(0, 8)}`;
}

function cleanupModeLabel(mode) {
  if (mode === "execute") return "真机执行（关闭前先安全暂停）";
  if (mode === "read-only") return "只读观测";
  if (mode === "dry-run") return "Dry-run";
  return "未知模式";
}

function cleanupTargetRequest(target) {
  return {
    robot_id: target.robot_id,
    robot_host: target.robot_host,
    robot_ssh_port: target.robot_ssh_port,
    session_id: target.session_id,
    pid: target.pid,
  };
}

function setCleanupStage(stage) {
  const second = stage === 2;
  dom.cleanupStepOne.hidden = second;
  dom.cleanupStepTwo.hidden = !second;
  dom.cleanupStepOne.classList.toggle("is-active", !second);
  dom.cleanupStepTwo.classList.toggle("is-active", second);
  if (second) {
    dom.cleanupConfirmationInput.value = "";
    dom.cleanupConfirm.disabled = true;
    window.setTimeout(() => dom.cleanupConfirmationInput.focus(), 0);
  }
}

function closeCleanupDialog() {
  if (state.cleanupBusy) return;
  state.cleanupTarget = null;
  state.cleanupChallenge = null;
  setCleanupStage(1);
  if (dom.cleanupDialog.open) dom.cleanupDialog.close();
}

function openCleanupDialog(result, { force = false } = {}) {
  const target = occupiedSessionTarget(result);
  if (!target) return;
  if (!force && state.cleanupPromptedSessionId === target.session_id) return;
  state.cleanupPromptedSessionId = target.session_id;
  state.cleanupTarget = target;
  state.cleanupChallenge = null;
  const robot = (state.bootstrap?.robots || []).find((item) => String(item.id) === target.robot_id);
  dom.cleanupRobot.textContent = robot?.label || target.robot_id;
  dom.cleanupEndpoint.textContent = `${target.robot_host}:${target.robot_ssh_port}`;
  dom.cleanupMode.textContent = cleanupModeLabel(target.hardware_mode);
  dom.cleanupPid.textContent = String(target.pid);
  dom.cleanupSession.textContent = target.session_id;
  dom.cleanupPhrase.textContent = "—";
  setCleanupStage(1);
  if (!dom.cleanupDialog.open) dom.cleanupDialog.showModal();
}

async function prepareSessionCleanup() {
  if (!state.cleanupTarget || state.cleanupBusy) return;
  state.cleanupBusy = true;
  setButtonLoading(dom.cleanupReview, true);
  dom.cleanupReview.disabled = true;
  dom.cleanupClose.disabled = true;
  dom.cleanupCancelOne.disabled = true;
  try {
    const result = await apiFetch("/robots/cleanup/prepare", {
      method: "POST",
      headers: { "X-CSRF-Token": state.csrfToken },
      body: JSON.stringify(cleanupTargetRequest(state.cleanupTarget)),
    });
    state.cleanupChallenge = result;
    dom.cleanupPhrase.textContent = result.confirmation_phrase;
    setCleanupStage(2);
  } catch (error) {
    if (!isAbortLike(error)) toast(`无法进入二次确认：${error.message}`, "error");
  } finally {
    state.cleanupBusy = false;
    setButtonLoading(dom.cleanupReview, false);
    dom.cleanupReview.disabled = false;
    dom.cleanupClose.disabled = false;
    dom.cleanupCancelOne.disabled = false;
  }
}

async function confirmSessionCleanup() {
  if (!state.cleanupTarget || !state.cleanupChallenge || state.cleanupBusy) return;
  const phrase = dom.cleanupConfirmationInput.value;
  if (phrase !== state.cleanupChallenge.confirmation_phrase) return;
  state.cleanupBusy = true;
  setButtonLoading(dom.cleanupConfirm, true);
  dom.cleanupConfirm.disabled = true;
  dom.cleanupBack.disabled = true;
  dom.cleanupClose.disabled = true;
  try {
    const result = await apiFetch("/robots/cleanup", {
      method: "POST",
      headers: { "X-CSRF-Token": state.csrfToken },
      body: JSON.stringify({
        ...cleanupTargetRequest(state.cleanupTarget),
        challenge_token: state.cleanupChallenge.challenge_token,
        confirmation_phrase: phrase,
      }),
    });
    state.robotProbe = result.probe;
    const presentation = robotProbePresentation(result.probe);
    setEndpointStatus(presentation.tone, presentation.label, `最近检测：${formatTime(result.probe.checked_at)} · TCP ${result.probe.latency_ms} ms。${presentation.note}`);
    updateRobotCardProbe(result.probe.robot_id, presentation.card);
    syncOccupiedSessionNotice(result.probe);
    state.cleanupBusy = false;
    closeCleanupDialog();
    toast(`会话 ${result.session_id.slice(0, 8)} 已强制关闭；模型服务保持运行。`);
    updateSummary();
    scheduleValidation();
  } catch (error) {
    if (!isAbortLike(error)) {
      toast(`强制关闭失败：${error.message} 请重新完成第一次核验。`, "error");
      state.cleanupChallenge = null;
      setCleanupStage(1);
    }
  } finally {
    state.cleanupBusy = false;
    setButtonLoading(dom.cleanupConfirm, false);
    dom.cleanupBack.disabled = false;
    dom.cleanupClose.disabled = false;
    dom.cleanupConfirm.disabled = dom.cleanupConfirmationInput.value !== state.cleanupChallenge?.confirmation_phrase;
  }
}

function robotProbePresentation(result) {
  if (!result.reachable) {
    const alternatives = (selectedRobot()?.connection?.endpoints || []).filter((item) => item.host !== result.host);
    const fallback = alternatives.length ? ` 可改用已绑定的备用地址 ${alternatives.map((item) => item.host).join("、")}。` : "";
    return { tone: "unreachable", label: "网络不可达", card: "连接失败", note: `${result.host}:${result.port} 未建立 TCP 连接。请检查 IP、端口、网络和 SSH 服务。${fallback}` };
  }
  const client = result.client || {};
  if (!result.ssh_authenticated) {
    return { tone: "reachable", label: "仅网络可达", card: "仅网络可达", note: client.reason || "SSH 端口可达，但平台尚未完成身份认证，因此客户端状态未知。" };
  }
  if (client.status === "wrong_robot") {
    const identity = result.identity || {};
    return {
      tone: "unreachable",
      label: "机器人身份不匹配",
      card: "连接到错误机器人",
      note: client.reason || `期望 AID ${identity.expected_aid || "—"}，实际 ${identity.observed_aid || "无法读取"}。为防止误操作，部署、观测和真机执行均已阻止。`,
    };
  }
  if (client.status === "not_deployed") {
    return { tone: "unreachable", label: "客户端未部署", card: "客户端未部署", note: "SSH 认证成功，但目标目录没有完整、可执行的 Nexus 机器人客户端。请先发布客户端。" };
  }
  if (client.status === "invalid_client") {
    const missing = Array.isArray(client.missing_files) && client.missing_files.length
      ? `缺少 ${client.missing_files.slice(0, 3).join("、")}`
      : "manifest、协议版本或机器人类型不匹配";
    return { tone: "unreachable", label: "客户端无效", card: "客户端无效", note: `SSH 认证成功，但部署内容未通过完整性检查：${missing}。` };
  }
  if (client.status === "running") {
    const process = Array.isArray(client.processes) ? client.processes[0] : null;
    const session = process?.session_id ? `，会话 ${process.session_id.slice(0, 8)}` : "，来源会话未知";
    return { tone: "reachable", label: "客户端运行中", card: "客户端运行中", note: `网络和 SSH 正常；检测到机器人客户端 PID ${process?.pid ?? "—"}${session}。新任务启动前需确认是否为本平台作业。` };
  }
  if (client.status === "idle") {
    const protocol = client.protocol?.name && client.protocol?.version ? ` · ${client.protocol.name} v${client.protocol.version}` : "";
    const observation = result.observation || {};
    const runtime = observation.ready
      ? (observation.ptp_ready === false
        ? "GDK 基础服务已就绪；PTP 权限警告不会阻止启动，关节与相机将在运行时实测。"
        : "PTP 与 GDK 基础服务已就绪；关节与相机将在运行时实测。")
      : `模型可以先加载，但开始观测前仍被阻止：${observation.reason || "运行环境未就绪"}。`;
    return { tone: "reachable", label: "已部署 · 空闲", card: "已部署 · 空闲", note: `网络、SSH 和客户端检查通过${protocol}。${runtime}` };
  }
  return { tone: "reachable", label: "状态未知", card: "状态未知", note: client.reason || "网络可达，但未能确认客户端部署或进程状态。" };
}

function applySelectedRobotConnection({ force = false } = {}) {
  const robot = selectedRobot();
  const robotId = String(robot?.id || "");
  if (!force && state.endpointRobotId === robotId) return;
  state.endpointRobotId = robotId;
  const endpointOptions = document.querySelector("#robot-host-options");
  const endpoints = Array.isArray(robot?.connection?.endpoints) ? robot.connection.endpoints : [];
  endpointOptions?.replaceChildren(...endpoints.map((item, index) => {
    const option = document.createElement("option");
    option.value = item.host;
    option.label = `${index === 0 ? "首选" : "备用"} · ${item.label || item.transport || "网络地址"}`;
    return option;
  }));
  dom.robotHost.value = String(robot?.connection?.host || "");
  dom.robotSshPort.value = String(robot?.connection?.ssh_port || 22);
  resetRobotProbe();
}

async function probeRobotEndpoint({ automatic = false, silent = false } = {}) {
  const robot = selectedRobot();
  if (!robot) {
    if (!silent) toast("请先选择机器人。", "warning");
    return;
  }
  const host = dom.robotHost.value.trim();
  const port = Number(dom.robotSshPort.value);
  if (!host || !Number.isInteger(port) || port < 1 || port > 65535) {
    if (!silent) toast("请先填写有效的机器人 IP 和 SSH 端口。", "warning");
    return;
  }
  const configured = Array.isArray(robot.connection?.endpoints)
    ? [...robot.connection.endpoints].sort((left, right) => Number(left.priority || 0) - Number(right.priority || 0))
    : [];
  const currentIsPreferred = configured.length > 0
    && configured[0].host === host
    && Number(configured[0].ssh_port) === port;
  const candidates = automatic || currentIsPreferred
    ? configured
    : [{ host, ssh_port: port, label: "手动地址", transport: "manual" }];
  if (!candidates.length) candidates.push({ host, ssh_port: port, label: "当前地址", transport: "manual" });
  setButtonLoading(dom.checkRobotEndpoint, true);
  dom.checkRobotEndpoint.disabled = true;
  setEndpointStatus("checking", "检测中");
  try {
    let result = null;
    let selectedIndex = 0;
    for (const [index, candidate] of candidates.entries()) {
      setEndpointStatus("checking", "检测中", `正在检测 ${candidate.label || candidate.transport || "网络地址"} ${candidate.host}:${candidate.ssh_port}…`);
      const candidateResult = await apiFetch("/robots/probe", {
        method: "POST",
        headers: { "X-CSRF-Token": state.csrfToken },
        body: JSON.stringify({
          robot_id: robot.id,
          robot_host: candidate.host,
          robot_ssh_port: Number(candidate.ssh_port),
        }),
      });
      result = candidateResult;
      selectedIndex = index;
      const authenticatedRobot = candidateResult.ssh_authenticated === true
        && candidateResult.client?.status !== "wrong_robot"
        && candidateResult.identity?.match !== false;
      if (authenticatedRobot || index === candidates.length - 1) break;
    }
    if (!result) throw new Error("没有可检测的机器人地址");
    dom.robotHost.value = String(result.host);
    dom.robotSshPort.value = String(result.port);
    state.robotProbe = result;
    syncOccupiedSessionNotice(result);
    // Surface an occupied managed session immediately after instance probing.
    // The cleanup dialog is guarded per session so polling/repeated probes do
    // not create a popup loop.
    if (occupiedSessionTarget(result)) openCleanupDialog(result);
    const presentation = robotProbePresentation(result);
    const fallback = selectedIndex > 0
      ? `首选 ${candidates[0].host} 未通过身份认证或 AID 校验，已自动回退到 ${result.host}。`
      : "";
    setEndpointStatus(presentation.tone, presentation.label, `${fallback}最近检测：${formatTime(result.checked_at)} · TCP ${result.latency_ms} ms。${presentation.note}`);
    updateRobotCardProbe(robot.id, presentation.card);
    updateSummary();
    scheduleValidation();
  } catch (error) {
    if (isAbortLike(error)) {
      setEndpointStatus("idle", "等待检测");
      return;
    }
    setEndpointStatus("unreachable", "检测失败");
    if (!silent) toast(`机器人连接检测失败：${error.message}`, "error");
  } finally {
    setButtonLoading(dom.checkRobotEndpoint, false);
    dom.checkRobotEndpoint.disabled = false;
  }
}

function renderModels(models) {
  dom.modelOptions.replaceChildren();
  if (!models.length) {
    const empty = document.createElement("div");
    empty.className = "empty-options";
    empty.textContent = "当前项目没有可用模型配置。";
    dom.modelOptions.append(empty);
    return;
  }

  models.forEach((model) => {
    const variant = selectedModelVariant(model);
    const selectable = model.selectable !== false;
    const healthy = Boolean(model.status?.running && model.status?.healthy);
    const running = Boolean(model.status?.running);
    const card = document.createElement("label");
    card.className = "selection-card";
    card.classList.toggle("is-disabled", !selectable);
    card.dataset.modelId = String(model.id);
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "model_id";
    input.value = String(model.id);
    input.disabled = !selectable;

    const icon = document.createElement("span");
    icon.className = "selection-card__icon";
    icon.innerHTML = createSvg("model");
    const body = document.createElement("span");
    body.className = "selection-card__body";
    const title = document.createElement("span");
    title.className = "selection-card__title";
    const strong = document.createElement("strong");
    strong.textContent = model.label || model.id;
    const healthLabel = healthy ? "健康" : running ? "异常" : "未启动";
    const healthClass = healthy ? "is-good" : running ? "is-warn" : "is-offline";
    title.append(strong, makeTag(healthLabel, `health-tag ${healthClass}`));
    const description = document.createElement("p");
    description.textContent = model.description || "未提供模型描述";
    const meta = document.createElement("span");
    meta.className = "selection-card__meta";
    const recommendedGpu = state.bootstrap?.gpu_recommendations?.[String(model.id)]?.gpu_id;
    if (model.status?.running && model.gpu !== undefined && model.gpu !== null && model.gpu !== "") {
      meta.append(makeTag(`运行于 GPU ${model.gpu}`));
    } else if (recommendedGpu !== undefined) {
      meta.append(makeTag(`当前推荐 GPU ${recommendedGpu}`));
    } else if (model.configured_gpu !== undefined && model.configured_gpu !== null && model.configured_gpu !== "") {
      meta.append(makeTag(`回退 GPU ${model.configured_gpu}`));
    }
    if (model.port !== undefined) meta.append(makeTag(`:${model.port}`));
    if (variant?.label) meta.append(makeTag(String(variant.label)));
    if (variant?.normalization?.label) meta.append(makeTag(String(variant.normalization.label)));
    (model.tags || []).slice(0, 2).forEach((tag) => meta.append(makeTag(String(tag))));
    const modes = (model.supported_modes || []).map((mode) => String(mode).toUpperCase()).join(" · ");
    if (modes) meta.append(makeTag(modes));
    body.append(title, description, meta);
    const radio = document.createElement("span");
    radio.className = "selection-card__radio";
    radio.setAttribute("aria-hidden", "true");
    card.append(input, icon, body, radio);
    dom.modelOptions.append(card);
  });
}

function renderModelVersions({ preferred = "" } = {}) {
  const variant = selectedModelVariant();
  const versions = Array.isArray(variant?.versions) ? variant.versions : [];
  const previous = preferred || dom.modelVersion.value;
  dom.modelVersion.replaceChildren();
  if (!versions.length) {
    dom.modelVersion.append(new Option("请先选择兼容的模型", ""));
    dom.modelVersion.disabled = true;
    dom.modelVersionDescription.textContent = "所选硬件组合当前没有可部署的模型版本。";
    return;
  }
  versions.forEach((version) => {
    dom.modelVersion.append(new Option(String(version.label || version.id), String(version.id)));
  });
  const available = new Set(versions.map((item) => String(item.id)));
  const fallback = String(variant.default_version || versions[0].id);
  dom.modelVersion.value = available.has(previous) ? previous : fallback;
  dom.modelVersion.disabled = false;
  const selected = selectedModelVersion();
  dom.modelVersionDescription.textContent = selected?.description
    || "版本来自部署清单白名单，任务创建后不可修改。";
}

function formatMemory(mb) {
  const value = Number(mb);
  if (!Number.isFinite(value)) return "—";
  return value >= 1024 ? `${(value / 1024).toFixed(1)} GB` : `${value} MB`;
}

function selectedGpuRecommendation() {
  const model = selectedModel();
  if (!model?.id) return null;
  const recommendation = state.bootstrap?.gpu_recommendations?.[String(model.id)] || null;
  if (recommendation || model.loadable !== false) return recommendation;
  const minimumFree = Number(model.gpu_min_free_memory_mb || 0);
  const eligible = (state.bootstrap?.gpus || [])
    .filter((gpu) => !gpu.reserved_by_job && Number(gpu.memory_free_mb) >= minimumFree)
    .sort((left, right) => {
      const processDelta = Number(left.process_count > 0) - Number(right.process_count > 0);
      return processDelta || Number(right.memory_free_mb) - Number(left.memory_free_mb)
        || Number(left.utilization_percent) - Number(right.utilization_percent)
        || Number(left.id) - Number(right.id);
    });
  const selected = eligible[0];
  if (!selected) return null;
  return {
    gpu_id: String(selected.id),
    model_id: String(model.id),
    minimum_free_memory_mb: minimumFree,
    completely_idle: Number(selected.process_count) === 0,
    reason: `空闲 ${selected.memory_free_mb} MiB，利用率 ${selected.utilization_percent}%`,
  };
}

function effectiveSelectedGpuId() {
  if ((radioValue("server_mode") || "reuse") !== "real") return null;
  const requested = dom.gpuSelect.value || "auto";
  return requested === "auto" ? selectedGpuRecommendation()?.gpu_id ?? null : requested;
}

function syncGpuCardSelection() {
  const selectedId = effectiveSelectedGpuId();
  const automatic = (dom.gpuSelect.value || "auto") === "auto";
  dom.gpuGrid.querySelectorAll(".gpu-card[data-gpu-id]").forEach((card) => {
    const selected = selectedId !== null && String(card.dataset.gpuId) === String(selectedId);
    card.classList.toggle("is-selected", selected);
    card.setAttribute("aria-checked", String(selected));
    const badge = card.querySelector("[data-gpu-badge]");
    if (badge) {
      badge.textContent = selected
        ? (automatic ? "自动选中" : "已选择")
        : badge.dataset.baseLabel || "";
    }
  });
}

function syncGpuCardAvailability() {
  const interactive = (radioValue("server_mode") || "reuse") === "real" && !state.activeJob;
  dom.gpuGrid.querySelectorAll(".gpu-card[data-gpu-id]").forEach((card) => {
    const unavailable = card.dataset.unavailable === "true";
    const disabled = unavailable || !interactive;
    card.setAttribute("aria-disabled", String(disabled));
    card.tabIndex = disabled ? -1 : 0;
    card.classList.toggle("is-interaction-disabled", !interactive && !unavailable);
  });
}

function updateGpuSelectionHelp() {
  const serverMode = radioValue("server_mode") || "reuse";
  const model = selectedModel();
  const recommendation = selectedGpuRecommendation();
  if (serverMode === "reuse") {
    dom.gpuSelectionHelp.textContent = `复用服务沿用其已绑定的 GPU ${model?.gpu ?? "—"}，无需重新选卡。`;
  } else if (serverMode === "mock") {
    dom.gpuSelectionHelp.textContent = "Mock 测试服务在 CPU 上运行，不加载真实权重，也不占用 GPU。";
  } else if (dom.gpuSelect.value && dom.gpuSelect.value !== "auto") {
    const selected = (state.bootstrap?.gpus || []).find((gpu) => String(gpu.id) === String(dom.gpuSelect.value));
    dom.gpuSelectionHelp.textContent = selected
      ? `已手动选择 GPU ${selected.id}：空闲 ${formatMemory(selected.memory_free_mb)}，利用率 ${selected.utilization_percent}%。启动前会重新校验。`
      : `已手动选择 GPU ${dom.gpuSelect.value}；启动前会重新校验。`;
  } else if (recommendation) {
    dom.gpuSelectionHelp.textContent = `当前推荐 GPU ${recommendation.gpu_id}：${recommendation.reason}。启动前会重新检测并原子锁定。`;
  } else {
    dom.gpuSelectionHelp.textContent = "当前没有满足模型显存门槛的 GPU；可刷新状态或释放显存后重试。";
  }
}

function renderGpus(gpus) {
  const previous = dom.gpuSelect.value || state.defaults?.gpu_id || "auto";
  const recommendation = selectedGpuRecommendation();
  dom.gpuSelect.replaceChildren();
  const automatic = document.createElement("option");
  automatic.value = "auto";
  automatic.textContent = recommendation
    ? `自动选择 GPU ${recommendation.gpu_id}（推荐）`
    : "自动选择空闲 GPU（暂无可用卡）";
  dom.gpuSelect.append(automatic);
  dom.gpuGrid.replaceChildren();

  if (!gpus.length) {
    const empty = document.createElement("p");
    empty.className = "gpu-empty";
    empty.textContent = "暂时无法读取 GPU 状态；自动模式将使用模型配置中的回退卡。";
    dom.gpuGrid.append(empty);
    dom.gpuSelectionHelp.textContent = "GPU 监控不可用；手动选卡已关闭，启动时将再次检测。";
    return;
  }

  gpus.forEach((gpu) => {
    const model = selectedModel();
    const minimumFree = Number(model?.gpu_min_free_memory_mb || 0);
    const insufficient = Number(gpu.memory_free_mb) < minimumFree;
    const option = document.createElement("option");
    option.value = String(gpu.id);
    option.textContent = `GPU ${gpu.id} · ${formatMemory(gpu.memory_free_mb)} 空闲 · ${gpu.utilization_percent}%`;
    option.disabled = Boolean(gpu.reserved_by_job) || insufficient;
    dom.gpuSelect.append(option);

    const card = document.createElement("article");
    const busy = Number(gpu.process_count) > 0 || Boolean(gpu.reserved_by_job);
    const recommended = String(recommendation?.gpu_id) === String(gpu.id);
    card.className = `gpu-card ${busy ? "is-busy" : "is-idle"}${insufficient ? " is-unavailable" : ""}${recommended ? " is-recommended" : ""}`;
    card.dataset.gpuId = String(gpu.id);
    card.dataset.unavailable = String(Boolean(gpu.reserved_by_job) || insufficient);
    card.setAttribute("role", "radio");
    card.setAttribute("aria-label", `GPU ${gpu.id}，空闲 ${formatMemory(gpu.memory_free_mb)}`);
    card.setAttribute("aria-disabled", String(Boolean(gpu.reserved_by_job) || insufficient));
    card.tabIndex = Boolean(gpu.reserved_by_job) || insufficient ? -1 : 0;
    const heading = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = `GPU ${gpu.id}`;
    const badge = document.createElement("span");
    badge.textContent = gpu.reserved_by_job
      ? "平台预留"
      : insufficient
        ? "显存不足"
        : recommended
          ? (recommendation.completely_idle ? "推荐 · 空闲" : "推荐 · 低负载")
          : busy ? `${gpu.process_count} 个进程` : "空闲";
    badge.dataset.gpuBadge = "";
    badge.dataset.baseLabel = badge.textContent;
    heading.append(title, badge);
    const name = document.createElement("p");
    name.textContent = gpu.name || "NVIDIA GPU";
    const memory = document.createElement("progress");
    memory.className = "gpu-memory";
    const totalMemory = Math.max(0, Number(gpu.memory_total_mb));
    const freeMemory = Math.min(totalMemory, Math.max(0, Number(gpu.memory_free_mb)));
    memory.setAttribute("aria-label", `GPU ${gpu.id} 空闲显存`);
    memory.max = totalMemory || 1;
    memory.value = freeMemory;
    const stats = document.createElement("small");
    stats.textContent = `空闲显存 ${formatMemory(gpu.memory_free_mb)} / ${formatMemory(gpu.memory_total_mb)} · 利用率 ${gpu.utilization_percent}% · ${gpu.temperature_c}°C`;
    const processes = document.createElement("small");
    processes.className = "gpu-processes";
    processes.textContent = (gpu.processes || []).length
      ? (gpu.processes || []).map((item) => `${item.name} ${formatMemory(item.memory_used_mb)}`).join(" · ")
      : "无计算进程";
    card.append(heading, name, memory, stats, processes);
    const selectCard = () => {
      if (!gpu.reserved_by_job && !insufficient && !state.activeJob && radioValue("server_mode") === "real") {
        dom.gpuSelect.value = String(gpu.id);
        syncGpuCardSelection();
        updateGpuSelectionHelp();
        updateSummary();
        scheduleValidation();
        markDirty(true);
      }
    };
    card.addEventListener("click", selectCard);
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectCard();
      }
    });
    dom.gpuGrid.append(card);
  });
  dom.gpuSelect.value = [...dom.gpuSelect.options].some((item) => item.value === String(previous) && !item.disabled)
    ? String(previous)
    : "auto";
  syncGpuCardAvailability();
  syncGpuCardSelection();
  updateGpuSelectionHelp();
}

function updateModeAvailability() {
  const model = selectedModel();
  const version = selectedModelVersion();
  const supported = new Set((model?.supported_modes || []).map(String));
  const cards = [...document.querySelectorAll("[data-mode-card]")];
  const checkpointKey = selectedCheckpointKey();
  const checked = radioValue("inference_mode");
  if (state.rtcSelectionKey !== checkpointKey && parseInferenceSelection(checked).inference_mode === "rtc") {
    setRadioValue("inference_mode", inferenceSelectionValue({ inference_mode: "rtc" }, version));
  }
  state.rtcSelectionKey = checkpointKey;

  cards.forEach((card) => {
    const mode = card.dataset.modeCard;
    const input = card.querySelector("input");
    const canonical = parseInferenceSelection(mode).inference_mode;
    const available = Boolean(model && supported.has(canonical)
      && (mode !== "rtc-trained" || trainedRtcSupported(version)));
    input.disabled = !available;
    card.classList.toggle("is-disabled", !available);
    card.title = !model ? "请先选择策略模型" : !available ? `所选权重不支持 ${modeLabel(mode)} 模式` : "";
  });

  const selectedCard = cards.find((card) => card.dataset.modeCard === radioValue("inference_mode"));
  if (!selectedCard || selectedCard.querySelector("input").disabled) {
    const preferredRtc = inferenceSelectionValue({ inference_mode: "rtc" }, version);
    const fallback = [preferredRtc, "rtc-guided", "async", "sync"].find((mode) =>
      cards.some((card) => card.dataset.modeCard === mode && !card.querySelector("input").disabled));
    if (fallback) setRadioValue("inference_mode", fallback);
  }
  dom.modeSupportHint.textContent = model
    ? `支持：${cards.filter((card) => !card.querySelector("input").disabled).map((card) => modeLabel(card.dataset.modeCard)).join(" / ") || "无"}`
    : "选择模型后显示能力";
}

function updateEffectorCompatibility() {
  const robot = selectedRobot();
  const available = new Set((robot?.effectors?.available_profiles || []).map(String));
  let selectedCompatible = false;
  let firstCompatible = null;
  [...dom.effectorProfileOptions.querySelectorAll("[data-effector-profile]")].forEach((card) => {
    const input = card.querySelector('input[name="effector_profile"]');
    const compatible = Boolean(robot && available.has(String(input.value)));
    input.disabled = !compatible;
    card.classList.toggle("is-disabled", !compatible);
    card.title = compatible ? "" : "所选机器人未配置该夹爪";
    if (compatible && firstCompatible === null) firstCompatible = input;
    if (compatible && input.checked) selectedCompatible = true;
  });
  if (!selectedCompatible && firstCompatible) {
    const preferred = String(robot?.effectors?.profile || "");
    const preferredInput = [...dom.effectorProfileOptions.querySelectorAll('input[name="effector_profile"]')]
      .find((input) => !input.disabled && input.value === preferred);
    (preferredInput || firstCompatible).checked = true;
    applyEffectorProfileDefaults({ updateTask: true });
  }
}

function updateModelCompatibility() {
  const robot = selectedRobot();
  const effectorProfile = radioValue("effector_profile");
  const compatibleModels = new Set((robot?.compatible_models || []).map(String));
  const robotEffectorProfiles = new Set((robot?.effectors?.available_profiles || []).map(String));
  let selectedCompatible = false;
  let firstCompatible = null;
  [...dom.modelOptions.querySelectorAll("[data-model-id]")].forEach((card) => {
    const model = state.bootstrap?.models?.find((item) => String(item.id) === card.dataset.modelId);
    const input = card.querySelector('input[name="model_id"]');
    const selectable = model?.selectable !== false;
    const previewCompatible = Boolean(
      model?.ui_only === true
      && (model.compatible_robots || []).map(String).includes(String(robot?.id))
    );
    const compatible = Boolean(
      selectable
      && robot
      && model
      && (previewCompatible || !compatibleModels.size || compatibleModels.has(String(model.id)))
      && robotEffectorProfiles.has(effectorProfile)
      && (model.effector_profiles || []).map(String).includes(effectorProfile)
    );
    input.disabled = !compatible;
    card.classList.toggle("is-disabled", !compatible);
    card.title = compatible ? "" : "所选夹爪没有对应的 checkpoint、归一化或机器人配置";
    if (compatible && firstCompatible === null) firstCompatible = input;
    if (compatible && input.checked) selectedCompatible = true;
  });
  if (!selectedCompatible && firstCompatible) {
    firstCompatible.checked = true;
    applySmartServerMode();
  }
}

function updateSafetyProfiles() {
  const robot = selectedRobot();
  const profiles = robot?.safety_profiles || [];
  dom.safetyProfile.replaceChildren(new Option("G2 桌面工作区（默认）", "__display_default__"));

  profiles.forEach((profile) => {
    const option = document.createElement("option");
    option.value = String(profile.id);
    const deployable = profile.deployable !== false || !state.bootstrap?.constraints?.actions_enabled;
    option.textContent = `${profile.label || profile.id}${profile.calibrated ? " · 已标定" : " · 未标定"}${deployable ? "" : " · 未发布"}`;
    option.disabled = !profile.calibrated || !deployable;
    dom.safetyProfile.append(option);
  });

  // 展示态：复选框看起来开启，但准备页始终提交关闭的安全基线。
  dom.workspaceSafety.checked = true;
  dom.workspaceSafety.disabled = true;
  dom.safetyProfile.value = "__display_default__";
  dom.safetyProfile.disabled = true;
  dom.workspaceSafetyHelp.textContent = "已选择 G2 默认工作区检测配置。";
  dom.profileSelectWrap.hidden = false;
}

function updateDependentControls() {
  updateEffectorCompatibility();
  updateModelCompatibility();
  renderModelVersions();
  updateModeAvailability();
  updateSafetyProfiles();
  updateServerModeAvailability();
  const activeMode = parseInferenceSelection(radioValue("inference_mode")).inference_mode;
  document.querySelectorAll("[data-modes]").forEach((field) => {
    const visible = field.dataset.modes.split(/\s+/).includes(activeMode);
    field.classList.toggle("is-inactive", !visible);
    field.hidden = !visible;
  });
  const modeCopy = {
    sync: ["同步模式参数", "逐块等待模型响应；可配置请求超时与预热次数。"],
    async: ["异步模式参数", "配置提前请求、重规划间隔与延迟统计窗口。"],
    rtc: ["RTC 模式参数", "在异步请求基础上配置动作前缀融合；条件不足时退化为异步。"],
  }[activeMode];
  dom.activeModePill.textContent = modeLabel(radioValue("inference_mode") || activeMode);
  dom.modeParameterTitle.textContent = modeCopy?.[0] || "模式参数";
  dom.modeParameterDescription.textContent = modeCopy?.[1] || "";
  dom.hitlCard.classList.toggle("is-active", dom.hitl.checked);
  dom.profileSelectWrap.hidden = false;
  const serverMode = radioValue("server_mode") || "reuse";
  const loadsRealModel = serverMode === "real";
  const fixedThorGpu = state.bootstrap?.project?.site_id === "thor";
  dom.computeResourceNote.textContent = fixedThorGpu
    ? "Thor 仅有 GPU 0；启动真实 PI0.5 时固定使用 cuda:0。复用与 Mock 将自动沿用对应资源。"
    : "启动真实模型时选择计算资源；复用与 Mock 将自动沿用对应资源。";
  dom.gpuSelect.disabled = !loadsRealModel;
  dom.gpuControls.hidden = !loadsRealModel || fixedThorGpu;
  dom.refreshGpus.hidden = !loadsRealModel || fixedThorGpu;
  dom.thorFixedGpu.hidden = !loadsRealModel || !fixedThorGpu;
  dom.resourceModeSummary.hidden = loadsRealModel;
  dom.resourceModeSummary.classList.toggle("is-mock", serverMode === "mock");
  if (serverMode === "reuse") {
    dom.resourceModeSummaryTitle.textContent = `沿用健康服务 · GPU ${selectedModel()?.gpu ?? "—"}`;
    dom.resourceModeSummaryNote.textContent = "不重新加载权重或迁移显存；任务连接时仍会校验协议与模型 Profile。";
  } else if (serverMode === "mock") {
    dom.resourceModeSummaryTitle.textContent = "CPU · Mock 测试服务";
    dom.resourceModeSummaryNote.textContent = "不加载真实权重，仅验证服务启动、协议握手与只读数据链路，不能代表模型推理效果。";
  }
  syncGpuCardAvailability();
  syncGpuCardSelection();
  updateGpuSelectionHelp();
  updatePrimaryActionLabel();
}

function valueIsDifferent(value, defaultValue) {
  if (typeof value === "number" && typeof defaultValue === "number") {
    return Number.isFinite(value) && Math.abs(value - defaultValue) > 1e-9;
  }
  return value !== defaultValue;
}

function formatChangedValue(key, value) {
  const units = {
    duration_s: " s",
    execution_fps: " FPS",
    servo_fps: " Hz",
    prediction_horizon: " 步",
    quality_horizon: " 步",
    request_lead_margin_steps: " 步",
    min_replan_after_steps: " 步",
    latency_window_size: " 次",
    request_timeout_s: " s",
    min_rtc_blend_steps: " 步",
    watchdog_timeout_s: " s",
    approval_timeout_s: " s",
  };
  if (key === "duration_s" && Number(value) === 0) return "0（一直执行）";
  return `${value}${units[key] || ""}`;
}

function updateDurationDisplay() {
  const continuous = Number(dom.durationInput.value) === 0;
  dom.durationUnit.textContent = continuous ? "（一直执行）" : "秒";
  dom.durationUnit.classList.toggle("is-continuous", continuous);
}

function updateSummary() {
  if (!state.loaded) return;
  const config = getConfiguration();
  const devicePosition = config.effector_control_mode === "device-position";
  ["effector_action_low_pass_cutoff_hz", "max_effector_velocity", "max_effector_acceleration"].forEach((key) => {
    const input = document.querySelector(`[data-config-key="${key}"]`);
    if (input) {
      input.disabled = devicePosition;
      input.title = devicePosition ? "持续位置模式使用设备内部运动规划；此参数只用于旧版软件轨迹" : "";
    }
  });
  const robot = selectedRobot();
  const model = selectedModel();
  dom.taskCount.textContent = String(dom.taskInput.value.length);
  dom.summaryRobot.textContent = robot?.label || robot?.id || "未选择";
  const effectorProfile = selectedEffectorProfile();
  const modelVariant = selectedModelVariant(model);
  const modelVersion = selectedModelVersion();
  dom.summaryEffector.textContent = effectorProfile?.label || "未选择";
  const probeMatches = state.robotProbe
    && state.robotProbe.robot_id === config.robot_id
    && state.robotProbe.host === config.robot_host
    && state.robotProbe.port === config.robot_ssh_port;
  const endpointState = probeMatches ? ` · ${robotProbePresentation(state.robotProbe).label}` : "";
  dom.summaryEndpoint.textContent = config.robot_host ? `${config.robot_host}:${config.robot_ssh_port || "—"}${endpointState}` : "未配置";
  dom.summaryModel.textContent = model?.label || model?.id || "未选择";
  dom.summaryVersion.textContent = modelVersion?.label || modelVersion?.id || "未选择";
  dom.summaryNormalization.textContent = modelVersion?.normalization?.label || modelVariant?.normalization?.label || "—";
  dom.summaryTask.textContent = config.task || "尚未填写";
  dom.summaryTask.title = config.task;
  dom.summaryInference.textContent = modeLabel(config.inference_mode);
  dom.summaryHardware.textContent = (
    state.bootstrap?.constraints?.physical_execute_available === true && robot?.execution_supported === true
  ) ? "准备阶段只读 · 运行页预选真机" : (HARDWARE_LABELS[config.hardware_mode] || config.hardware_mode);
  dom.summaryServer.textContent = SERVER_LABELS[config.server_mode] || config.server_mode;
  dom.summaryGpu.textContent = config.server_mode === "reuse"
    ? `沿用 GPU ${model?.gpu ?? "—"}`
    : config.server_mode === "mock"
      ? "CPU（Mock）"
      : state.bootstrap?.project?.site_id === "thor"
        ? "Thor · GPU 0（固定 CUDA:0）"
        : config.gpu_id === "auto"
      ? (selectedGpuRecommendation() ? `自动 · GPU ${selectedGpuRecommendation().gpu_id}` : "自动 · 暂无可用卡")
      : `GPU ${config.gpu_id}`;
  dom.summaryHitl.textContent = config.human_in_loop ? "已启用 · 平板逐包采集" : "关闭";
  dom.summaryDuration.textContent = Number.isFinite(config.duration_s) ? (config.duration_s === 0 ? "0（一直执行）" : `${config.duration_s} 秒`) : "—";
  dom.summaryFps.textContent = Number.isFinite(config.execution_fps) && Number.isFinite(config.servo_fps) ? `${config.execution_fps} / ${config.servo_fps}` : "—";
  dom.summaryHorizon.textContent = Number.isFinite(config.prediction_horizon) && Number.isFinite(config.quality_horizon) ? `${config.prediction_horizon} / ${config.quality_horizon}` : "—";
  const safetyEnabled = Object.entries(config.safety).filter(([key, enabled]) => key.endsWith("_enabled") && enabled).length;
  dom.summarySafety.textContent = `${safetyEnabled} 项开启`;

  const changes = [];
  Object.keys(PARAMETER_LABELS).forEach((key) => {
    if (key === "approval_timeout_s") return;
    if (key === "min_rtc_blend_steps" && config.inference_mode !== "rtc") return;
    if (valueIsDifferent(config[key], state.defaults[key])) {
      changes.push({ label: PARAMETER_LABELS[key], value: formatChangedValue(key, config[key]) });
    }
  });
  if (config.inference_mode !== state.defaults.inference_mode
      || config.rtc_inference_mode !== rtcInferenceMode(state.defaults, selectedModelVersion())) {
    changes.unshift({ label: "推理模式", value: modeLabel(config.inference_mode) });
  }
  // Hardware mode is deliberately staged as read-only before model load;
  // the runtime page proposes execute and requires a validated save.
  if (config.server_mode !== state.defaults.server_mode) changes.unshift({ label: "模型服务", value: SERVER_LABELS[config.server_mode] });
  if (config.human_in_loop !== Boolean(state.defaults.human_in_loop)) changes.unshift({ label: "人在环路", value: config.human_in_loop ? "开启" : "关闭" });
  ["low_pass_filter_enabled", "effector_low_pass_filter_enabled", "hold_effectors", "workspace_safety_enabled"].forEach((key) => {
    if (config.safety[key] !== Boolean(state.defaults.safety?.[key])) {
      const labels = { low_pass_filter_enabled: "手臂低通滤波", effector_low_pass_filter_enabled: "末端低通滤波", hold_effectors: "保持末端执行器", workspace_safety_enabled: "碰撞危险拦截" };
      changes.push({ label: labels[key], value: config.safety[key] ? "开启" : "关闭" });
    }
  });

  dom.changedCount.textContent = String(changes.length);
  dom.changesList.replaceChildren();
  if (!changes.length) {
    const li = document.createElement("li");
    li.textContent = "当前使用全部默认值";
    dom.changesList.append(li);
  } else {
    changes.slice(0, 10).forEach((change) => {
      const li = document.createElement("li");
      const label = document.createElement("span");
      const value = document.createElement("span");
      label.textContent = change.label;
      value.textContent = change.value;
      li.append(label, value);
      dom.changesList.append(li);
    });
    if (changes.length > 10) {
      const li = document.createElement("li");
      li.textContent = `另有 ${changes.length - 10} 项调整`;
      dom.changesList.append(li);
    }
  }
}

function localValidation(config) {
  const errors = [];
  const warnings = [];
  const addError = (field, message) => errors.push({ field, message });

  if (!config.robot_id) addError("robot_id", "请选择部署机器人。");
  if (!config.effector_profile) addError("effector_profile", "请选择夹爪类型。");
  if (!config.robot_host) addError("robot_host", "请填写机器人 IP。");
  if (!Number.isInteger(config.robot_ssh_port) || config.robot_ssh_port < 1 || config.robot_ssh_port > 65535) addError("robot_ssh_port", "SSH 端口必须是 1–65535 的整数。");
  if (!config.model_id) addError("model_id", "请选择策略模型。");
  if (!config.model_version) addError("model_version", "请选择模型版本。");
  if (!config.task) addError("task", "请填写任务指令。");
  if (config.task.length > taskMaxLength()) addError("task", `任务指令不可超过 ${taskMaxLength()} 个字符。`);

  const robot = selectedRobot();
  const model = selectedModel();
  if (robot?.online === false) addError("robot_id", "所选机器人当前离线。");
  const probeMatches = state.robotProbe
    && state.robotProbe.robot_id === config.robot_id
    && state.robotProbe.host === config.robot_host
    && state.robotProbe.port === config.robot_ssh_port;
  if (state.bootstrap?.constraints?.actions_enabled && !robot?.ui_only) {
    if (!probeMatches) addError("robot_host", "请先检测机器人实例，确认客户端已部署且空闲。");
    else if (!state.robotProbe.reachable) addError("robot_host", "机器人网络不可达。");
    else if (!state.robotProbe.ssh_authenticated) addError("robot_host", "SSH 身份或该实例的凭据映射未通过。");
    else if (state.robotProbe.client?.status === "not_deployed") addError("robot_host", "Nexus 机器人客户端尚未部署。");
    else if (state.robotProbe.client?.status === "running") addError("robot_host", "机器人客户端已在运行，当前实例被占用。");
    else if (state.robotProbe.client?.status !== "idle") addError("robot_host", "无法确认机器人客户端处于可启动状态。");
  } else if (!probeMatches) {
    warnings.push("机器人实例尚未完成网络、SSH 和客户端状态检测。");
  }
  if (model && !(model.supported_modes || []).includes(config.inference_mode)) {
    addError("inference_mode", `所选模型不支持${modeLabel(config.inference_mode)}模式。`);
  }
  if (model && config.server_mode === "reuse" && !modelHasHealthyService(model)) {
    addError("server_mode", "复用模式要求相同夹爪和版本的模型服务正在运行且健康；请重新加载模型。");
  }
  if (config.hardware_mode === "execute" && robot?.execution_supported !== true) {
    addError("hardware_mode", "当前机器人或控制服务未开放真机执行。");
  }
  if (config.hardware_mode === "execute" && robot?.effectors?.mode === "absent_fixed") {
    addError("hardware_mode", "该机器人未安装夹爪，末端状态仅用于补齐只读模型输入，禁止真机执行。");
  }

  const positive = [
    "execution_fps", "servo_fps", "prediction_horizon", "quality_horizon", "watchdog_timeout_s",
    "request_timeout_s", "min_replan_after_steps", "latency_window_size",
    "action_low_pass_cutoff_hz", "effector_action_low_pass_cutoff_hz", "max_arm_velocity",
    "max_arm_acceleration", "max_effector_velocity", "max_effector_acceleration",
    "max_head_waist_velocity", "max_head_waist_acceleration", "max_arm_tracking_error",
    "max_effector_tracking_error", "max_head_waist_tracking_error",
  ];
  positive.forEach((key) => {
    if (!Number.isFinite(config[key]) || config[key] <= 0) addError(key, `${PARAMETER_LABELS[key] || key}必须为正数。`);
  });
  ["duration_s", "request_lead_margin_steps", "min_rtc_blend_steps", "warmup_requests", "execution_preflight_requests"].forEach((key) => {
    if (!Number.isFinite(config[key]) || config[key] < 0) addError(key, `${PARAMETER_LABELS[key] || key}不可为负数。`);
  });
  if (config.quality_horizon > config.prediction_horizon) addError("quality_horizon", "可信执行步长不可大于预测步长。");
  if (config.action_low_pass_cutoff_hz >= config.servo_fps / 2) addError("action_low_pass_cutoff_hz", "手臂低通截止频率必须低于伺服频率的一半。");
  if (config.effector_action_low_pass_cutoff_hz >= config.servo_fps / 2) addError("effector_action_low_pass_cutoff_hz", "末端低通截止频率必须低于伺服频率的一半。");
  if (config.duration_s === 0) warnings.push("运行时长为 0，任务会一直执行，直至手动暂停或结束任务。");
  if (config.inference_mode !== "rtc" && config.min_rtc_blend_steps !== state.defaults.min_rtc_blend_steps) warnings.push("RTC 最小融合步数仅在 RTC 模式下生效。");

  return { valid: errors.length === 0, errors, warnings };
}

function clearFieldErrors() {
  document.querySelectorAll("[data-error-for]").forEach((element) => {
    element.hidden = true;
    element.textContent = "";
  });
  document.querySelectorAll(".has-error").forEach((element) => element.classList.remove("has-error"));
  dom.parameterErrors.replaceChildren();
}

function showFieldErrors(errors = []) {
  clearFieldErrors();
  const grouped = new Map();
  errors.forEach((error) => {
    const field = error.field || "configuration";
    if (!grouped.has(field)) grouped.set(field, []);
    grouped.get(field).push(error.message || "配置无效");
  });

  grouped.forEach((messages, field) => {
    const direct = document.querySelector(`[data-error-for="${CSS.escape(field)}"]`);
    if (direct) {
      direct.textContent = messages.join("；");
      direct.hidden = false;
    }
    const selector = FIELD_SELECTORS[field] || `[data-config-key="${CSS.escape(field)}"]`;
    const target = document.querySelector(selector);
    if (target) target.closest(".form-field, .field-group, .number-input")?.classList.add("has-error");

    if (!direct && PARAMETER_LABELS[field]) {
      const item = document.createElement("p");
      item.textContent = messages.join("；");
      dom.parameterErrors.append(item);
      if (["action_low_pass_cutoff_hz", "effector_action_low_pass_cutoff_hz", "max_arm_velocity", "max_arm_acceleration", "max_effector_velocity", "max_effector_acceleration", "max_head_waist_velocity", "max_head_waist_acceleration", "max_arm_tracking_error", "max_effector_tracking_error", "max_head_waist_tracking_error", "request_lead_margin_steps", "min_rtc_blend_steps", "warmup_requests", "execution_preflight_requests"].includes(field)) {
        dom.advancedParameters.open = true;
      }
    }
  });
}

function setValidationView(kind, title, subtitle, errors = [], warnings = [], summary = null) {
  dom.validationCard.className = `validation-card ${kind ? `is-${kind}` : ""}`.trim();
  dom.validationTitle.textContent = title;
  dom.validationSubtitle.textContent = subtitle;
  dom.validationMessages.replaceChildren();
  [...errors.map((item) => ({ ...item, tone: "error" })), ...warnings.map((item) => ({ message: typeof item === "string" ? item : item.message, tone: "warning" }))].forEach((item) => {
    const li = document.createElement("li");
    li.className = `is-${item.tone}`;
    li.textContent = item.message;
    dom.validationMessages.append(li);
  });
  renderServerSummary(summary);

  const statusMap = {
    valid: ["is-valid", warnings.length ? "校验通过 · 有提示" : "校验通过"],
    invalid: ["is-invalid", "需要调整"],
    checking: ["is-checking", "校验中"],
    draft: ["is-draft", "待校验"],
  };
  const [className, text] = statusMap[kind] || statusMap.draft;
  dom.summaryStatus.className = `summary-status ${className}`;
  dom.summaryStatus.textContent = text;
}

function renderServerSummary(summary) {
  dom.serverSummary.replaceChildren();
  if (summary === undefined || summary === null || summary === "") {
    dom.serverSummary.hidden = true;
    return;
  }
  dom.serverSummary.hidden = false;
  if (typeof summary !== "object") {
    dom.serverSummary.textContent = String(summary);
    return;
  }
  const details = document.createElement("details");
  const heading = document.createElement("summary");
  heading.textContent = "查看技术校验详情";
  const dl = document.createElement("dl");
  Object.entries(summary).slice(0, 8).forEach(([key, value]) => {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = key;
    dd.textContent = typeof value === "object" ? JSON.stringify(value) : String(value);
    dl.append(dt, dd);
  });
  details.append(heading, dl);
  dom.serverSummary.append(details);
}

function setActionsEnabled() {
  const available = state.loaded && !state.bootstrapping && !state.validating && !state.deploying && !state.activeJob;
  dom.validateButton.disabled = !available;
  const loadable = selectedModel()?.loadable !== false;
  dom.submitButtons.forEach((button) => { button.disabled = !available || !loadable; });
  syncGpuCardAvailability();
}

function markDirty(schedule = true) {
  state.dirty = true;
  state.validation = null;
  if (state.loaded && !state.validating) {
    const config = getConfiguration();
    const local = localValidation(config);
    showFieldErrors(local.errors);
    if (local.valid) {
      setValidationView("draft", "配置尚未校验", "停止输入后将自动检查兼容性", [], local.warnings);
    } else {
      setValidationView("invalid", `${local.errors.length} 项需要调整`, "请修正标记字段后重新验证", local.errors, local.warnings);
    }
  }
  setActionsEnabled();
  if (schedule) scheduleValidation();
}

function scheduleValidation() {
  window.clearTimeout(state.validationTimer);
  if (!state.loaded || state.activeJob) return;
  const local = localValidation(getConfiguration());
  if (!local.valid) return;
  state.validationTimer = window.setTimeout(() => validateConfiguration({ silent: true }), 650);
}

async function validateConfiguration({ silent = false } = {}) {
  if (!state.loaded || state.validating) return null;
  const config = getConfiguration();
  const local = localValidation(config);
  showFieldErrors(local.errors);
  if (!local.valid) {
    state.validation = local;
    setValidationView("invalid", `${local.errors.length} 项需要调整`, "本地基础校验未通过", local.errors, local.warnings);
    if (!silent) focusFirstError(local.errors);
    return local;
  }

  if (selectedModel()?.loadable === false) {
    const preview = {
      ...local,
      normalized: config,
      sourceConfig: config,
    };
    state.validation = preview;
    state.dirty = false;
    setValidationView("valid", "配置与目标兼容", "模型配置可查看", [], local.warnings);
    return preview;
  }

  state.validating = true;
  const sequence = ++state.validationSequence;
  setActionsEnabled();
  setButtonLoading(dom.validateButton, true);
  setValidationView("checking", "正在校验配置", "检查目标能力、参数约束和安全前置条件");

  try {
    const result = await apiFetch("/validate", { method: "POST", body: JSON.stringify(config) });
    if (sequence !== state.validationSequence) return null;
    const errors = Array.isArray(result?.errors) ? result.errors : [];
    const warnings = Array.isArray(result?.warnings) ? result.warnings : [];
    const valid = Boolean(result?.valid) && errors.length === 0;
    state.validation = { ...result, valid, errors, warnings, sourceConfig: config };
    state.dirty = false;
    showFieldErrors(errors);
    setValidationView(valid ? "valid" : "invalid", valid ? "配置与目标兼容" : `${errors.length || 1} 项校验未通过`, valid ? "可以准备模型服务" : "请根据提示调整后重试", errors, warnings, result?.summary);
    if (!silent) toast(valid ? "配置校验通过" : "配置仍有未通过项", valid ? "info" : "warning");
    return state.validation;
  } catch (error) {
    if (sequence !== state.validationSequence) return null;
    if (isAbortLike(error)) return null;
    const errors = [{ field: "configuration", message: error.message }];
    state.validation = { valid: false, errors, warnings: [] };
    setValidationView("invalid", "服务端校验失败", "控制服务未返回有效结果", errors);
    if (!silent) toast(`校验失败：${error.message}`, "error");
    return state.validation;
  } finally {
    if (sequence === state.validationSequence) {
      state.validating = false;
      setButtonLoading(dom.validateButton, false);
      setActionsEnabled();
    }
  }
}

function focusFirstError(errors) {
  const field = errors?.[0]?.field;
  if (!field) return;
  const selector = FIELD_SELECTORS[field] || `[data-config-key="${CSS.escape(field)}"]`;
  const element = document.querySelector(selector);
  element?.scrollIntoView({ behavior: "smooth", block: "center" });
  element?.focus?.({ preventScroll: true });
}

function occupiedDeploymentId(message) {
  const match = String(message || "").match(/(?:模型已有部署任务占用|机器人已有任务占用)：([0-9a-f]{32})/i);
  return match?.[1] || null;
}

async function surfaceOccupiedDeployment(message) {
  const jobId = occupiedDeploymentId(message);
  if (!jobId) return false;
  try {
    const job = await apiFetch(`/deployments/${encodeURIComponent(jobId)}`);
    renderJob(job);
    lockForm(true);
    if (state.activeJob) startPolling(jobId);
    dom.jobCard.scrollIntoView({ behavior: "smooth", block: "center" });
    toast("已找到当前占用任务，请在机器人控制台查看并处理。", "warning", 6000);
    return true;
  } catch (lookupError) {
    if (!isAbortLike(lookupError)) {
      toast(`占用任务 ${jobId.slice(0, 8)} 无法载入：${lookupError.message}`, "error", 6000);
    }
    return false;
  }
}

async function createDeployment() {
  if (state.deploying || state.activeJob) return;
  if (selectedModel()?.loadable === false) return;
  state.deploying = true;
  window.clearTimeout(state.validationTimer);
  setActionsEnabled();
  dom.submitButtons.forEach((button) => setButtonLoading(button, true));

  try {
    const validation = await validateConfiguration({ silent: true });
    if (validation === null) return;
    if (!validation?.valid) {
      focusFirstError(validation?.errors || []);
      toast("创建已暂停，请先修正校验问题。", "warning");
      return;
    }
    // Submit only browser-owned form fields.  The validation response also
    // contains immutable server policy (for example presence-gate timings),
    // which the server intentionally refuses to accept from a browser.
    const config = getConfiguration();
    const submitDeployment = () => apiFetch("/deployments", {
      method: "POST",
      headers: { "X-CSRF-Token": state.csrfToken },
      body: JSON.stringify(config),
    });
    const job = await submitDeployment();
    if (!job?.id) throw new Error("控制服务未返回部署任务 ID");
    window.location.assign(`./session.html?id=${encodeURIComponent(job.id)}`);
    return;
  } catch (error) {
    if (isAbortLike(error)) return;
    const message = String(error.message || "");
    const surfacedJob = await surfaceOccupiedDeployment(message);
    // A robot process without a tracked deployment is handled by the guarded
    // two-step cleanup dialog. Model/job conflicts are handled above by
    // loading their owning deployment, whose normal stop action is auditable.
    if (!surfacedJob && state.robotProbe && /当前实例被占用/.test(message)) {
      openCleanupDialog(state.robotProbe, { force: true });
    }
    toast(`无法创建部署：${error.message}`, "error", 6000);
    setValidationView("invalid", "创建部署失败", "配置仍保留，可修正后重试", [{ message: error.message }]);
  } finally {
    state.deploying = false;
    dom.submitButtons.forEach((button) => setButtonLoading(button, false));
    setActionsEnabled();
  }
}

function lockForm(locked) {
  [...dom.form.elements].forEach((element) => {
    if (element.matches("[data-submit-deployment]")) return;
    if (locked) {
      element.dataset.wasDisabled = String(element.disabled);
      element.disabled = true;
    } else if (element.dataset.wasDisabled !== undefined) {
      element.disabled = element.dataset.wasDisabled === "true";
      delete element.dataset.wasDisabled;
    }
  });
  dom.form.classList.toggle("is-locked", locked);
}

function normalizedJobStatus(job) {
  return String(job?.status || job?.state || "queued").toLowerCase().replaceAll("-", "_");
}

function renderJob(job) {
  state.activeJob = job;
  const statusKey = normalizedJobStatus(job);
  const meta = JOB_STATUS[statusKey] || { label: statusKey || "处理中", stage: 1, tone: "is-active", progress: 20 };
  dom.jobCard.hidden = false;
  dom.jobStatus.className = `job-status ${meta.tone}`;
  dom.jobStatus.textContent = meta.label;
  dom.jobId.textContent = String(job.id || "—");
  dom.jobId.title = String(job.id || "");
  dom.jobMessage.textContent = job.message || job.detail || job.error || defaultJobMessage(statusKey);
  const candidateOnly = job.model_service?.candidate_only === true;
  dom.jobConsoleLink.href = `${candidateOnly ? "./session.html" : "./monitor"}?id=${encodeURIComponent(job.id)}`;
  dom.jobConsoleLink.textContent = candidateOnly ? "进入运行配置并应用" : "在机器人控制台查看任务";
  renderModelService(job, meta.tone === "is-failed");
  renderTaskSession(job, statusKey);
  renderJobEvents(job.events || job.logs || []);

  if (meta.terminal) {
    stopPolling();
    state.activeJob = null;
    lockForm(false);
    setActionsEnabled();
    if (meta.tone === "is-success") toast("部署任务已完成");
    else toast(job.error || job.message || meta.label, "error", 6000);
  }
}

function defaultJobMessage(status) {
  const messages = {
    queued: "任务已进入执行队列。",
    validating: "正在重新确认配置快照与目标状态。",
    starting_server: "正在准备所选模型服务。",
    deploying_client: "正在发布并启动机器人端客户端。",
    warmup: "正在预热模型，首批响应不会被执行。",
    preflight: "正在生成白名单配置计划；运行目标能力将在连接时握手。",
    waiting_approval: "旧版批准状态正在自动迁移为等待开始。",
    model_ready: "模型服务已经完成健康检查。",
    ready_to_start: "模型服务已就绪，任务尚未开始。",
    task_starting: "正在建立任务控制会话。",
    running: "任务正在按已冻结的配置运行。",
    paused: "控制面会话已暂停，等待继续或结束。",
    completed: "任务已安全结束。",
    failed: "任务执行失败，请查看运行记录。",
    rejected: "操作员已拒绝本次任务。",
  };
  return messages[status] || "部署任务正在处理。";
}

function renderModelService(job, failed) {
  const service = job.model_service && typeof job.model_service === "object" ? job.model_service : {};
  const serviceStatus = String(service.status || "pending");
  const labels = { pending: "等待检查", starting: "加载中", ready: "服务健康", simulated: "模拟完成" };
  dom.modelServiceStatus.textContent = labels[serviceStatus] || serviceStatus;
  dom.modelServiceStatus.className = `runtime-status ${failed ? "is-failed" : serviceStatus === "ready" ? "is-ready" : serviceStatus === "starting" ? "is-active" : ""}`;
  dom.modelServiceStrategy.textContent = SERVER_LABELS[service.strategy] || service.strategy || "—";
  let phaseLabel = service.phase_label || "等待检查";
  if (service.shard_current && service.shard_total) phaseLabel += ` · 分片 ${service.shard_current}/${service.shard_total}`;
  dom.modelServicePhase.textContent = phaseLabel;
  dom.modelServiceElapsed.textContent = service.started_at == null
    ? serviceStatus === "ready" ? "已复用服务" : "未开始"
    : formatDuration(service.elapsed_s || 0);
  dom.modelServiceTimeout.textContent = Number.isFinite(Number(service.timeout_s)) ? formatDuration(Number(service.timeout_s)) : "—";
  dom.modelServiceTarget.textContent = `${service.gpu != null ? `GPU ${service.gpu}` : "GPU —"} · ${service.port ? `:${service.port}` : "端口 —"}`;
  dom.modelLastLog.textContent = service.recent_log || job.message || "等待模型服务检查";
  dom.modelLastLog.title = dom.modelLastLog.textContent;
  const phase = String(service.phase || "checking");
  const activeIndex = MODEL_PHASE_INDEX[phase] ?? 0;
  const complete = serviceStatus === "ready" || serviceStatus === "simulated";
  let modelProgress = complete ? 100 : [8, 18, 30, 52, 70, 84, 94][activeIndex];
  if (service.shard_current && service.shard_total) {
    modelProgress = 38 + 30 * Number(service.shard_current) / Math.max(1, Number(service.shard_total));
  }
  dom.jobProgress.value = Math.min(100, Math.max(4, modelProgress));
  renderJobStages(activeIndex, failed, complete);
}

function renderTaskSession(job, status) {
  const labels = {
    queued: "等待模型",
    validating: "等待模型",
    preflight: "配置冻结中",
    deploying: "等待模型",
    model_ready: "模型已就绪",
    waiting_approval: "迁移旧会话",
    ready_to_start: "等待开始",
    task_starting: "启动中",
    running: "运行中",
    paused: "已暂停",
    succeeded: "已结束",
    failed: "启动失败",
    rejected: "已拒绝",
  };
  dom.taskSessionStatus.textContent = labels[status] || "尚未开始";
  dom.taskSessionStatus.className = `runtime-status ${status === "running" ? "is-ready" : status === "paused" || status === "ready_to_start" || status === "waiting_approval" ? "is-waiting" : status === "failed" || status === "rejected" ? "is-failed" : ""}`;
  const notes = {
    waiting_approval: "旧版批准状态正在自动迁移为等待开始。",
    ready_to_start: "模型服务已就绪；任务尚未开始，也没有向机器人发送动作。",
    task_starting: "正在建立机器人、相机与模型推理会话。",
    running: "机器人客户端和推理会话正在运行；可原地软暂停。",
    paused: "软暂停已确认；旧动作计划已清空，客户端连接保持。",
    succeeded: "任务会话已结束；模型服务保持运行，可供下一任务复用。",
  };
  dom.taskSessionNote.textContent = notes[status] || "模型服务就绪后，前往机器人控制台操作任务。";
}

function formatDuration(value) {
  const seconds = Math.max(0, Math.round(Number(value) || 0));
  if (seconds < 60) return `${seconds} 秒`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}分 ${String(seconds % 60).padStart(2, "0")}秒`;
}

function renderJobStages(activeIndex, failed, complete = false) {
  dom.jobStages.replaceChildren();
  MODEL_STAGES.forEach((label, index) => {
    const li = document.createElement("li");
    li.textContent = label;
    if (failed && index === activeIndex) li.className = "is-failed";
    else if (index < activeIndex || complete) li.className = "is-complete";
    else if (index === activeIndex) li.className = "is-active";
    dom.jobStages.append(li);
  });
}

function renderJobEvents(events) {
  if (!Array.isArray(events) || !events.length) {
    dom.jobEventsWrap.hidden = true;
    return;
  }
  dom.jobEventsWrap.hidden = false;
  dom.jobEvents.replaceChildren();
  events.slice(-20).reverse().forEach((event) => {
    const li = document.createElement("li");
    const time = document.createElement("time");
    const message = document.createElement("span");
    if (typeof event === "string") {
      time.textContent = "";
      message.textContent = event;
    } else {
      const rawTime = event.timestamp || event.created_at || event.time;
      time.textContent = rawTime ? formatTime(rawTime) : "";
      message.textContent = event.message || event.detail || event.status || JSON.stringify(event);
    }
    li.append(time, message);
    dom.jobEvents.append(li);
  });
}

function formatTime(value) {
  const normalized = typeof value === "number" && value < 10_000_000_000 ? value * 1000 : value;
  const date = new Date(normalized);
  return Number.isNaN(date.getTime()) ? String(value).slice(0, 8) : date.toLocaleTimeString("zh-CN", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function startPolling(jobId) {
  stopPolling();
  const generation = state.pollGeneration;
  const poll = async () => {
    if (state.destroyed || generation !== state.pollGeneration) return;
    if (document.hidden) { state.pollTimer = window.setTimeout(poll, 1500); return; }
    try {
      const job = await apiFetch(`/deployments/${encodeURIComponent(jobId)}`);
      if (state.destroyed || generation !== state.pollGeneration) return;
      renderJob(job);
      if (state.pollErrorShown) toast("部署进度连接已恢复", "success");
      state.pollFailures = 0;
      state.pollErrorShown = false;
      if (state.activeJob) state.pollTimer = window.setTimeout(poll, 1400);
    } catch (error) {
      if (state.destroyed || generation !== state.pollGeneration) return;
      if (!isAbortLike(error)) {
        state.pollFailures += 1;
        if (state.pollFailures >= 3 && !state.pollErrorShown) {
          toast(`部署进度连续更新失败：${error.message}。页面会自动重试。`, "warning");
          state.pollErrorShown = true;
        }
      }
      if (state.activeJob) state.pollTimer = window.setTimeout(poll, 3000);
    }
  };
  state.pollTimer = window.setTimeout(poll, 700);
}

function stopPolling() {
  state.pollGeneration = (state.pollGeneration || 0) + 1;
  window.clearTimeout(state.pollTimer);
  state.pollTimer = null;
  state.pollFailures = 0;
  state.pollErrorShown = false;
}

async function refreshCurrentModel() {
  try {
    const station = await apiFetch("/workstation");
    const id = station.active_deployment_id;
    dom.currentModelCard.hidden = !id;
    if (!id) return;
    const job = await apiFetch(`/deployments/${encodeURIComponent(id)}`);
    const service = job.model_service || {};
    const model = state.bootstrap?.models?.find((item) => item.id === job.config?.model_id);
    dom.currentModelSummary.textContent = `${model?.label || job.config?.model_id || "模型"} · ${job.config?.model_version || "版本未知"} · ${service.healthy ? "服务健康" : "服务未就绪"}`;
    dom.currentModelConfig.href = `./session.html?id=${encodeURIComponent(id)}`;
    dom.currentModelClose.dataset.jobId = id;
    dom.currentModelClose.disabled = !service.healthy || station.active_tasks?.length > 0;
    dom.currentModelClose.title = station.active_tasks?.length ? "先在机器人控制台结束当前任务" : "停止当前模型服务并释放资源";
  } catch (error) {
    dom.currentModelClose.disabled = true;
    if (!isAbortLike(error)) dom.currentModelSummary.textContent = `当前模型状态无法读取：${error.message}`;
  }
}

async function loadBootstrap() {
  state.bootstrapping = true;
  setActionsEnabled();
  setConnection("loading", "连接中");
  dom.loadError.hidden = true;
  dom.retryBootstrap.disabled = true;
  try {
    const bootstrap = await apiFetch("/bootstrap");
    state.bootstrap = {
      ...bootstrap,
      models: Array.isArray(bootstrap?.models) ? bootstrap.models : [],
      robots: Array.isArray(bootstrap?.robots) ? bootstrap.robots : [],
      effector_profiles: Array.isArray(bootstrap?.effector_profiles) ? bootstrap.effector_profiles : [],
      constraints: bootstrap?.constraints || {},
      gpu_recommendations: bootstrap?.gpu_recommendations || {},
    };
    state.csrfToken = bootstrap?.csrf_token || "";
    state.defaults = cloneDefaults(bootstrap?.defaults || {});
    const projectProtocol = typeof bootstrap?.project === "object" ? bootstrap.project?.protocol : "";
    dom.projectProtocol.textContent = projectProtocol || "未声明";
    dom.projectProtocol.title = projectProtocol || "";
    configureAlignmentLink(bootstrap, () => ({
      robotHost: dom.robotHost.value.trim(),
      robotId: radioValue("robot_id"),
    }));
    const maxTaskLength = taskMaxLength();
    dom.taskInput.maxLength = maxTaskLength;
    dom.taskLimit.textContent = String(maxTaskLength);
    updateActionScope();
    renderEffectorProfiles(state.bootstrap.effector_profiles);
    setRadioValue("effector_profile", state.defaults.effector_profile || "dahuan");
    renderRobots(state.bootstrap.robots);
    renderModels(state.bootstrap.models);
    state.loaded = true;
    applyDefaults();
    await refreshCurrentModel();
    applyEffectorProfileDefaults();
    renderGpus(Array.isArray(state.bootstrap?.gpus) ? state.bootstrap.gpus : []);
    await refreshGpuInventory({ silent: true });
    state.bootstrapping = false;
    setConnection("", "控制服务正常");
    setActionsEnabled();
    await probeRobotEndpoint({ automatic: true, silent: true });
    scheduleValidation();
  } catch (error) {
    state.bootstrapping = false;
    if (isAbortLike(error)) {
      setConnection("loading", "正在重连");
      if (!state.destroyed) {
        window.clearTimeout(state.bootstrapRetryTimer);
        state.bootstrapRetryTimer = window.setTimeout(loadBootstrap, 700);
      }
      return;
    }
    state.loaded = false;
    dom.loadError.hidden = false;
    dom.loadErrorMessage.textContent = error.message;
    dom.projectProtocol.textContent = "载入失败";
    setConnection("error", "服务不可用");
    setActionsEnabled();
  } finally {
    dom.retryBootstrap.disabled = false;
  }
}

dom.currentModelClose.addEventListener("click", async () => {
  const id = dom.currentModelClose.dataset.jobId;
  if (!id || !window.confirm("确认关闭当前模型服务？正在执行的任务不能关闭模型。")) return;
  dom.currentModelClose.disabled = true;
  try {
    await apiFetch(`/deployments/${encodeURIComponent(id)}/close-model`, {
      method: "POST", headers: { "X-CSRF-Token": state.csrfToken }, body: JSON.stringify({}),
    });
    toast("当前模型服务已关闭", "success");
  } catch (error) {
    if (!isAbortLike(error)) toast(`关闭模型失败：${error.message}`, "error");
  } finally {
    await refreshCurrentModel();
  }
});

async function refreshGpuInventory({ silent = false } = {}) {
  if (!state.loaded || state.activeJob) return;
  setButtonLoading(dom.refreshGpus, true);
  try {
    const result = await apiFetch("/gpus");
    state.bootstrap.gpus = Array.isArray(result?.gpus) ? result.gpus : [];
    state.bootstrap.gpu_recommendations = result?.recommendations || {};
    const currentModel = radioValue("model_id");
    const currentModelVersion = dom.modelVersion.value;
    renderModels(state.bootstrap.models);
    if (currentModel) setRadioValue("model_id", currentModel);
    renderModelVersions({ preferred: currentModelVersion });
    renderGpus(state.bootstrap.gpus);
    updateDependentControls();
    updateSummary();
    if (!silent) toast("GPU 状态已刷新");
  } catch (error) {
    if (isAbortLike(error)) return;
    if (!silent) toast(`GPU 状态刷新失败：${error.message}`, "error");
  } finally {
    setButtonLoading(dom.refreshGpus, false);
  }
}

function onFormChange(event) {
  if (!state.loaded || state.activeJob) return;
  const target = event.target;
  if (target === dom.durationInput) updateDurationDisplay();
  if (target.name === "robot_id") applySelectedRobotConnection({ force: true });
  if (target.name === "effector_profile") {
    const currentModel = radioValue("model_id");
    applyEffectorProfileDefaults({ updateTask: true });
    renderModels(state.bootstrap.models);
    if (currentModel) setRadioValue("model_id", currentModel);
  }
  if (target === dom.robotHost || target === dom.robotSshPort) resetRobotProbe();
  if (target === dom.gpuSelect) {
    syncGpuCardSelection();
    updateGpuSelectionHelp();
  }
  if (target.name === "model_id" || target.name === "robot_id" || target.name === "effector_profile" || target.name === "inference_mode" || target.name === "server_mode" || target === dom.modelVersion || target === dom.hitl || target === dom.workspaceSafety) {
    if (target.name === "model_id") {
      renderGpus(Array.isArray(state.bootstrap?.gpus) ? state.bootstrap.gpus : []);
    }
    updateDependentControls();
    if (["robot_id", "effector_profile", "model_id", "inference_mode"].includes(target.name)) {
      applySmartServerMode();
    }
  }
  if (target === dom.modelVersion) {
    const selected = selectedModelVersion();
    dom.modelVersionDescription.textContent = selected?.description
      || "版本来自部署清单白名单，任务创建后不可修改。";
    applySmartServerMode();
  }
  updateSummary();
  markDirty(true);
}

dom.form.addEventListener("submit", (event) => {
  event.preventDefault();
  createDeployment();
});
dom.form.addEventListener("input", onFormChange);
dom.form.addEventListener("change", onFormChange);
dom.validateButton.addEventListener("click", () => validateConfiguration({ silent: false }));
dom.retryBootstrap.addEventListener("click", loadBootstrap);
dom.refreshGpus.addEventListener("click", refreshGpuInventory);
dom.checkRobotEndpoint.addEventListener("click", () => {
  state.cleanupPromptedSessionId = "";
  probeRobotEndpoint();
});
dom.openSessionCleanup.addEventListener("click", () => {
  if (!state.robotProbe) {
    toast("请先重新检测机器人实例。", "warning");
    return;
  }
  openCleanupDialog(state.robotProbe, { force: true });
});
dom.cleanupClose.addEventListener("click", closeCleanupDialog);
dom.cleanupCancelOne.addEventListener("click", closeCleanupDialog);
dom.cleanupReview.addEventListener("click", prepareSessionCleanup);
dom.cleanupBack.addEventListener("click", () => {
  if (state.cleanupBusy) return;
  state.cleanupChallenge = null;
  setCleanupStage(1);
});
dom.cleanupConfirmationInput.addEventListener("input", () => {
  dom.cleanupConfirm.disabled = state.cleanupBusy
    || dom.cleanupConfirmationInput.value !== state.cleanupChallenge?.confirmation_phrase;
});
dom.cleanupConfirm.addEventListener("click", confirmSessionCleanup);
dom.cleanupDialog.addEventListener("cancel", (event) => {
  if (state.cleanupBusy) {
    event.preventDefault();
    return;
  }
  closeCleanupDialog();
});
dom.resetParameters.addEventListener("click", () => {
  const currentRobot = radioValue("robot_id");
  const currentModel = radioValue("model_id");
  const currentEffectorProfile = radioValue("effector_profile");
  const currentModelVersion = dom.modelVersion.value;
  applyDefaults({ preserveTargets: true });
  if (currentRobot) setRadioValue("robot_id", currentRobot);
  if (currentModel) setRadioValue("model_id", currentModel);
  if (currentEffectorProfile) setRadioValue("effector_profile", currentEffectorProfile);
  renderModelVersions({ preferred: currentModelVersion });
  applyEffectorProfileDefaults({ updateTask: true });
  updateDependentControls();
  updateSummary();
  markDirty(true);
  toast("运行参数已恢复为项目默认值");
});
window.addEventListener("unhandledrejection", (event) => {
  if (isAbortLike(event.reason)) event.preventDefault();
});
window.addEventListener("error", (event) => {
  if (isAbortLike(event.error || event.message)) event.preventDefault();
});
window.addEventListener("pagehide", () => {
  state.destroyed = true;
  window.clearTimeout(state.bootstrapRetryTimer);
  stopPolling();
});
window.addEventListener("pageshow", (event) => {
  if (!event.persisted) return;
  state.destroyed = false;
  if (state.activeJob?.id) startPolling(state.activeJob.id);
});

loadBootstrap();
