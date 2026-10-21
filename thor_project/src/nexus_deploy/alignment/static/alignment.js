"use strict";

const state = {
  bootstrap: null,
  csrf: "",
  session: null,
  token: "",
  viewerToken: "",
  socket: null,
  socketSessionId: null,
  frameImage: null,
  frameObjectUrl: "",
  fetchingFrame: false,
  lastFrameSequence: null,
  lastDetectionMethod: "",
  lastDetectionAt: 0,
  frameAspectRatioSet: false,
  starting: false,
  destroyed: false,
  reconnectTimer: null,
  renderTimer: null,
  lastRenderAt: -Infinity,
  frameRequest: null,
  cameraVisible: true,
  checksKey: "",
  guidanceKey: "",
  drawKey: "",
};

const $ = (selector) => document.querySelector(selector);
const dom = {
  serviceDot: $("#service-dot"), serviceLabel: $("#service-label"),
  connectView: $("#connect-view"), workspaceView: $("#workspace-view"), connectForm: $("#connect-form"),
  robotHost: $("#robot-host"), profile: $("#profile-select"), adapter: $("#adapter-select"),
  adapterPort: $("#adapter-port"),
  connectButton: $("#connect-button"), connectError: $("#connect-error"), robotTitle: $("#robot-title"),
  message: $("#session-message"), preflight: $("#preflight-panel"), checkGrid: $("#check-grid"),
  start: $("#start-button"), recheck: $("#recheck-button"), disconnect: $("#disconnect-button"),
  alignmentPanel: $("#alignment-panel"), canvas: $("#alignment-canvas"), cameraStage: $("#camera-stage"), cameraEmpty: $("#camera-empty"),
  stale: $("#stale-cover"), cameraName: $("#camera-name"), freshness: $("#camera-freshness"), alignmentState: $("#alignment-state"),
  tagDetection: $("#tag-detection"), tagDetectionMethod: $("#tag-detection-method"), tagDetectionDetail: $("#tag-detection-detail"),
  forward: $("#forward-value"), lateral: $("#lateral-value"), yaw: $("#yaw-value"), guidance: $("#guidance-list"),
  topTable: $("#top-table"), complete: $("#complete-button"), stop: $("#stop-button"),
  done: $("#done-panel"), doneCopy: $("#done-copy"), newSession: $("#new-session-button"),
  targetVersion: $("#target-version"), setStandard: $("#set-standard-button"),
  surfaceStatus: $("#surface-status"), safetyProfileStatus: $("#safety-profile-status"),
  targetAdminHint: $("#target-admin-hint"), targetDialog: $("#target-dialog"),
  targetForm: $("#target-form"), targetDialogClose: $("#target-dialog-close"),
  targetCancel: $("#target-cancel"), targetConfirm: $("#target-confirm"),
  adminPassword: $("#admin-password"), targetError: $("#target-error"),
  targetRobot: $("#target-robot"), targetTag: $("#target-tag"),
  targetPose: $("#target-pose"), targetCalibration: $("#target-calibration"),
};

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("Accept", "application/json");
  if (options.body !== undefined) headers.set("Content-Type", "application/json");
  if (options.method && options.method !== "GET") headers.set("X-CSRF-Token", state.csrf);
  if (state.token) headers.set("X-Alignment-Control-Token", state.token);
  const response = await fetch(`/api${path}`, { ...options, headers });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `请求失败 (${response.status})`);
  return body;
}

function option(select, value, label) {
  const item = document.createElement("option");
  item.value = value; item.textContent = label; select.append(item);
}

async function bootstrap() {
  try {
    const value = await api("/bootstrap");
    state.bootstrap = value; state.csrf = value.csrf_token || "";
    value.profiles.forEach((item) => {
      const artifacts = item.artifacts || {};
      const target = item.target_configured ? `目标 v${item.version}` : "目标未配置";
      const surface = artifacts.surface?.configured ? "桌面已标定" : "桌面未标定";
      const safety = artifacts.safety_profile?.generated ? "安全配置已生成" : "安全配置未生成";
      option(dom.profile, item.id, `${item.display_name} · ${target} · ${surface} · ${safety}`);
    });
    value.adapters.forEach((item) => option(dom.adapter, item.id, item.display_name));
    dom.profile.value = value.defaults.profile_id;
    dom.adapter.value = value.defaults.adapter_id;
    const defaultAdapter = value.adapters.find((item) => item.id === value.defaults.adapter_id);
    dom.adapterPort.value = defaultAdapter?.default_port || "";
    dom.adapterPort.placeholder = defaultAdapter?.default_port ? `默认 ${defaultAdapter.default_port}` : "使用适配器默认值";
    const query = new URLSearchParams(location.search);
    dom.robotHost.value = query.get("host") || value.defaults.host || "";
    dom.serviceDot.classList.add("ok"); dom.serviceLabel.textContent = "对齐服务正常";
  } catch (error) {
    dom.serviceLabel.textContent = "对齐服务不可用"; showError(error.message);
  }
}

function showError(message) {
  dom.connectError.textContent = message; dom.connectError.hidden = false;
}

function setBusy(button, busy, label) {
  button.disabled = busy;
  if (label) button.querySelector("span") ? button.querySelector("span").textContent = label : button.textContent = label;
}

dom.connectForm.addEventListener("submit", async (event) => {
  event.preventDefault(); dom.connectError.hidden = true; setBusy(dom.connectButton, true, "正在连接…");
  try {
    const session = await api("/sessions", { method: "POST", body: JSON.stringify({
      host: dom.robotHost.value.trim(), profile_id: dom.profile.value, adapter_id: dom.adapter.value,
      port: dom.adapterPort.value ? Number(dom.adapterPort.value) : null, adjustment_mode: "table",
    }) });
    state.token = session.control_token; state.viewerToken = session.viewer_token;
    delete session.control_token; delete session.viewer_token; state.session = session;
    dom.connectView.hidden = true; dom.workspaceView.hidden = false; render(session); connectEvents();
  } catch (error) { showError(error.message); }
  finally { setBusy(dom.connectButton, false, "连接并检查"); }
});

dom.adapter.addEventListener("change", () => {
  const item = state.bootstrap?.adapters?.find((value) => value.id === dom.adapter.value);
  dom.adapterPort.value = item?.default_port || "";
  dom.adapterPort.placeholder = item?.default_port ? `默认 ${item.default_port}` : "使用适配器默认值";
});

function render(session) {
  state.session = session;
  dom.robotTitle.textContent = session.robot?.model || session.robot?.id || `机器人 ${session.host}`;
  dom.message.textContent = session.message;
  renderChecks(session.checks || []);
  const active = ["SEARCHING", "UNSTABLE", "ADJUSTING", "ALIGNED", "CONNECTION_LOST"].includes(session.state);
  dom.preflight.hidden = active || session.state === "COMPLETED";
  dom.alignmentPanel.hidden = !active;
  dom.done.hidden = session.state !== "COMPLETED";
  dom.start.hidden = session.state !== "READY";
  dom.start.disabled = state.starting || session.state !== "READY";
  dom.recheck.hidden = !["READY", "ERROR"].includes(session.state);
  renderTargetAdmin(session);
  if (active) renderAlignment(session);
  if (session.state === "COMPLETED") dom.doneCopy.textContent = "检查结果、配置版本和标注截图已经保存。";
}

function renderTargetAdmin(session) {
  const profile = session.profile || {};
  const configuredProfile = state.bootstrap?.profiles?.find((item) => item.id === profile.id) || {};
  const artifacts = configuredProfile.artifacts || {};
  const enabled = Boolean(state.bootstrap?.service?.target_admin_enabled);
  const surfaceStableFor = Number(session.surface_stable_s || 0);
  const freshState = ["ADJUSTING", "ALIGNED"].includes(session.state);
  const stable = Boolean(session.result?.stable) && freshState;
  const standardDetection = session.result?.detection?.method !== "fallback";
  const strictStable = session.result?.surface?.eligible === true && surfaceStableFor >= 2;
  dom.targetVersion.textContent = profile.target_configured ? `v${profile.version}` : "尚未配置";
  dom.surfaceStatus.textContent = artifacts.surface?.configured
    ? `${Number(artifacts.surface.height_m).toFixed(3)} m`
    : "尚未标定";
  dom.safetyProfileStatus.textContent = artifacts.safety_profile?.generated
    ? "已生成"
    : "尚未生成";
  dom.setStandard.disabled = !(enabled && stable && standardDetection && strictStable);
  if (!enabled) dom.targetAdminHint.textContent = "服务端尚未配置管理员密码";
  else if (!standardDetection) dom.targetAdminHint.textContent = "当前为容错识别，不能保存标准位置";
  else if (!stable) dom.targetAdminHint.textContent = "AprilTag 稳定识别 2 秒后可设置";
  else if (!session.result?.surface?.eligible) dom.targetAdminHint.textContent = "位置显示已稳定，严格标定测量仍有波动";
  else if (surfaceStableFor < 2) dom.targetAdminHint.textContent = `严格标定测量还需保持 ${(2 - surfaceStableFor).toFixed(1)} 秒`;
  else dom.targetAdminHint.textContent = "检测稳定，可以保存为新的标准位置";
}

function renderChecks(checks) {
  const key = JSON.stringify(checks);
  if (state.checksKey === key) return;
  state.checksKey = key;
  dom.checkGrid.replaceChildren();
  checks.forEach((check) => {
    const article = document.createElement("article"); article.className = `check-item${check.ok ? "" : " bad"}`;
    const mark = document.createElement("i"); mark.textContent = check.ok ? "✓" : "!";
    const copy = document.createElement("div"); const title = document.createElement("strong"); const detail = document.createElement("small");
    title.textContent = check.label; detail.textContent = check.detail; copy.append(title, detail); article.append(mark, copy); dom.checkGrid.append(article);
  });
}

dom.start.addEventListener("click", async () => {
  if (state.starting || state.session?.state !== "READY") return;
  state.starting = true;
  dom.start.disabled = true;
  try {
    const session = await api(`/sessions/${state.session.id}/start`, { method: "POST", body: "{}" });
    state.starting = false;
    render(session);
  } catch (error) {
    state.starting = false;
    dom.start.disabled = state.session?.state !== "READY";
    alert(error.message);
  }
});

dom.recheck.addEventListener("click", async () => {
  dom.recheck.disabled = true;
  try { render(await api(`/sessions/${state.session.id}/recheck`, { method: "POST", body: "{}" })); }
  catch (error) { alert(error.message); }
  finally { dom.recheck.disabled = false; }
});

function connectEvents() {
  if (state.destroyed || !state.session || ["COMPLETED", "STOPPED"].includes(state.session.state)) return;
  if (state.socket && state.socketSessionId === state.session.id
      && [WebSocket.CONNECTING, WebSocket.OPEN].includes(state.socket.readyState)) return;
  disconnectEvents();
  const sessionId = state.session.id;
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${location.host}/api/sessions/${sessionId}/events?token=${encodeURIComponent(state.viewerToken)}`);
  state.socket = socket;
  state.socketSessionId = sessionId;
  socket.onmessage = (event) => {
    if (state.destroyed || state.socket !== socket || state.session?.id !== sessionId) return;
    state.session = JSON.parse(event.data);
    scheduleAlignmentRender();
  };
  socket.onclose = () => {
    if (state.destroyed || state.socket !== socket || state.session?.id !== sessionId) return;
    state.socket = null;
    if (!["COMPLETED", "STOPPED"].includes(state.session.state)) {
      state.reconnectTimer = window.setTimeout(() => {
        state.reconnectTimer = null;
        if (state.session?.id === sessionId) connectEvents();
      }, 1000);
    }
  };
}

function disconnectEvents() {
  window.clearTimeout(state.reconnectTimer);
  state.reconnectTimer = null;
  const socket = state.socket;
  state.socket = null;
  state.socketSessionId = null;
  if (socket) {
    socket.onmessage = socket.onclose = socket.onerror = null;
    socket.close();
  }
}

function scheduleAlignmentRender() {
  if (state.destroyed || document.hidden || !state.session || state.renderTimer !== null) return;
  state.renderTimer = window.setTimeout(() => {
    state.renderTimer = null;
    if (state.destroyed || document.hidden || !state.session) return;
    state.lastRenderAt = performance.now();
    render(state.session);
    refreshFrame(state.session);
  }, Math.max(0, 250 - (performance.now() - state.lastRenderAt)));
}

function finishFrameRequest(request, keepImage = false) {
  window.clearTimeout(request.timer);
  if (request.image) {
    request.image.onload = request.image.onerror = null;
    if (!keepImage) request.image.src = "";
  }
  if (request.url && !keepImage) URL.revokeObjectURL(request.url);
  if (state.frameRequest === request) {
    state.frameRequest = null;
    state.fetchingFrame = false;
  }
}

function cancelFrameRequest() {
  const request = state.frameRequest;
  if (!request) return;
  request.controller.abort();
  finishFrameRequest(request);
}

async function refreshFrame(session) {
  if (state.destroyed || document.hidden || !state.cameraVisible || !session.frame_sequence
      || session.frame_sequence === state.lastFrameSequence || state.fetchingFrame) return;
  state.fetchingFrame = true;
  const sequence = session.frame_sequence;
  const request = { controller: new AbortController(), image: null, url: "", timer: null };
  state.frameRequest = request;
  const current = () => state.frameRequest === request && !state.destroyed && !document.hidden
    && state.cameraVisible && state.session?.id === session.id && !request.controller.signal.aborted;
  request.timer = window.setTimeout(() => {
    request.controller.abort(); finishFrameRequest(request);
  }, 6000);
  try {
    const response = await fetch(`/api/sessions/${session.id}/frame?v=${sequence}`,
      { cache: "no-store", signal: request.controller.signal });
    if (!response.ok || !current()) { finishFrameRequest(request); return; }
    const blob = await response.blob(); const url = URL.createObjectURL(blob); const image = new Image();
    request.url = url; request.image = image;
    if (!current()) { finishFrameRequest(request); return; }
    image.onload = () => {
      if (!current() || (state.lastFrameSequence !== null && sequence < state.lastFrameSequence)) {
        finishFrameRequest(request); return;
      }
      if (state.frameObjectUrl) URL.revokeObjectURL(state.frameObjectUrl);
      state.frameObjectUrl = url; state.frameImage = image; state.lastFrameSequence = sequence;
      if (!state.frameAspectRatioSet && image.naturalWidth > 0 && image.naturalHeight > 0) {
        dom.cameraStage.style.aspectRatio = `${image.naturalWidth} / ${image.naturalHeight}`;
        state.frameAspectRatioSet = true;
      }
      dom.cameraEmpty.hidden = true; draw(state.session || session);
      dom.freshness.textContent = `${image.naturalWidth}×${image.naturalHeight} · 帧 ${sequence}`;
      finishFrameRequest(request, true);
    };
    image.onerror = () => finishFrameRequest(request);
    image.src = url;
  } catch (_) {
    finishFrameRequest(request);
  }
}

function draw(session) {
  if (state.destroyed || document.hidden || !state.cameraVisible) return;
  const image = state.frameImage; if (!image) return;
  const result = session.result || {};
  const drawKey = JSON.stringify([state.frameObjectUrl, session.target_corners || result.target_corners,
    result.actual_corners, result.stale]);
  if (drawKey === state.drawKey) return;
  state.drawKey = drawKey;
  const canvas = dom.canvas; const width = image.naturalWidth || session.frame_width || 1280; const height = image.naturalHeight || session.frame_height || 720;
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const context = canvas.getContext("2d"); context.drawImage(image, 0, 0, width, height);
  const targetCorners = session.target_corners || result.target_corners;
  const targetOutline = expandPolygon(targetCorners, 10);
  polygon(context, targetOutline, "#ff5bd6", true, 6);
  polygon(context, result.actual_corners, result.stale ? "#99a59f" : "#5ff0a9", false, 5);
}

function expandPolygon(points, padding) {
  if (!Array.isArray(points) || points.length !== 4) return points;
  const origin = center(points);
  return points.map((point) => {
    const dx = point[0] - origin[0], dy = point[1] - origin[1];
    const length = Math.hypot(dx, dy) || 1;
    return [point[0] + padding * dx / length, point[1] + padding * dy / length];
  });
}

function polygon(context, points, color, dashed, width) {
  if (!Array.isArray(points) || points.length !== 4) return;
  context.save(); context.strokeStyle = color; context.lineWidth = width; context.setLineDash(dashed ? [12, 8] : []);
  context.shadowColor = color; context.shadowBlur = 8; context.beginPath(); context.moveTo(points[0][0], points[0][1]);
  points.slice(1).forEach((point) => context.lineTo(point[0], point[1])); context.closePath(); context.stroke(); context.restore();
}
function center(points) { return points.reduce((value, point) => [value[0] + point[0] / 4, value[1] + point[1] / 4], [0, 0]); }

function renderAlignment(session) {
  const names = {SEARCHING:"正在寻找 AprilTag",UNSTABLE:"检测尚未稳定",ADJUSTING:"请按提示调整",ALIGNED:"已稳定对齐",CONNECTION_LOST:"连接中断"};
  const cameraNames = {cam_head:"头部彩色相机",head_stereo_left:"头部左双目相机",head_stereo_right:"头部右双目相机"};
  const cameraId = session.profile?.camera_name || "";
  dom.cameraName.textContent = cameraNames[cameraId] || cameraId || "头部相机";
  dom.alignmentState.className = `alignment-state ${session.state === "ALIGNED" ? "aligned" : session.state === "ADJUSTING" ? "adjusting" : session.state === "CONNECTION_LOST" ? "error" : "searching"}`;
  const surfaceStableFor = Number(session.surface_stable_s || 0);
  let stateLabel = session.profile?.target_configured ? (names[session.state] || session.message) : session.message;
  if (session.state === "ALIGNED" && session.result?.surface?.eligible !== true) {
    stateLabel = "位置已对齐，安全测量待稳定";
  } else if (session.state === "ALIGNED" && surfaceStableFor < 2) {
    stateLabel = `位置已对齐，安全测量还需 ${(2 - surfaceStableFor).toFixed(1)} 秒`;
  } else if (session.state === "ALIGNED") {
    stateLabel = "已稳定对齐，可以完成";
  }
  dom.alignmentState.querySelector("strong").textContent = stateLabel;
  dom.stale.hidden = session.state !== "CONNECTION_LOST";
  const result = session.result || {}, error = result.error || {}, guidance = result.guidance || {};
  renderDetectionMethod(result);
  dom.forward.textContent = number(error.forward_mm); dom.lateral.textContent = number(error.lateral_mm); dom.yaw.textContent = number(error.yaw_deg);
  const steps = guidance.steps || [session.message];
  const guidanceKey = JSON.stringify(steps);
  if (guidanceKey !== state.guidanceKey) {
    state.guidanceKey = guidanceKey;
    dom.guidance.replaceChildren(); steps.forEach((text) => { const item=document.createElement("li");item.textContent=text;dom.guidance.append(item); });
  }
  dom.complete.disabled = session.state !== "ALIGNED" || result.surface?.eligible !== true || surfaceStableFor < 2;
  dom.complete.title = result.surface?.reason === "fallback_detection_not_allowed"
    ? "当前使用容错识别；请更换具有完整白色静区、无遮挡的 AprilTag"
    : result.surface?.eligible === false
      ? "桌面高度或倾角尚未达到安全标定要求"
      : surfaceStableFor < 2
        ? "严格安全测量需要连续稳定保持 2 秒"
      : "完成对齐并保存桌面安全高度";
  const x = Math.max(-35,Math.min(35,Number(error.lateral_mm||0)*.35)); const y = Math.max(-24,Math.min(24,-Number(error.forward_mm||0)*.25));
  dom.topTable.setAttribute("transform", `translate(${x} ${y}) rotate(${Number(error.yaw_deg||0)} 150 44)`);
  // A newer frame is about to be loaded; draw its overlay together on load,
  // rather than first repainting the previous full-resolution image.
  if (session.frame_sequence === state.lastFrameSequence) draw(session);
}

function renderDetectionMethod(result) {
  const observed = result.detection?.method || "";
  if (observed) {
    state.lastDetectionMethod = observed;
    state.lastDetectionAt = Date.now();
  }
  const recent = Date.now() - state.lastDetectionAt < 2000;
  const method = observed || (recent ? state.lastDetectionMethod : "");
  if (method === "fallback") {
    const relaxedMeasurement = state.session?.profile?.surface?.allow_hamming0_fallback_measurement === true
      && Number(result.detection?.fallback_hamming) === 0;
    dom.tagDetection.className = "tag-detection fallback";
    dom.tagDetectionMethod.textContent = relaxedMeasurement ? "当前为宽松标定识别" : "当前为容错识别";
    dom.tagDetectionDetail.textContent = relaxedMeasurement
      ? "标签解码完全一致（Hamming 0）；本次允许生成实测配置，但不会自动取得真机执行资格。"
      : observed
        ? "请检查标签白色静区和遮挡；该容错结果不能生成桌面安全配置。"
        : "标签短暂丢失，暂时保留上一识别方式；该结果不能生成安全配置。";
  } else if (method === "standard") {
    dom.tagDetection.className = "tag-detection standard";
    dom.tagDetectionMethod.textContent = "当前为标准识别";
    dom.tagDetectionDetail.textContent = observed
      ? "标签边界完整，正在持续评估位置与安全测量稳定性。"
      : "标签短暂丢失，暂时保留上一识别方式并等待恢复。";
  } else {
    dom.tagDetection.className = "tag-detection waiting";
    dom.tagDetectionMethod.textContent = "等待标签识别";
    dom.tagDetectionDetail.textContent = "请确保 AprilTag 完整出现在画面中。";
  }
}

function number(value) { return Number.isFinite(Number(value)) ? (Number(value) > 0 ? "+" : "") + Number(value).toFixed(1) : "—"; }

document.querySelectorAll("[data-mode]").forEach((button) => button.addEventListener("click", async () => {
  try {
    const value = await api(`/sessions/${state.session.id}/mode`, {method:"POST",body:JSON.stringify({adjustment_mode:button.dataset.mode})});
    document.querySelectorAll("[data-mode]").forEach((item)=>item.classList.toggle("active",item===button)); render(value);
  } catch (error) { alert(error.message); }
}));

dom.complete.addEventListener("click", async () => {
  dom.complete.disabled = true;
  try {
    const overlay = dom.canvas.toDataURL("image/png");
    const report = await api(`/sessions/${state.session.id}/complete`, {method:"POST",body:JSON.stringify({overlay_data_url:overlay})});
    const published = report.surface_calibration || {};
    const configuredProfile = state.bootstrap?.profiles?.find((item) => item.id === state.session.profile?.id);
    if (configuredProfile && published.measurement_id) {
      configuredProfile.artifacts = {
        ...(configuredProfile.artifacts || {}),
        surface: {
          configured: true,
          measurement_id: published.measurement_id,
          height_m: published.surface?.height_m,
        },
        safety_profile: {
          generated: published.safety_profile?.generated === true,
          filename: published.safety_profile?.id || null,
        },
      };
    }
    const snapshot = {...state.session,state:"COMPLETED",message:"对齐已完成并保存"}; render(snapshot); if(state.socket)state.socket.close();
    dom.doneCopy.textContent = published.safety_profile?.generated === true
      ? "桌面高度已保存，对应安全配置已经生成。"
      : `桌面高度已保存，但安全配置未生成${published.safety_profile?.error ? `：${published.safety_profile.error}` : "。"}`;
  } catch (error) { alert(error.message); dom.complete.disabled=false; }
});

dom.setStandard.addEventListener("click", () => {
  const session = state.session;
  if (!session || dom.setStandard.disabled) return;
  const pose = session.result?.pose || {};
  dom.targetRobot.textContent = `${session.robot?.id || session.robot?.model || "未知"} · ${session.host}`;
  dom.targetTag.textContent = `${session.profile?.tag?.family || "—"} · ID ${session.result?.tag_id ?? (session.profile?.tag?.ids || []).join(", ")}`;
  dom.targetPose.textContent = `x ${formatPose(pose.x_m)} / y ${formatPose(pose.y_m)} / z ${formatPose(pose.z_m)} m · yaw ${formatPose(pose.yaw_deg)}°`;
  dom.targetCalibration.textContent = session.calibration_version || "—";
  dom.targetError.hidden = true;
  dom.targetError.textContent = "";
  dom.adminPassword.value = "";
  dom.targetDialog.showModal();
  dom.adminPassword.focus();
});

function closeTargetDialog() {
  dom.adminPassword.value = "";
  dom.targetDialog.close();
}

dom.targetDialogClose.addEventListener("click", closeTargetDialog);
dom.targetCancel.addEventListener("click", closeTargetDialog);
dom.targetDialog.addEventListener("click", (event) => {
  if (event.target === dom.targetDialog) closeTargetDialog();
});

dom.targetForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  dom.targetError.hidden = true;
  setBusy(dom.targetConfirm, true, "正在保存…");
  try {
    const value = await api(`/sessions/${state.session.id}/target`, {
      method: "POST",
      body: JSON.stringify({ admin_password: dom.adminPassword.value }),
    });
    dom.adminPassword.value = "";
    dom.targetDialog.close();
    render(value.session);
    draw(value.session);
  } catch (error) {
    dom.targetError.textContent = error.message;
    dom.targetError.hidden = false;
    dom.adminPassword.select();
  } finally {
    setBusy(dom.targetConfirm, false, "确认保存并启用");
  }
});

function formatPose(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : "—";
}

dom.stop.addEventListener("click", async () => {
  if (!confirm("结束本次对齐并记录为未完成？")) return;
  try { await api(`/sessions/${state.session.id}/stop`, {method:"POST",body:JSON.stringify({reason:"user_cancelled"})}); reset(); }
  catch (error) { alert(error.message); }
});
dom.disconnect.addEventListener("click", async () => {
  if (state.session && !["COMPLETED","STOPPED"].includes(state.session.state)) {
    try { await api(`/sessions/${state.session.id}/stop`, {method:"POST",body:JSON.stringify({reason:"connection_closed"})}); } catch (_) { /* best effort */ }
  } reset();
});
dom.newSession.addEventListener("click", reset);
function reset(){disconnectEvents();cancelFrameRequest();window.clearTimeout(state.renderTimer);state.renderTimer=null;if(state.frameObjectUrl)URL.revokeObjectURL(state.frameObjectUrl);state.frameObjectUrl="";if(dom.targetDialog.open)dom.targetDialog.close();state.socket=null;state.session=null;state.token="";state.viewerToken="";state.frameImage=null;state.lastFrameSequence=null;state.lastDetectionMethod="";state.lastDetectionAt=0;state.frameAspectRatioSet=false;state.starting=false;dom.start.disabled=false;dom.cameraStage.style.aspectRatio="";dom.workspaceView.hidden=true;dom.connectView.hidden=false;dom.alignmentPanel.hidden=true;dom.done.hidden=true;}

// Keep the one event connection alive while hidden so viewing another tab
// does NOT end alignment or release its session. Only presentation sleeps.
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    window.clearTimeout(state.renderTimer); state.renderTimer = null;
    cancelFrameRequest();
  } else scheduleAlignmentRender();
});
const cameraObserver = typeof IntersectionObserver === "function" ? new IntersectionObserver((entries) => {
  state.cameraVisible = entries[0].isIntersecting;
  if (!state.cameraVisible) cancelFrameRequest();
  else scheduleAlignmentRender();
}) : null;
cameraObserver?.observe(dom.cameraStage);
window.addEventListener("pagehide", () => {
  state.destroyed = true;
  disconnectEvents(); cancelFrameRequest();
  window.clearTimeout(state.renderTimer); state.renderTimer = null;
  if (state.frameObjectUrl) URL.revokeObjectURL(state.frameObjectUrl);
  state.frameObjectUrl = ""; state.frameImage = null; state.lastFrameSequence = null;
  cameraObserver?.disconnect();
});
window.addEventListener("pageshow", () => {
  state.destroyed = false;
  cameraObserver?.observe(dom.cameraStage);
  connectEvents(); scheduleAlignmentRender();
});

bootstrap();
