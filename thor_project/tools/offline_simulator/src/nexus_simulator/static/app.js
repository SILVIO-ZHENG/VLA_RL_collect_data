const state = {
  bootstrap: null,
  jointTimer: null,
  surfaceTimer: null,
  toastTimer: null,
  replayDrainPromise: null,
  queuedReplayFrame: null,
  replayTimer: null,
  replayTimerKind: null,
  replaySeekTimer: null,
  lastReplayFrame: null,
  cameraView: "near",
};

const pageParams = new URLSearchParams(window.location.search);
if (pageParams.get("record") === "1") document.body.classList.add("is-recording");
if (pageParams.get("view") === "overview") state.cameraView = "overview";

const elements = {
  frame: document.querySelector("#meshcat-frame"),
  loading: document.querySelector("#viewer-loading"),
  viewerStatus: document.querySelector("#viewer-status"),
  cameraViewButtons: [...document.querySelectorAll("[data-camera-view]")],
  surfaceHeight: document.querySelector("#surface-height"),
  surfaceHeightOutput: document.querySelector("#surface-height-output"),
  hardMargin: document.querySelector("#hard-margin"),
  hardMarginOutput: document.querySelector("#hard-margin-output"),
  jointGroups: document.querySelector("#joint-groups"),
  resetButton: document.querySelector("#reset-button"),
  modelName: document.querySelector("#model-name"),
  jointCount: document.querySelector("#joint-count"),
  geometryCount: document.querySelector("#geometry-count"),
  assetStatus: document.querySelector("#asset-status"),
  toast: document.querySelector("#toast"),
  headVideo: document.querySelector("#head-video"),
  episodeChip: document.querySelector("#episode-chip"),
  replayTask: document.querySelector("#replay-task"),
  replayPlay: document.querySelector("#replay-play"),
  replayPause: document.querySelector("#replay-pause"),
  replayReset: document.querySelector("#replay-reset"),
  replaySource: document.querySelector("#replay-source"),
  replayTimeline: document.querySelector("#replay-timeline"),
  replayFrameLabel: document.querySelector("#replay-frame-label"),
  replayTimeLabel: document.querySelector("#replay-time-label"),
  safetyResult: document.querySelector("#safety-result"),
  safetyTitle: document.querySelector("#safety-title"),
  safetyDetail: document.querySelector("#safety-detail"),
  safetyLimit: document.querySelector("#safety-limit"),
  safetyLimitState: document.querySelector("#safety-limit-state"),
  safetyLimitControl: document.querySelector("#safety-limit-control"),
  safetyLimitLabel: document.querySelector("#safety-limit-label"),
  collisionAlert: document.querySelector("#collision-alert"),
  collisionAlertTitle: document.querySelector("#collision-alert-title"),
  collisionAlertDetail: document.querySelector("#collision-alert-detail"),
  viewerPanel: document.querySelector(".viewer-panel"),
};

async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const payload = await response.json();
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || `请求失败：HTTP ${response.status}`);
  }
  return payload.data;
}

function showToast(message, error = false) {
  clearTimeout(state.toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.toggle("is-error", error);
  elements.toast.classList.add("is-visible");
  state.toastTimer = setTimeout(() => elements.toast.classList.remove("is-visible"), 2800);
}

function formatJointValue(value, unit = "rad") {
  return unit === "m"
    ? `${Number(value).toFixed(4)} m`
    : `${Number(value).toFixed(3)} rad`;
}

function formatTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const remainder = (safe - minutes * 60).toFixed(1).padStart(4, "0");
  return `${String(minutes).padStart(2, "0")}:${remainder}`;
}

function groupJoints(joints) {
  return joints.reduce((groups, joint) => {
    if (!groups.has(joint.group)) groups.set(joint.group, []);
    groups.get(joint.group).push(joint);
    return groups;
  }, new Map());
}

function renderJoints(joints) {
  const groups = groupJoints(joints);
  elements.jointGroups.replaceChildren();
  for (const [name, items] of groups.entries()) {
    const details = document.createElement("details");
    details.className = "joint-group";
    details.open = ["左臂", "右臂"].includes(name);

    const summary = document.createElement("summary");
    summary.append(document.createTextNode(name));
    const count = document.createElement("span");
    count.textContent = `${items.length} joints`;
    summary.append(count);
    details.append(summary);

    const list = document.createElement("div");
    list.className = "joint-list";
    for (const joint of items) {
      const row = document.createElement("div");
      row.className = "joint-control";
      const label = document.createElement("label");
      label.htmlFor = `joint-${joint.name}`;
      label.textContent = joint.name;
      label.title = joint.name;
      const output = document.createElement("output");
      output.htmlFor = `joint-${joint.name}`;
      output.textContent = formatJointValue(joint.value, joint.unit);
      const input = document.createElement("input");
      input.id = `joint-${joint.name}`;
      input.dataset.joint = joint.name;
      input.dataset.unit = joint.unit;
      input.type = "range";
      input.min = String(joint.lower);
      input.max = String(joint.upper);
      input.step = joint.unit === "m" ? "0.0005" : "0.005";
      input.value = String(joint.value);
      input.addEventListener("input", () => {
        output.textContent = formatJointValue(input.value, input.dataset.unit);
        scheduleJointUpdate(joint.name, Number(input.value));
      });
      row.append(label, output, input);
      list.append(row);
    }
    details.append(list);
    elements.jointGroups.append(details);
  }
}

function syncJointControls(joints) {
  const byName = new Map(joints.map((joint) => [joint.name, joint]));
  for (const input of document.querySelectorAll("[data-joint]")) {
    const joint = byName.get(input.dataset.joint);
    if (!joint) continue;
    input.value = String(joint.value);
    input.dataset.unit = joint.unit;
    input.parentElement.querySelector("output").textContent = formatJointValue(
      joint.value,
      joint.unit,
    );
  }
}

function renderSafety(result) {
  const collisionDetected = result.collision_detected ?? !result.safe;
  const monitorOnly = collisionDetected && !result.enforcement_enabled;
  const enforcementFailed = collisionDetected
    && result.enforcement_enabled
    && result.enforcement_effective === false;
  const enforcementSucceeded = collisionDetected
    && result.enforcement_enabled
    && result.enforcement_applied
    && result.enforcement_effective;
  elements.safetyResult.className = `safety-result ${collisionDetected ? "is-danger" : "is-safe"}${monitorOnly ? " is-monitor-only" : ""}${enforcementSucceeded ? " is-enforced" : ""}`;
  if (!collisionDetected) {
    elements.safetyTitle.textContent = "当前显示步：安全";
  } else if (monitorOnly) {
    elements.safetyTitle.textContent = "危险：检测到碰撞，安全限制已关闭";
  } else if (enforcementFailed) {
    elements.safetyTitle.textContent = "危险：没有可恢复的安全姿态";
  } else if (result.enforcement_applied) {
    elements.safetyTitle.textContent = "检测到碰撞风险 · 候选动作已拦截";
  } else {
    elements.safetyTitle.textContent = "检测到工作面碰撞风险（仅观测）";
  }
  const clearance = result.clearance == null ? "—" : `${(result.clearance * 1000).toFixed(1)} mm`;
  const disposition = {
    hold_previous_successful_action: "保持上一安全动作",
    hold_measured_pose: "保持当前测量姿态",
    hold_measured_pose_previous_recheck_failed: "上一动作复检失败，保持测量姿态",
    monitor_only_allow_unsafe_candidate: "仅告警，危险候选动作继续通过",
    allow_candidate: "采用候选动作",
    cannot_recover_without_safe_pose: "初始或当前姿态已不安全，无法保持历史安全姿态",
  }[result.action_disposition] || result.action_disposition || "仅检测";
  elements.safetyDetail.textContent = `${disposition} · ${result.reason} · 净余量 ${clearance} · ${result.limiting_frame || "—"} · ${result.latency_ms.toFixed(2)} ms`;
  const prominentAlert = monitorOnly || enforcementFailed || enforcementSucceeded;
  elements.collisionAlert.hidden = !prominentAlert;
  elements.collisionAlert.classList.toggle("is-enforced", enforcementSucceeded);
  if (monitorOnly) {
    elements.collisionAlertTitle.textContent = "碰撞风险 · 安全限制已关闭";
    elements.collisionAlertDetail.textContent = "检测仍在运行，但危险候选姿态不会被拦截";
  } else if (enforcementFailed) {
    elements.collisionAlertTitle.textContent = "碰撞风险 · 无可用安全保持姿态";
    elements.collisionAlertDetail.textContent = "初始或当前姿态已经不安全，请降低工作面或从安全帧重新开始";
  } else if (enforcementSucceeded) {
    elements.collisionAlertTitle.textContent = "安全限制已生效 · 危险候选已拦截";
    elements.collisionAlertDetail.textContent = `${disposition}，三维画面显示的是最终采用的安全姿态`;
  }
  elements.viewerPanel.classList.toggle(
    "has-unrestricted-collision",
    monitorOnly || enforcementFailed,
  );
  elements.viewerPanel.classList.toggle("has-enforced-collision", enforcementSucceeded);
}

function renderSafetyLimit(enabled) {
  elements.safetyLimit.checked = enabled;
  elements.safetyLimitControl.classList.toggle("is-disabled", !enabled);
  elements.safetyLimitLabel.textContent = enabled ? "开启" : "关闭";
  elements.safetyLimitState.textContent = enabled
    ? "开启：危险动作将被拦截"
    : "关闭：仅检测告警，不拦截动作";
}

async function updateSafetyLimit() {
  const desired = elements.safetyLimit.checked;
  elements.safetyLimit.disabled = true;
  try {
    const result = await request("/api/safety-limit", {
      method: "POST",
      body: JSON.stringify({ enabled: desired }),
    });
    state.bootstrap.safety.enforcement_enabled = result.enforcement_enabled;
    renderSafetyLimit(result.enforcement_enabled);
    await requestReplayFrame(Number(elements.replayTimeline.value));
    showToast(result.enforcement_enabled
      ? "碰撞安全限制已开启"
      : "碰撞安全限制已关闭；检测和红色告警仍保持开启");
  } catch (error) {
    renderSafetyLimit(state.bootstrap.safety.enforcement_enabled);
    showToast(error.message, true);
  } finally {
    elements.safetyLimit.disabled = false;
  }
}

function updateReplayLabels(frameIndex) {
  const replay = state.bootstrap.replay;
  const index = Math.max(0, Math.min(replay.frame_count - 1, frameIndex));
  elements.replayTimeline.value = String(index);
  elements.replayFrameLabel.textContent = `${index} / ${replay.frame_count - 1}`;
  elements.replayTimeLabel.textContent = `${formatTime(index / replay.fps)} / ${formatTime(replay.duration_s)}`;
}

function requestReplayFrame(frameIndex) {
  state.queuedReplayFrame = frameIndex;
  if (state.replayDrainPromise) return state.replayDrainPromise;
  state.replayDrainPromise = (async () => {
    try {
      while (state.queuedReplayFrame !== null) {
        const next = state.queuedReplayFrame;
        state.queuedReplayFrame = null;
        const result = await request("/api/replay/frame", {
          method: "POST",
          body: JSON.stringify({ frame_index: next, source: elements.replaySource.value }),
        });
        updateReplayLabels(result.frame_index);
        syncJointControls(result.joints);
        renderSafety(result.safety);
      }
    } catch (error) {
      pauseReplay();
      showToast(error.message, true);
    } finally {
      state.replayDrainPromise = null;
    }
  })();
  return state.replayDrainPromise;
}

function currentVideoFrame(mediaTime = elements.headVideo.currentTime) {
  const replay = state.bootstrap.replay;
  const elapsed = Math.max(0, mediaTime - replay.video.from_s);
  return Math.min(replay.frame_count - 1, Math.floor(elapsed * replay.fps));
}

function replayTick(mediaTime = elements.headVideo.currentTime) {
  const replay = state.bootstrap.replay;
  if (elements.headVideo.paused) return;
  if (mediaTime >= replay.video.to_s) {
    elements.headVideo.currentTime = replay.video.to_s;
    pauseReplay();
    requestReplayFrame(replay.frame_count - 1);
    return;
  }
  const frame = currentVideoFrame(mediaTime);
  updateReplayLabels(frame);
  if (frame !== state.lastReplayFrame) {
    state.lastReplayFrame = frame;
    requestReplayFrame(frame);
  }
}

function waitForMediaEvent(eventName, timeoutMs = 2500) {
  return new Promise((resolve) => {
    let timeout;
    const finish = () => {
      clearTimeout(timeout);
      elements.headVideo.removeEventListener(eventName, finish);
      resolve();
    };
    elements.headVideo.addEventListener(eventName, finish, { once: true });
    timeout = setTimeout(finish, timeoutMs);
  });
}

async function seekVideo(time) {
  if (elements.headVideo.readyState < HTMLMediaElement.HAVE_METADATA) {
    await waitForMediaEvent("loadedmetadata");
  }
  if (Math.abs(elements.headVideo.currentTime - time) < 0.01 && !elements.headVideo.seeking) {
    return;
  }
  const completed = waitForMediaEvent("seeked");
  elements.headVideo.currentTime = time;
  await completed;
}

function startReplayLoop() {
  cancelReplayLoop();
  state.lastReplayFrame = null;
  replayTick();
  if (typeof elements.headVideo.requestVideoFrameCallback === "function") {
    state.replayTimerKind = "video-frame";
    const onVideoFrame = (_now, metadata) => {
      replayTick(metadata.mediaTime);
      if (!elements.headVideo.paused) {
        state.replayTimer = elements.headVideo.requestVideoFrameCallback(onVideoFrame);
      }
    };
    state.replayTimer = elements.headVideo.requestVideoFrameCallback(onVideoFrame);
    return;
  }
  state.replayTimerKind = "animation-frame";
  const onAnimationFrame = () => {
    replayTick();
    if (!elements.headVideo.paused) {
      state.replayTimer = requestAnimationFrame(onAnimationFrame);
    }
  };
  state.replayTimer = requestAnimationFrame(onAnimationFrame);
}

async function playReplay() {
  const replay = state.bootstrap.replay;
  elements.replayPlay.disabled = true;
  try {
    if (elements.headVideo.currentTime >= replay.video.to_s - 0.02) {
      await seekVideo(replay.video.from_s);
      updateReplayLabels(0);
      await requestReplayFrame(0);
    } else if (elements.headVideo.seeking) {
      await waitForMediaEvent("seeked");
    }
    await elements.headVideo.play();
    elements.replayPause.disabled = false;
    startReplayLoop();
  } catch (error) {
    elements.replayPlay.disabled = false;
    showToast(`视频无法播放：${error.message}`, true);
  }
}

function pauseReplay() {
  elements.headVideo.pause();
  cancelReplayLoop();
  elements.replayPlay.disabled = false;
  elements.replayPause.disabled = true;
}

function cancelReplayLoop() {
  if (state.replayTimer !== null && state.replayTimerKind === "video-frame") {
    elements.headVideo.cancelVideoFrameCallback(state.replayTimer);
  } else if (state.replayTimer !== null && state.replayTimerKind === "animation-frame") {
    cancelAnimationFrame(state.replayTimer);
  }
  state.replayTimer = null;
  state.replayTimerKind = null;
}

async function resetReplay() {
  pauseReplay();
  elements.replayReset.disabled = true;
  try {
    await seekVideo(state.bootstrap.replay.video.from_s);
    updateReplayLabels(0);
    await requestReplayFrame(0);
    showToast("机器人已复位到 Episode 0 的第 0 帧");
  } finally {
    elements.replayReset.disabled = false;
  }
}

async function setCameraView(view, { silent = false } = {}) {
  for (const button of elements.cameraViewButtons) button.disabled = true;
  try {
    await request("/api/camera/view", {
      method: "POST",
      body: JSON.stringify({ view }),
    });
    state.cameraView = view;
    for (const button of elements.cameraViewButtons) {
      button.classList.toggle("is-active", button.dataset.cameraView === view);
    }
    if (!silent) showToast(view === "near" ? "已恢复双臂近景" : "已切换到整机远景");
  } catch (error) {
    if (!silent) showToast(error.message, true);
  } finally {
    for (const button of elements.cameraViewButtons) button.disabled = false;
  }
}

const pendingJoints = new Map();

function scheduleJointUpdate(name, value) {
  pendingJoints.set(name, value);
  clearTimeout(state.jointTimer);
  state.jointTimer = setTimeout(async () => {
    const joints = Object.fromEntries(pendingJoints);
    pendingJoints.clear();
    try {
      await request("/api/joints", {
        method: "POST",
        body: JSON.stringify({ joints }),
      });
    } catch (error) {
      showToast(error.message, true);
    }
  }, 45);
}

function updateSurfaceOutputs() {
  elements.surfaceHeightOutput.textContent = Number(elements.surfaceHeight.value).toFixed(3);
  elements.hardMarginOutput.textContent = Number(elements.hardMargin.value).toFixed(3);
}

function scheduleSurfaceUpdate() {
  updateSurfaceOutputs();
  clearTimeout(state.surfaceTimer);
  state.surfaceTimer = setTimeout(async () => {
    try {
      await request("/api/work-surface", {
        method: "POST",
        body: JSON.stringify({
          height: Number(elements.surfaceHeight.value),
          hard_margin: Number(elements.hardMargin.value),
        }),
      });
      await requestReplayFrame(Number(elements.replayTimeline.value));
    } catch (error) {
      showToast(error.message, true);
    }
  }, 60);
}

async function resetPose() {
  elements.resetButton.disabled = true;
  try {
    const result = await request("/api/reset", { method: "POST", body: "{}" });
    const byName = new Map(result.joints.map((joint) => [joint.name, joint]));
    for (const input of document.querySelectorAll("[data-joint]")) {
      const joint = byName.get(input.dataset.joint);
      if (!joint) continue;
      input.value = String(joint.value);
      input.dataset.unit = joint.unit;
      input.parentElement.querySelector("output").textContent = formatJointValue(
        joint.value,
        joint.unit,
      );
    }
    showToast("姿态已复位到 URDF 零位");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    elements.resetButton.disabled = false;
  }
}

function renderBootstrap(data) {
  state.bootstrap = data;
  const surface = data.work_surface;
  elements.surfaceHeight.value = String(surface.height);
  elements.hardMargin.value = String(surface.hard_margin);
  updateSurfaceOutputs();
  renderSafetyLimit(data.safety.enforcement_enabled);
  renderJoints(data.joints);
  const replay = data.replay;
  elements.episodeChip.textContent = `Episode ${replay.episode_index}`;
  elements.replayTask.textContent = replay.task;
  elements.replayTimeline.max = String(replay.frame_count - 1);
  elements.headVideo.src = replay.video.url;
  elements.headVideo.addEventListener("loadedmetadata", () => {
    elements.headVideo.currentTime = replay.video.from_s;
  }, { once: true });
  const markReplayReady = () => {
    elements.replayPlay.disabled = false;
    elements.replayPlay.textContent = "开始回放";
  };
  if (elements.headVideo.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA) {
    markReplayReady();
  } else {
    elements.headVideo.addEventListener("canplay", markReplayReady, { once: true });
  }
  elements.headVideo.load();
  updateReplayLabels(0);

  elements.modelName.textContent = data.model.variant?.label || data.model.name;
  elements.jointCount.textContent = String(data.model.controllable_joint_count);
  elements.geometryCount.textContent = `${data.model.geometry_count} / ${data.model.face_count.toLocaleString()}`;
  const geometryStatus = data.safety.visual_collision_geometry_aligned
    ? "安全几何已对齐"
    : "视觉预览 · 安全几何仍使用旧版";
  elements.assetStatus.textContent = `${data.assets.file_count} 文件 · SHA 已验证 · ${geometryStatus}`;

  const viewerUrl = `${window.location.origin}${data.viewer.path}?meshcat_port=${data.viewer.meshcat_port}`;
  elements.frame.src = viewerUrl;
  elements.frame.addEventListener(
    "load",
    () => {
      elements.loading.classList.add("is-hidden");
      elements.viewerStatus.textContent = "三维资源已连接";
      elements.viewerStatus.className = "connection-state is-ready";
      // iframe load 早于 MeshCat WebSocket 完成场景同步；稍后应用视角，
      // 否则命令可能被新连接的默认 camera 覆盖。
      window.setTimeout(() => setCameraView(state.cameraView, { silent: true }), 750);
    },
    { once: true },
  );
  requestReplayFrame(0);
}

async function initialize() {
  elements.surfaceHeight.addEventListener("input", scheduleSurfaceUpdate);
  elements.hardMargin.addEventListener("input", scheduleSurfaceUpdate);
  elements.safetyLimit.addEventListener("change", updateSafetyLimit);
  elements.resetButton.addEventListener("click", resetPose);
  for (const button of elements.cameraViewButtons) {
    button.addEventListener("click", () => setCameraView(button.dataset.cameraView));
  }
  elements.replayPlay.addEventListener("click", playReplay);
  elements.replayPause.addEventListener("click", pauseReplay);
  elements.replayReset.addEventListener("click", resetReplay);
  elements.replayTimeline.addEventListener("input", () => {
    pauseReplay();
    const frame = Number(elements.replayTimeline.value);
    const replay = state.bootstrap.replay;
    updateReplayLabels(frame);
    clearTimeout(state.replaySeekTimer);
    state.replaySeekTimer = setTimeout(async () => {
      await seekVideo(replay.video.from_s + frame / replay.fps);
      await requestReplayFrame(frame);
    }, 40);
  });
  elements.replaySource.addEventListener("change", () => {
    requestReplayFrame(Number(elements.replayTimeline.value));
  });
  try {
    renderBootstrap(await request("/api/bootstrap"));
  } catch (error) {
    elements.viewerStatus.textContent = "加载失败";
    elements.viewerStatus.className = "connection-state is-error";
    elements.loading.querySelector("p").textContent = error.message;
    elements.loading.querySelector("small").textContent = "请检查服务端日志";
    showToast(error.message, true);
  }
}

initialize();
