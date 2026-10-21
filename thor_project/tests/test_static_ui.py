from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "nexus_deploy"
APP_JS = PACKAGE_ROOT / "static" / "app.js"
INDEX_HTML = PACKAGE_ROOT / "static" / "index.html"
STYLES_CSS = PACKAGE_ROOT / "static" / "styles.css"
SESSION_CSS = PACKAGE_ROOT / "static" / "session.css"
SESSION_JS = PACKAGE_ROOT / "static" / "session.js"
SESSION_HTML = PACKAGE_ROOT / "static" / "session.html"
ALIGNMENT_LINK_JS = PACKAGE_ROOT / "static" / "alignment-link.js"


def test_pages_use_the_bundled_company_logo_as_favicon() -> None:
    index = INDEX_HTML.read_text(encoding="utf-8")
    session = SESSION_HTML.read_text(encoding="utf-8")

    favicon = '<link rel="icon" type="image/png" href="./assets/anu-intelligent-logo.png" />'
    assert favicon in index
    assert favicon in session


def test_three_function_pages_keep_controls_on_the_robot_console() -> None:
    model_page = INDEX_HTML.read_text(encoding="utf-8")
    console = SESSION_HTML.read_text(encoding="utf-8")
    styles = (PACKAGE_ROOT / "static" / "session.css").read_text(encoding="utf-8")
    controller = (PACKAGE_ROOT / "static" / "workstation.js").read_text(encoding="utf-8")
    assert 'href="./session.html">运行配置</a>' in model_page
    assert 'href="./">机器人控制台</a>' in model_page
    assert 'id="workstation-handoff"' in console
    assert 'id="workstation-mode"' not in console
    assert 'id="console-probe-robot"' in console
    assert '.settings-page .workstation-panel,' in styles
    assert '.monitor-page #runtime-config-card' in styles
    assert 'mode: station.mode === "automatic" ? "debug" : "automatic"' in controller
    assert 'accepting: station.mode === "automatic" ? !station.robot_gate?.required : true' in controller


def test_motion_experiments_are_selectable_but_not_default() -> None:
    index = INDEX_HTML.read_text(encoding="utf-8")
    app = APP_JS.read_text(encoding="utf-8")
    session = SESSION_HTML.read_text(encoding="utf-8")
    session_js = SESSION_JS.read_text(encoding="utf-8")
    assert 'value="urdf-velocity-only"' in index
    assert 'value="monotone-cubic"' in index
    assert 'value="urdf-velocity-only"' in session
    assert 'value="monotone-cubic"' in session
    assert 'arm_limit_profile: "urdf-velocity-uniform-acceleration"' in app
    assert 'action_interpolation: "linear"' in app
    assert 'action_interpolation: dom.action_interpolation.value' in session_js
    assert 'inference_mode: "rtc"' in app
    assert 'hardware_mode: "execute"' in app
    assert 'name="inference_mode" value="rtc" checked' in index
    assert 'job.task_session?.runtime_configured === false' in session_js


def test_package_completion_is_configured_after_model_load() -> None:
    index = INDEX_HTML.read_text(encoding="utf-8")
    app = APP_JS.read_text(encoding="utf-8")
    session = SESSION_HTML.read_text(encoding="utf-8")
    session_js = SESSION_JS.read_text(encoding="utf-8")
    assert 'presence-gate-options' not in index
    assert 'name="presence_gate_enabled"' not in index
    assert 'presence_gate: { enabled: false }' in app
    assert 'id="presence-gate-enabled"' in session
    assert 'enabled: dom.presence_gate_enabled.value === "enabled"' in session_js
    assert '"presence_gate.enabled": "presence-gate-enabled"' in session_js


def test_gpu_recommendation_and_selection_have_distinct_states() -> None:
    app = APP_JS.read_text(encoding="utf-8")
    styles = STYLES_CSS.read_text(encoding="utf-8")

    assert 'card.classList.toggle("is-selected", selected)' in app
    assert 'badge.dataset.baseLabel' in app
    assert 'automatic ? "自动选中" : "已选择"' in app
    assert ".gpu-card.is-recommended" in styles
    assert ".gpu-card.is-selected" in styles


def test_bootstrap_selects_model_before_refreshing_gpu_inventory() -> None:
    app = APP_JS.read_text(encoding="utf-8")
    start = app.index("async function loadBootstrap()")
    end = app.index("async function refreshGpuInventory", start)
    bootstrap = app[start:end]

    assert bootstrap.index("applyDefaults();") < bootstrap.index("renderGpus(")
    assert bootstrap.index("renderGpus(") < bootstrap.index(
        "await refreshGpuInventory({ silent: true });"
    )


def test_robot_probe_prefers_declared_wired_route_and_falls_back_safely() -> None:
    app = APP_JS.read_text(encoding="utf-8")

    assert "async function probeRobotEndpoint({ automatic = false, silent = false } = {})" in app
    assert "const candidates = automatic || currentIsPreferred" in app
    assert 'candidateResult.client?.status !== "wrong_robot"' in app
    assert "candidateResult.identity?.match !== false" in app
    assert "已自动回退到" in app
    assert "await probeRobotEndpoint({ automatic: true, silent: true });" in app


def test_occupied_robot_session_uses_a_two_step_cleanup_dialog() -> None:
    index = INDEX_HTML.read_text(encoding="utf-8")
    app = APP_JS.read_text(encoding="utf-8")
    styles = STYLES_CSS.read_text(encoding="utf-8")

    assert 'id="session-cleanup-dialog"' in index
    assert "确认 1 / 2" in index
    assert "确认 2 / 2" in index
    assert 'id="cleanup-confirmation-input"' in index
    assert 'id="cleanup-confirm"' in index
    assert 'apiFetch("/robots/cleanup/prepare"' in app
    assert 'apiFetch("/robots/cleanup"' in app
    assert "function cleanupTargetRequest(target)" in app
    assert "JSON.stringify(cleanupTargetRequest(state.cleanupTarget))" in app
    assert "...cleanupTargetRequest(state.cleanupTarget)" in app
    assert "result.confirmation_phrase" in app
    assert 'id="occupied-session-notice"' in index
    assert 'id="open-session-cleanup"' in index
    assert "syncOccupiedSessionNotice(result);" in app
    assert "if (occupiedSessionTarget(result)) openCleanupDialog(result);" in app
    assert '!surfacedJob && state.robotProbe && /当前实例被占用/' in app
    assert "openCleanupDialog(state.robotProbe, { force: true });" in app
    assert ".session-cleanup-dialog::backdrop" in styles


def test_occupied_deployment_is_linked_to_robot_console() -> None:
    app = APP_JS.read_text(encoding="utf-8")
    index = INDEX_HTML.read_text(encoding="utf-8")

    assert "function occupiedDeploymentId(message)" in app
    assert "async function surfaceOccupiedDeployment(message)" in app
    assert "await surfaceOccupiedDeployment(message)" in app
    assert "已找到当前占用任务，请在机器人控制台查看并处理" in app
    assert 'candidateOnly ? "./session.html" : "./monitor"' in app
    assert 'dom.jobConsoleLink.textContent = candidateOnly ? "进入运行配置并应用"' in app
    assert 'id="job-console-link"' in index
    assert 'id="start-job"' not in index
    assert 'id="stop-job"' not in index


def test_saved_twin_camera_wins_after_viewer_reload() -> None:
    session = SESSION_JS.read_text(encoding="utf-8")

    assert "function stabilizeInitialTwinCamera(camera)" in session
    assert "[0, 180, 500].forEach" in session
    assert 'const TWIN_VIEW_STORAGE_KEY = "nexus-platform-twin-view";' in session
    assert "twinView: savedTwinView()" in session
    assert "rememberTwinView(view);" in session
    assert "setTwinView(state.twinView, { force: true, stabilize: true })" in session
    assert "if (stabilize) stabilizeInitialTwinCamera(state.twinPresets[view]);" in session


def test_live_twin_uses_the_current_job_effector_profile() -> None:
    session = SESSION_JS.read_text(encoding="utf-8")

    assert 'effectorProfile: state.job?.config?.effector_profile || "dahuan"' in session
    assert "deployment.config?.effector_profile" not in session


def test_model_selection_recomputes_service_mode() -> None:
    app = APP_JS.read_text(encoding="utf-8")

    assert 'return modelHasHealthyService(model) ? "reuse" : "real";' in app
    assert '["robot_id", "effector_profile", "model_id"].includes(target.name)' in app
    assert "updateDependentControls();\n    if ([" in app
    assert 'reuseInput.disabled = !reusable' in app


def test_gpu_cards_are_noninteractive_when_model_is_not_loading() -> None:
    app = APP_JS.read_text(encoding="utf-8")
    styles = STYLES_CSS.read_text(encoding="utf-8")

    assert "function syncGpuCardAvailability()" in app
    assert 'card.setAttribute("aria-disabled", String(disabled))' in app
    assert "card.tabIndex = disabled ? -1 : 0" in app
    assert ".gpu-card.is-interaction-disabled" in styles


def test_homepage_has_no_static_draft_badge() -> None:
    index = INDEX_HTML.read_text(encoding="utf-8")
    styles = STYLES_CSS.read_text(encoding="utf-8")

    assert "草稿" not in index
    assert "draft-badge" not in index
    assert ".draft-badge" not in styles


def test_arm_filter_keeps_execute_default_until_readonly_trajectory_acceptance() -> None:
    index = INDEX_HTML.read_text(encoding="utf-8")
    app = APP_JS.read_text(encoding="utf-8")

    assert "action_low_pass_cutoff_hz: 4" in app
    assert "真机默认仍为已验证的 4 Hz" in index


def test_postload_page_exposes_arm_dynamic_limits() -> None:
    session = SESSION_HTML.read_text(encoding="utf-8")
    session_js = SESSION_JS.read_text(encoding="utf-8")

    assert 'id="arm-limit-profile"' in session
    assert 'id="max-arm-velocity"' in session
    assert 'id="max-arm-acceleration"' in session
    assert 'value="urdf-velocity-uniform-acceleration">当前 URDF 逐关节速度 + 统一加速度（默认）' in session
    assert 'value="urdf-velocity-p99-acceleration"' in session
    assert "dom.arm_limit_profile.value = config.arm_limit_profile" in session_js
    assert "arm_limit_profile: dom.arm_limit_profile.value" in session_js


def test_both_pages_expose_mutually_exclusive_chunk_smoothing_modes() -> None:
    index = INDEX_HTML.read_text(encoding="utf-8")
    app = APP_JS.read_text(encoding="utf-8")
    session = SESSION_HTML.read_text(encoding="utf-8")
    session_js = SESSION_JS.read_text(encoding="utf-8")

    for markup in (index, session):
        assert 'value="adaptive-fixed-time"' in markup
        assert 'value="savgol"' in markup
    assert 'chunk_smoothing_mode: "savgol"' in app
    assert "chunk_smoothing_mode: document.querySelector" in app
    assert '"chunk-smoothing-mode"' in session_js
    assert "chunk_smoothing_mode: dom.chunk_smoothing_mode.value" in session_js


def test_alignment_is_available_from_navigation_when_configured() -> None:
    for page in (INDEX_HTML, SESSION_HTML, PACKAGE_ROOT / "static" / "history.html"):
        html = page.read_text(encoding="utf-8")
        assert html.count('id="alignment-entry"') == 1
        assert 'id="alignment-entry" href="#" target="_blank" rel="noopener" hidden' in html
    helper = ALIGNMENT_LINK_JS.read_text(encoding="utf-8")
    assert "bootstrap?.tools?.alignment_url" in helper
    assert 'target.searchParams.set("host", robotHost)' in helper
    assert "configureAlignmentLink" in APP_JS.read_text(encoding="utf-8")
    assert "configureAlignmentLink" in SESSION_JS.read_text(encoding="utf-8")


def test_mock_mode_is_visible_and_explicitly_test_only() -> None:
    index = INDEX_HTML.read_text(encoding="utf-8")
    app = APP_JS.read_text(encoding="utf-8")

    assert 'name="server_mode" value="mock"' in index
    assert "Mock 测试" in index
    assert "仅用于测试服务启动、协议握手和只读数据链路" in index
    assert "mockOption.hidden" not in app
    assert 'text = "启动 Mock 测试"' in app


def test_workspace_safety_can_be_enabled_after_model_loading() -> None:
    index = INDEX_HTML.read_text(encoding="utf-8")
    app = APP_JS.read_text(encoding="utf-8")
    session = SESSION_HTML.read_text(encoding="utf-8")
    session_js = SESSION_JS.read_text(encoding="utf-8")

    assert "模型就绪后配置运行安全" in index
    assert (
        'id="workspace-safety-enabled" type="checkbox" '
        'data-safety-key="workspace_safety_enabled" checked disabled'
        in index
    )
    assert "G2 桌面工作区（默认）" in index
    assert "已选择 G2 默认工作区检测配置" in index
    assert 'workspace_safety_enabled: false' in app
    assert 'safety_profile_id: ""' in app
    assert "dom.workspaceSafety.checked = true" in app
    assert 'id="safety-availability"' in session
    assert 'id="workspace-safety" type="checkbox" checked disabled' in session
    assert "仅界面显示" not in index
    assert "仅界面显示" not in session
    assert "仅界面显示" not in session_js
    assert "实际状态：碰撞检测与危险动作拦截均未启用" not in session_js
    assert "加载安全配置…" in session
    assert 'id="safety-availability" role="status" hidden' in session
    assert "profile.read_only_ready === true" in session_js
    assert "profile.execute_approved === true" in session_js
    assert 'profile.inter_arm_collision?.geometry === "cross-chain-capsules-v2"' in session_js
    assert 'profile.inter_arm_collision.validation_status === "execute-approved"' in session_js
    assert "preferredSafetyProfile(profiles, execute)?.id" in session_js
    assert "workspace_safety_enabled: dom.workspace_safety.checked" in session_js
    assert (
        "safety_profile_id: dom.workspace_safety.checked ? dom.safety_profile.value : \"\""
        in session_js
    )
    assert "只读验证可使用 0.750 m 默认桌面高度" in session_js
    assert 'const startableState = ["ready_to_start", "succeeded"].includes(status);' in session_js
    assert "active ? 500 : 1200" in session_js


def test_session_safety_distinguishes_proximity_warning_from_hard_limit() -> None:
    session = SESSION_HTML.read_text(encoding="utf-8")
    session_js = SESSION_JS.read_text(encoding="utf-8")
    session_css = SESSION_CSS.read_text(encoding="utf-8")

    assert 'id="collision-title"' in session
    assert 'id="safety-distance-panel"' in session
    assert 'id="inter-arm-distance-current"' in session
    assert 'id="table-distance-current"' in session
    assert "双臂净距" in session
    assert "夹爪离桌面" in session
    assert "safety.proximity_warning" in session_js
    assert "distanceAtHardLimit(workSurface?.distance_m, workSurface?.hard_margin_m)" in session_js
    assert "distanceAtHardLimit(interArm?.minimum_separation_m, interArm?.hard_margin_m)" in session_js
    assert "safety.distance_to_surface" in session_js
    assert '"桌面几何净距"' in session_js
    assert '"双臂实际净距"' in session_js
    assert "硬限制阈值" in session_js
    assert "tableWarningMm" in session_js
    assert "interArmWarningMm" in session_js
    assert "renderSafetyDistanceRow" in session_js
    assert 'Boolean(candidateInterArm?.warning || candidateInterArm?.collision)' in session_js
    assert "已命中硬限制条件；当前为只读观测，不接管遥操作" in session_js
    assert 'distanceMm: interArmLimiting ? interArmDistanceMm : tableDistanceMm' in session_js
    assert "模型候选动作已拦截" in session_js
    assert "incomingHazardKey !== state.safetyHazardKey" in session_js
    assert "预测路径最小间距" in session_js
    assert "当前双臂间距" in session_js
    assert 'const readOnly = String(control?.hardware_mode || state.job?.config?.hardware_mode) === "read-only";' in session_js
    assert "currentCollision || (!readOnly && candidateCollision)" in session_js
    assert "实测姿态碰撞" in session_js
    assert "不代表当前姿态碰撞" in session_js
    assert "遥测中断，无法判定" in session_js
    assert "遥测中断，结果已过期" in session_js
    assert "当前姿态触发桌面硬限制" in session_js
    assert "夹爪接近桌面" in session_js
    assert "多项碰撞限制已触发" in session_js
    assert 'hazardScope === "multiple"' in session_js
    assert "SAFETY_NOTICE_DURATION_MS = { collision: 2000, warning: 1200 }" in session_js
    assert "preserveActiveCollision" in session_js
    assert "state.safetyNotice.expiresAt = now + SAFETY_NOTICE_DURATION_MS" in session_js
    assert "当前状态已恢复" in session_js
    assert ".collision-banner.is-warning" in session_css
    assert ".safety-distance-panel" in session_css
    assert ".safety-distance-row.is-warning" in session_css
    assert ".safety-distance-row.is-danger" in session_css
    assert 'src="./session.js?v=' in session
    assert 'id="collision-geometry-toggle"' in session
    assert 'type: "nexus-collision-overlay"' in session_js
    assert "stale," in session_js


def test_zero_duration_is_presented_as_continuous_execution() -> None:
    index = INDEX_HTML.read_text(encoding="utf-8")
    app = APP_JS.read_text(encoding="utf-8")
    session = SESSION_HTML.read_text(encoding="utf-8")
    session_js = SESSION_JS.read_text(encoding="utf-8")

    assert 'id="duration-unit"' in index
    assert 'id="duration-unit"' in session
    assert "0（一直执行）" in app
    assert 'continuous ? "（一直执行）" : "秒"' in app
    assert 'continuous ? "（一直执行）" : "秒"' in session_js
    assert 'dom.duration_s.value = "60"' not in session_js


def test_gpu_meter_represents_free_memory_in_green() -> None:
    app = APP_JS.read_text(encoding="utf-8")
    styles = STYLES_CSS.read_text(encoding="utf-8")

    assert 'document.createElement("progress")' in app
    assert "memory.max = totalMemory || 1" in app
    assert "memory.value = freeMemory" in app
    assert "style=\"width:" not in app
    assert ".gpu-memory::-webkit-progress-value" in styles
    assert ".gpu-memory::-moz-progress-bar" in styles
    assert "background: var(--success);" in styles


def test_service_and_compute_resources_share_one_aligned_panel() -> None:
    index = INDEX_HTML.read_text(encoding="utf-8")
    styles = STYLES_CSS.read_text(encoding="utf-8")

    assert 'class="resource-panel"' in index
    assert 'class="resource-panel__service"' in index
    assert 'class="resource-panel__compute gpu-field"' in index
    assert 'class="resource-panel__subheading"' in index
    assert '<legend>\n                <span>计算资源</span>' not in index
    assert ".resource-panel__compute" in styles
    assert ".resource-panel__subheading" in styles
    assert "align-items: center;" in styles


def test_effector_profile_selection_atomically_drives_model_and_limits() -> None:
    index = INDEX_HTML.read_text(encoding="utf-8")
    app = APP_JS.read_text(encoding="utf-8")
    session = SESSION_HTML.read_text(encoding="utf-8")

    assert 'id="summary-effector"' in index
    assert 'id="summary-normalization"' in index
    assert 'id="effector-profile-options"' in index
    assert 'input.name = "effector_profile"' in app
    assert "model?.effector_variants?.[profile]" in app
    assert "applyEffectorProfileDefaults()" in app
    assert "applyEffectorProfileDefaults({ updateTask: true })" in app
    assert "profile.default_task" in app
    assert "平台会据此筛选模型，并绑定对应归一化、行程、开合方向和安全阈值" in index
    session_js = (PACKAGE_ROOT / "static" / "session.js").read_text(
        encoding="utf-8"
    )
    assert "G2_dahuan 末端几何已接入 · 真机测试已授权" in session_js
    assert 'id="model-normalization"' in session
    assert 'id="twin-geometry-note"' in session
    assert "<em>m/s</em>" in index
    assert "<em>m/s²</em>" in index


def test_hardware_is_selected_before_allow_listed_model_version() -> None:
    index = INDEX_HTML.read_text(encoding="utf-8")
    app = APP_JS.read_text(encoding="utf-8")

    assert index.index('id="robot-options"') < index.index('id="effector-profile-options"')
    assert index.index('id="effector-profile-options"') < index.index('id="model-options"')
    assert index.index('id="model-options"') < index.index('id="model-version"')
    assert 'name="model_version"' in index
    assert "model_version: dom.modelVersion.value || null" in app
    assert "model?.running_model_version === selectedVersion" in app
    assert "variant?.versions" in app


def test_dynamic_progress_does_not_depend_on_csp_blocked_inline_styles() -> None:
    app = APP_JS.read_text(encoding="utf-8")
    index = INDEX_HTML.read_text(encoding="utf-8")
    session_js = (PACKAGE_ROOT / "static" / "session.js").read_text(
        encoding="utf-8"
    )
    session_html = (PACKAGE_ROOT / "static" / "session.html").read_text(
        encoding="utf-8"
    )

    assert '<progress class="progress-track" id="job-progress"' in index
    assert '<progress class="runtime-progress" id="model-progress"' in session_html
    assert "dom.jobProgress.value =" in app
    assert "dom.model_progress.value =" in session_js
    assert ".style." not in app
    assert ".style." not in session_js


def test_browser_abort_errors_are_suppressed_at_every_ui_boundary() -> None:
    app = APP_JS.read_text(encoding="utf-8")
    session = (PACKAGE_ROOT / "static" / "session.js").read_text(
        encoding="utf-8"
    )

    for source in (app, session):
        assert "function isAbortLike(error)" in source
        assert "if (isAbortLike(message)) return;" in source
        assert 'window.addEventListener("unhandledrejection"' in source
        assert 'window.addEventListener("error"' in source
        assert "event.preventDefault();" in source


def test_task_control_buttons_share_one_full_width_size() -> None:
    session_styles = SESSION_CSS.read_text(encoding="utf-8")

    assert ".control-actions .button {" in session_styles
    assert "grid-template-columns: repeat(3, minmax(0, 1fr));" in session_styles
    assert "width: 100%;" in session_styles
    assert "min-height: 41px;" in session_styles


def test_task_state_changes_keep_the_runtime_layout_stable() -> None:
    session_styles = SESSION_CSS.read_text(encoding="utf-8")

    assert '.session-main[data-phase="configuration"]' not in session_styles
    assert '.session-main[data-phase="runtime"]' not in session_styles
    assert "grid-template-rows: 41px 41px 41px 41px;" in session_styles
    assert "min-height: 188px;" in session_styles
    assert "#start," in session_styles
    assert "#pause," in session_styles
    assert "#resume {" in session_styles
    assert "#stop {\n  grid-row: 2;\n  grid-column: 1 / -1;" in session_styles
    assert "#reset-robot {\n  grid-row: 3;\n  grid-column: 1 / -1;" in session_styles
    assert "#close-model {\n  grid-row: 4;\n  grid-column: 1 / -1;" in session_styles
    assert ".control-note {\n  box-sizing: border-box;\n  min-height: 58px;" in session_styles


def test_task_controls_disable_unavailable_actions_and_hide_manual_start_in_automatic() -> None:
    session = SESSION_JS.read_text(encoding="utf-8")
    html = SESSION_HTML.read_text(encoding="utf-8")
    session_styles = SESSION_CSS.read_text(encoding="utf-8")

    for control_id in ("start", "pause", "resume", "stop", "reset-robot", "close-model"):
        button = html.split(f'id="{control_id}"', 1)[1].split(">", 1)[0]
        assert "hidden" not in button
        assert "disabled" in button
    for expression in (
        "dom.start.disabled = state.busy || automatic || !canStart;",
        "dom.pause.disabled = state.busy || !canPause;",
        "dom.resume.disabled = state.busy || !canResume;",
        "dom.stop.disabled = state.busy || !canStop;",
        "dom.reset_robot.disabled = state.busy || !canReset;",
    ):
        assert expression in session
    assert "dom.start.hidden = automatic;" in session
    assert "dom.pause.hidden" not in session
    assert "dom.resume.hidden" not in session
    assert ".control-actions .button:disabled {" in session_styles
    assert "opacity: 1;" in session_styles


def test_robot_reset_requires_stopped_task_and_declares_dahuan_gripper_cycle() -> None:
    html = SESSION_HTML.read_text(encoding="utf-8")
    session = SESSION_JS.read_text(encoding="utf-8")

    assert 'id="robot-reset-dialog"' not in html
    assert "resetArmed: false" in session
    assert 'armForSecondClick("reset")' in session
    assert "再次点击确认复位" in session
    assert "RESET ${state.job?.config?.robot_id" not in session
    assert 'postAction("reset-robot"' in session
    assert 'const resettableState = ["ready_to_start", "succeeded"' in session
    assert "!job.task_session?.remote_started" in session


def test_execute_and_reset_use_button_double_click_without_phrase_inputs() -> None:
    html = SESSION_HTML.read_text(encoding="utf-8")
    session = SESSION_JS.read_text(encoding="utf-8")

    assert "startArmed: false" in session
    assert 'armForSecondClick("start")' in session
    assert "再次点击确认启动真机" in session
    assert 'id="execute-confirmation"' not in html
    assert 'id="execute-ack"' not in html
    assert 'id="robot-reset-confirmation"' not in html
    assert "execute_confirmation" not in session
    assert "reset_confirmation" not in session
    assert "dom.hitl.checked = true" not in session


def test_inference_frames_are_preloaded_and_swapped_as_one_batch() -> None:
    session = SESSION_JS.read_text(encoding="utf-8")
    html = SESSION_HTML.read_text(encoding="utf-8")

    assert '<h2 id="vision-title">真实相机图像 · 最近一次推理输入</h2>' in html
    assert "画面按请求更新，不是实时视频" in html
    assert "Promise.all(readyCameras.map(preloadInferenceImage))" in session
    assert "generation !== state.cameraSwapGeneration" in session
    assert "frame.replaceChildren(image)" in session
    assert 'if (frameKey === state.renderedCameraFrameKey' in session
    assert "实时画面" not in html


def test_runtime_robot_view_uses_full_g2_cached_browser_model() -> None:
    session = SESSION_JS.read_text(encoding="utf-8")
    html = SESSION_HTML.read_text(encoding="utf-8")

    assert '<h2 id="robot-twin-title">机器人实时三维状态</h2>' in html
    assert "完整 G2 URDF、网格和原始材质" in html
    assert '<iframe id="robot-twin"' in html
    assert '<canvas id="robot-twin"' not in html
    assert "离线回放" not in html
    assert "function initializeTwinViewer()" in session
    assert "/simulator/sync" not in session
    assert "缓存模型在线" in session
    assert "armPoints" not in session
    assert "drawPose" not in session
    assert "碰撞风险已拦截" in html
    assert "window.setTimeout(pollTwin, state.twinPollIntervalMs)" in session
    assert 'data-twin-view="near"' in html
    assert 'data-twin-view="overview"' in html
    assert "/simulator/view" in session
    assert 'id="twin-save-view"' in html
    assert "/simulator/preset" in session
    assert "nexus-camera-capture" in session
    assert "platform_origin: window.location.origin" in session
    assert "new URLSearchParams" in session
    assert '<div class="twin-placeholder is-loading" id="twin-placeholder" aria-label=' in html
    assert "margin-top: -22px" not in SESSION_CSS.read_text(encoding="utf-8")


def test_runtime_robot_view_uses_cached_model_and_interpolates_joints_in_browser() -> None:
    viewer = (
        PACKAGE_ROOT / "viewer" / "static" / "viewer.js"
    ).read_text(encoding="utf-8")

    assert "const INTERPOLATION_MS = 150;" in viewer
    assert 'fetch("/api/bootstrap", { cache: "no-store"' in viewer
    assert 'function updateMeasuredPose(rawValues, effectorProfile = "dahuan")' in viewer
    assert "function displayJointValues(values)" in viewer
    assert "joint.resultMatrix.copy(joint.originMatrix).multiply(motion)" in viewer
    assert "requestAnimationFrame" in viewer
    assert 'cache: "force-cache"' in viewer
    assert "async function startCachedViewer(modelUrl, workSurface, collisionOverlay)" in viewer
    assert "bootstrap.collision_overlay" in viewer
    assert "collisionCapsules" in viewer
    assert "latchedCollisionLevel" in viewer
    assert "页面姿态复算" not in viewer
    assert "Client实测" not in viewer
    assert "Client桌面净距" not in viewer
    assert "双臂胶囊安全" not in viewer

    viewer_html = (
        PACKAGE_ROOT / "viewer" / "static" / "viewer.html"
    ).read_text(encoding="utf-8")
    assert 'id="collision-readout"' not in viewer_html

    session = SESSION_JS.read_text(encoding="utf-8")
    assert 'type: "nexus-measured-joints"' in session
    assert "joints: control.measured_state" in session
    assert "sequence !== state.twinRenderedSequence || stale !== state.twinRenderedStale" in session
