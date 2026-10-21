const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const root = path.join(__dirname, "..", "src", "nexus_deploy", "static");

function browser(pathname = "/session.html") {
  const elements = new Map(), handlers = new Map(), redirects = [];
  const element = () => ({
    disabled: true, hidden: false, children: [], listeners: {},
    classList: { add() {}, remove() {} },
    querySelector() { return get("automatic-option"); },
    addEventListener(name, listener) { this.listeners[name] = listener; },
    replaceChildren() { this.children = []; },
    append(...items) { this.children.push(...items); },
    setAttribute() {},
  });
  const get = (id) => {
    if (!elements.has(id)) elements.set(id, element());
    return elements.get(id);
  };
  const context = vm.createContext({
    document: { getElementById: (id) => {
      if (id === "workstation-panel" && !["/session.html", "/monitor"].includes(pathname)) return null;
      if (id === "workstation-records" && pathname !== "/history.html") return null;
      return get(id);
    }, createElement: element, hidden: false },
    window: { location: { search: "", pathname, replace: (url) => redirects.push(url) }, confirm: () => true,
      dispatchEvent() {}, setInterval() {}, addEventListener: (name, fn) => handlers.set(name, fn) },
    CustomEvent: class {}, URLSearchParams, URL, Headers,
  });
  const run = (code) => vm.runInContext(code, context);
  // Load the real browser helpers before executing the module in this VM.
  for (const module of ["alignment-link.js", "ui-session.js"]) {
    run(fs.readFileSync(path.join(root, module), "utf8").replace(/^export /gm, ""));
  }
  return { context, run, get, handlers, redirects };
}

async function workstationFixture(pathname = "/session.html") {
  const b = browser(pathname);
  const station = {
    mode: "debug", accepting: false, reason: "控制服务已重启",
    profiles: [{ ready: true }], active_tasks: [], active_deployment_id: "deployment",
    console_deployments: [{ id: "deployment", phase: "standby", robot_id: "g2a", model_id: "pi05", is_current: true }],
  };
  const taskItems = [];
  const writes = [];
  let deny = false;
  b.context.fetch = async (url, options = {}) => {
    if (options.method === "POST") {
      const change = JSON.parse(options.body);
      writes.push(change);
      if (deny) return { ok: false, json: async () => ({ reason_code: "NOT_READY", error: "模型未就绪" }) };
      Object.assign(station, change);
      if ("accepting" in change) station.reason = station.accepting ? "" : "操作员已暂停新任务";
    }
    const data = /\/(bootstrap|csrf-token)$/.test(url) ? { csrf_token: "test" }
      : url.endsWith("/tasks") ? { tasks: structuredClone(taskItems) } : structuredClone(station);
    return { ok: true, json: async () => data };
  };
  b.run(fs.readFileSync(path.join(root, "workstation.js"), "utf8").replace(/^import .*;\n/gm, ""));
  await new Promise(setImmediate);
  return { ...b, station, taskItems, writes, reject: () => { deny = true; } };
}

test("handoff grants the selected entry only after backend admission; it never submits a task", async () => {
  const b = await workstationFixture();
  const handoff = b.get("workstation-handoff"), action = b.get("workstation-admission-action");
  assert.equal(handoff.disabled, false);
  assert.equal(handoff.textContent, "交还调度");
  assert.equal(action.textContent, "核实后允许新任务");
  await handoff.listeners.click();
  assert.deepEqual(b.writes[0], { mode: "automatic", accepting: true });
  assert.equal(b.station.accepting, true);
  assert.match(b.get("workstation-status").textContent, /等待调度 API/);
  assert.equal(action.textContent, "暂停新任务");
  assert.match(action.title, /不会暂停、恢复或结束正在运行的任务/);
  await action.listeners.click();
  assert.deepEqual(b.writes[1], { accepting: false });
  assert.equal(b.station.mode, "automatic");
  await handoff.listeners.click();
  assert.deepEqual(b.writes[2], { mode: "debug", accepting: true });
  assert.equal(b.station.mode, "debug");
});

test("unready or busy control cannot hand off; rejected change preserves authority", async () => {
  const b = await workstationFixture();
  b.station.profiles = [];
  await b.run("refresh()");
  assert.equal(b.get("workstation-handoff").disabled, true);
  b.station.profiles = [{ ready: true }];
  await b.run("refresh()");
  b.reject();
  await b.get("workstation-handoff").listeners.click();
  assert.equal(b.station.mode, "debug");
  assert.match(b.get("workstation-status").textContent, /NOT_READY/);
  b.station.active_tasks = ["task"];
  await b.run("refresh()");
  assert.equal(b.get("workstation-handoff").disabled, true);
});

test("reset gate keeps scheduler release disabled and takeover does not reopen admission", async () => {
  const b = await workstationFixture();
  b.station.mode = "automatic";
  b.station.robot_gate = { required: true, status: "failed", reason: "复位失败" };
  await b.run("refresh()");
  await b.get("workstation-handoff").listeners.click();
  assert.deepEqual(b.writes[0], { mode: "debug", accepting: false });
  assert.equal(b.get("workstation-handoff").disabled, true);
  assert.equal(b.get("workstation-admission-action").disabled, true);
});

test("run status shows one current task; task records use separate page", async () => {
  const b = await workstationFixture();
  b.station.console_deployments.push(
    { id: "candidate", phase: "candidate", model_id: "pi05", is_current: false },
  );
  b.station.current_task_id = "active";
  b.taskItems.push(
    { task_id: "active", source: "scheduler", status: "running" },
    { task_id: "past", source: "scheduler", status: "completed" },
    { task_id: "recovery", source: "scheduler", status: "recovery_required" },
  );
  await b.run("refresh()");
  assert.match(b.get("workstation-current-task").children[0].textContent, /当前任务/);

  const records = await workstationFixture("/history.html");
  records.station.console_deployments = structuredClone(b.station.console_deployments);
  records.station.current_task_id = "active";
  records.taskItems.push(...b.taskItems);
  await records.run("refresh()");
  assert.equal(records.get("records-tasks").children.length, 1);
  assert.match(records.get("records-status").textContent, /已结束任务 1 条/);
});

test("saving a replacement applies it through the backend and never starts a task", async () => {
  const calls = [], redirects = [];
  const state = { busy: false, jobId: "current", job: {}, formDirty: true };
  const context = vm.createContext({
    state,
    window: { confirm: () => true, location: { assign: (url) => redirects.push(url) },
      history: { replaceState() {} } },
    api: async (url, options = {}) => {
      calls.push([url, options.method || "GET"]);
      if (url === "/workstation") return { active_tasks: [], active_deployment_id: "current" };
      if (url.endsWith("/configure")) return { id: "replacement" };
      if (url === "/workstation/activate") return { switching_deployment_id: "replacement" };
      throw new Error(`Unexpected request: ${url}`);
    },
    setFormEnabled() {}, renderControls() {}, runtimePayload: () => ({ task: "new" }),
    csrfHeaders: () => ({}), render() {}, toast() {}, isAbortLike: () => false,
    showRuntimeValidationErrors() {},
  });
  const source = fs.readFileSync(path.join(root, "session.js"), "utf8");
  vm.runInContext(source.slice(source.indexOf("async function applyConfiguration()"),
    source.indexOf("function showRuntimeValidationErrors", source.indexOf("async function applyConfiguration()"))), context);
  assert.equal(await vm.runInContext("applyConfiguration()", context), true);
  assert.deepEqual(calls, [
    ["/workstation", "GET"], ["/deployments/current/configure", "POST"],
    ["/workstation", "GET"], ["/workstation/activate", "POST"],
  ]);
  assert.deepEqual(redirects, ["./session.html?id=replacement"]);
  assert.equal(state.formDirty, false);

  calls.length = 0;
  context.api = async (url) => { calls.push(url); return { active_tasks: ["task"] }; };
  assert.equal(await vm.runInContext("applyConfiguration()", context), false);
  assert.deepEqual(calls, ["/workstation"]);
});

function sessionFixture(station, monitor = true) {
  const b = browser();
  b.context.monitorView = monitor;
  const calls = [];
  b.context.api = async (path) => { calls.push(path); return path === "/bootstrap" ? {} : station; };
  b.context.setConnection = () => {};
  b.context.initializeTwinViewer = () => { throw new Error("No task or model may be started by navigation"); };
  b.context.renderControls = () => {};
  b.run('const state={jobId:"",formDirty:false};const dom={route_label:document.getElementById("route-label")};');
  const source = fs.readFileSync(path.join(root, "session.js"), "utf8");
  b.run(source.slice(source.indexOf("async function bootstrap()"), source.indexOf('dom.runtime_form.addEventListener("input"')));
  return { ...b, calls };
}

test("empty and multiple-deployment entry shows monitor overview without starting any session", async () => {
  for (const entries of [[], [{ id: "a" }, { id: "b" }]]) {
    const b = sessionFixture({ landing_url: "/monitor", console_deployments: entries });
    await b.run("bootstrap()");
    assert.deepEqual(b.redirects, []);
    assert.deepEqual(b.calls, ["/bootstrap", "/workstation"]);
    assert.equal(b.get("console-empty").hidden, true);
    assert.equal(b.get("monitor-empty").hidden, false);
  }
});

test("single current deployment resolves to its console", async () => {
  const b = sessionFixture({ active_deployment_id: "ready" });
  await b.run("bootstrap()");
  assert.deepEqual(b.redirects, ["./monitor?id=ready"]);
});

test("settings entrance selects the same saved deployment without creating a task", async () => {
  const b = sessionFixture({ active_deployment_id: "ready", console_deployments: [
    { id: "ready", phase: "standby" },
  ] }, false);
  await b.run("bootstrap()");
  assert.deepEqual(b.redirects, ["./session.html?id=ready"]);
  assert.deepEqual(b.calls, ["/bootstrap", "/workstation"]);
});

test("runtime payload preserves the selected tracking and RTC prefix experiment", () => {
  const source = fs.readFileSync(path.join(root, "session.js"), "utf8");
  const payloadSource = source.slice(source.indexOf("function runtimePayload()"), source.indexOf("function armForSecondClick"));
  const dom = new Proxy({}, { get(target, key) {
    return target[key] ||= { value: "1", checked: false };
  } });
  const context = vm.createContext({ dom, state: { job: { config: { safety: { watchdog_enabled: true } } } } });
  vm.runInContext(payloadSource, context);
  const html = fs.readFileSync(path.join(root, "session.html"), "utf8");
  for (const [mode, omega, prefix] of [["legacy", 12, "model"], ["damped", 16, "command-shadow"], ["legacy", 12, "command"]]) {
    dom.arm_tracking_mode.value = mode;
    dom.arm_tracking_omega.value = String(omega);
    dom.rtc_prefix_mode.value = prefix;
    const payload = vm.runInContext("runtimePayload()", context);
    assert.equal(payload.arm_tracking_mode, mode);
    assert.equal(payload.arm_tracking_omega, omega);
    assert.equal(payload.rtc_prefix_mode, prefix);
    assert.equal(payload.safety.watchdog_enabled, true);
  }
  for (const field of ["arm-tracking-mode", "arm-tracking-omega", "rtc-prefix-mode"]) {
    assert.ok(html.includes(`id="${field}"`));
    assert.ok(source.includes(`"${field}"`));
  }
});

test("settings entrance prefers the current configuration over an old pending edit", async () => {
  const b = sessionFixture({ active_deployment_id: "current", console_deployments: [
    { id: "pending", phase: "candidate" }, { id: "current", phase: "standby" },
  ] }, false);
  await b.run("bootstrap()");
  assert.deepEqual(b.redirects, ["./session.html?id=current"]);
});

test("stale settings URL returns to the current configuration", async () => {
  const b = sessionFixture({ active_deployment_id: "current", console_deployments: [
    { id: "current", phase: "standby" },
  ] }, false);
  b.run('state.jobId="old-task"');
  await b.run("bootstrap()");
  assert.deepEqual(b.redirects, ["./session.html"]);
  assert.deepEqual(b.calls, ["/bootstrap", "/workstation"]);
});

test("model timer advances while loading and freezes at the completed duration", () => {
  const source = fs.readFileSync(path.join(root, "session.js"), "utf8");
  const duration = source.slice(source.indexOf("function formatDuration("), source.indexOf("function formatTime("));
  const elapsed = source.slice(source.indexOf("function modelElapsedText("), source.indexOf("function renderModelElapsed("));
  const context = vm.createContext({});
  vm.runInContext(duration + elapsed, context);
  const text = (service, ageMs) => {
    context.service = service;
    context.ageMs = ageMs;
    return vm.runInContext("modelElapsedText(service, ageMs)", context);
  };
  assert.equal(text({ status: "candidate", started_at: null, timeout_s: 600 }, 0), "未开始");
  assert.equal(text({ status: "ready", started_at: null, timeout_s: 600 }, 0), "已复用服务");
  assert.equal(text({ status: "starting", started_at: 100, elapsed_s: 9, timeout_s: 600 }, 2000), "11 秒");
  assert.equal(text({ status: "ready", started_at: 100, finished_at: 160, elapsed_s: 60 }, 2000), "1分 00秒");
});

test("idle automatic console follows its own new batch, not history or unsaved edits", () => {
  const b = sessionFixture({});
  b.run('state.jobId="template";state.job={status:"ready_to_start",task_session:{},workstation:{mode:"automatic"}};');
  const changed = b.handlers.get("nexus-workstation-change");
  const station = { mode: "automatic", console_deployments: [
    { id: "unrelated", phase: "active", source: "scheduler", deployment_id: "other" },
    { id: "new-batch", phase: "active", source: "scheduler", deployment_id: "template" },
  ] };
  b.run("state.formDirty=true"); changed({ detail: station });
  assert.deepEqual(b.redirects, []);
  b.run("state.formDirty=false;state.job.task_session.accepted=true"); changed({ detail: station });
  assert.deepEqual(b.redirects, []);
  b.run("state.job.task_session.accepted=false"); changed({ detail: station });
  assert.deepEqual(b.redirects, ["./monitor?id=new-batch"]);
});
