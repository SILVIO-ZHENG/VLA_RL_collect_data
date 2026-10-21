// Obtain the live token before a user-requested write. Never replay an action.
export async function freshCsrfHeaders(options = {}) {
  const headers = new Headers(options.headers || {});
  const method = String(options.method || "GET").toUpperCase();
  if (["GET", "HEAD", "OPTIONS"].includes(method)) return headers;
  const response = await fetch("/api/csrf-token", {
    method: "GET", credentials: "same-origin", cache: "no-store",
    redirect: "error", signal: options.signal,
    headers: { Accept: "application/json" },
  });
  let payload = null;
  try { payload = await response.json(); } catch { /* Reject non-JSON login/error pages. */ }
  if (!response.ok || typeof payload?.csrf_token !== "string" || !payload.csrf_token.trim()) {
    throw new Error(payload?.error || `无法更新网页令牌（HTTP ${response.status}）；操作未发送，请刷新页面并核实状态。`);
  }
  headers.set("X-CSRF-Token", payload.csrf_token);
  return headers;
}

export function sessionErrorMessage(response, payload) {
  if (response.status === 403 && payload?.error === "CSRF token 无效") {
    return "CSRF token 无效：控制服务在操作期间重启，操作未执行；请刷新页面、核实当前状态后再次点击。";
  }
  return null;
}
