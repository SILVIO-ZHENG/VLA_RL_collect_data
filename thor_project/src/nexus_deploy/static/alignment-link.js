// The alignment tool is a separate service; the control plane supplies its URL.
export function configureAlignmentLink(bootstrap, context = () => ({})) {
  const link = document.getElementById("alignment-entry");
  if (!link) return;
  const baseUrl = bootstrap?.tools?.alignment_url;
  link.hidden = typeof baseUrl !== "string" || !baseUrl;
  link.onclick = link.hidden ? null : (event) => {
    const target = new URL(baseUrl, window.location.href);
    if (window.location.hostname) target.hostname = window.location.hostname;
    if (["127.0.0.1", "localhost"].includes(window.location.hostname) && window.location.port === "18765") {
      target.port = "18775";
    }
    const { robotHost, robotId } = context();
    if (robotHost) target.searchParams.set("host", robotHost);
    if (robotId) target.searchParams.set("robot", robotId);
    link.href = target.toString();
  };
}
