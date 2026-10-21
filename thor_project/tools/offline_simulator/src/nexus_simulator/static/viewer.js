const params = new URLSearchParams(window.location.search);
const meshcatPort = Number(params.get("meshcat_port")) || 8767;
const platformOrigin = `${window.location.protocol}//${window.location.hostname}:8765`;
const loading = document.querySelector("#viewer-loading");
let viewer = null;

function postToPlatform(message) {
  if (window.parent !== window) window.parent.postMessage(message, platformOrigin);
}

function finiteVector(value, length) {
  return Array.isArray(value)
    && value.length === length
    && value.every((item) => Number.isFinite(Number(item)));
}

function captureCamera() {
  if (!viewer?.camera || !viewer?.controls) throw new Error("三维相机尚未就绪");
  return {
    position: viewer.camera.position.toArray().map(Number),
    quaternion: viewer.camera.quaternion.toArray().map(Number),
    target: viewer.controls.target.toArray().map(Number),
    fov: Number(viewer.camera.fov),
    zoom: Number(viewer.camera.zoom || 1),
  };
}

function applyCamera(camera) {
  if (!viewer?.camera || !viewer?.controls) return false;
  if (!finiteVector(camera?.position, 3) || !finiteVector(camera?.target, 3)) return false;
  viewer.camera.position.fromArray(camera.position.map(Number));
  if (finiteVector(camera.quaternion, 4)) {
    viewer.camera.quaternion.fromArray(camera.quaternion.map(Number));
  }
  viewer.controls.target.fromArray(camera.target.map(Number));
  const fov = Number(camera.fov);
  if (Number.isFinite(fov) && fov >= 10 && fov <= 120) viewer.camera.fov = fov;
  const zoom = Number(camera.zoom);
  if (Number.isFinite(zoom) && zoom > 0 && zoom <= 20) viewer.camera.zoom = zoom;
  viewer.camera.updateProjectionMatrix();
  viewer.controls.update();
  viewer.set_dirty();
  return true;
}

window.addEventListener("message", (event) => {
  if (event.origin !== platformOrigin || event.source !== window.parent) return;
  const message = event.data;
  if (!message || typeof message !== "object") return;
  if (message.type === "nexus-camera-capture") {
    try {
      postToPlatform({
        type: "nexus-camera-state",
        requestId: String(message.requestId || ""),
        camera: captureCamera(),
      });
    } catch (error) {
      postToPlatform({
        type: "nexus-camera-error",
        requestId: String(message.requestId || ""),
        error: String(error?.message || error),
      });
    }
  } else if (message.type === "nexus-camera-apply") {
    applyCamera(message.camera);
  }
});

function startViewer() {
  viewer = new MeshCat.Viewer(document.querySelector("#meshcat-pane"));
  viewer.connect(`ws://${window.location.hostname}:${meshcatPort}`);
  viewer.connection.addEventListener("open", () => {
    window.setTimeout(() => {
      loading.hidden = true;
      postToPlatform({ type: "nexus-viewer-ready" });
    }, 500);
  }, { once: true });
  viewer.connection.addEventListener("error", () => {
    loading.hidden = true;
    postToPlatform({ type: "nexus-viewer-error", error: "MeshCat WebSocket 连接失败" });
  });
}

const meshcatScript = document.createElement("script");
meshcatScript.src = `${window.location.protocol}//${window.location.hostname}:${meshcatPort}/static/main.min.js`;
meshcatScript.addEventListener("load", startViewer, { once: true });
meshcatScript.addEventListener("error", () => {
  loading.hidden = true;
  postToPlatform({ type: "nexus-viewer-error", error: "MeshCat 渲染资源加载失败" });
}, { once: true });
document.head.append(meshcatScript);
