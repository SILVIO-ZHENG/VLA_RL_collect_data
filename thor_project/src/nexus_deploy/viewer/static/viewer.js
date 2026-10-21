const params = new URLSearchParams(window.location.search);
const meshcatPort = Number(params.get("meshcat_port")) || 8767;
const INTERPOLATION_MS = 150;
const COLLISION_VISUAL_HOLD_MS = 1000;
const SCENE_Z = { near: -0.70, overview: -0.86 };

let platformOrigin = "";
try {
  const requestedOrigin = params.get("platform_origin");
  const origin = requestedOrigin ? new URL(requestedOrigin) : null;
  if (["http:", "https:"].includes(origin?.protocol)
      && origin.hostname === window.location.hostname) {
    platformOrigin = origin.origin;
  }
} catch {
  platformOrigin = "";
}

const pane = document.querySelector("#meshcat-pane");
const loading = document.querySelector("#viewer-loading");
let viewer = null;
let robotRoot = null;
let surfaceRoot = null;
let kinematicJoints = [];
let visualJointNames = [];
let poseStart = [];
let poseTarget = [];
let poseStartedAt = 0;
let poseFinishedAt = 0;
let poseAnimating = false;
let displayedPose = null;
let renderTimer = null;
let renderFrame = 0;
let rendering = false;
let lastRenderAt = -Infinity;
let parentVisible = true;
let suspended = false;
let destroyed = false;
let renderFps = 20;
let softwareRenderer = false;
let rendererMode = "cached-browser-fk";
let collisionCapsules = [];
let collisionLine = null;
let collisionOverlayVisible = true;
let collisionOverlayConfig = null;
let latestTelemetryStale = true;
let collisionVisualLatch = null;

function displayActive() {
  return !destroyed && !suspended && !document.hidden && parentVisible;
}

function cancelRender() {
  if (renderTimer !== null) window.clearTimeout(renderTimer);
  if (renderFrame) cancelAnimationFrame(renderFrame);
  renderTimer = null;
  renderFrame = 0;
}

// Own the only render loop. MeshCat's default never-ending RAF is disabled.
function requestRender() {
  if (!viewer || !displayActive() || rendering || renderTimer !== null || renderFrame) return;
  const delay = Math.max(0, 1000 / renderFps - (performance.now() - lastRenderAt));
  renderTimer = window.setTimeout(() => {
    renderTimer = null;
    if (displayActive()) renderFrame = requestAnimationFrame(renderTick);
  }, delay);
}

function renderTick() {
  renderFrame = 0;
  if (!displayActive() || !viewer) return;
  const now = performance.now();
  if (now - lastRenderAt < 1000 / renderFps - 0.5) { requestRender(); return; }
  rendering = true;
  try {
    if (poseAnimating) animatePose(now);
    if (viewer.needs_render) {
      viewer.render();
      lastRenderAt = now;
    }
  } finally {
    rendering = false;
  }
  if (poseAnimating || viewer.needs_render) requestRender();
}

function createViewer() {
  if (viewer) return;
  viewer = new MeshCat.Viewer(pane, false);
  const gl = viewer.renderer.getContext();
  const info = gl.getExtension("WEBGL_debug_renderer_info");
  const rendererName = info ? String(gl.getParameter(info.UNMASKED_RENDERER_WEBGL)) : "";
  softwareRenderer = /swiftshader|llvmpipe|softpipe|software/i.test(rendererName);
  renderFps = softwareRenderer ? 8 : 20;
  viewer.renderer.shadowMap.enabled = false;
  viewer.set_dirty = () => { viewer.needs_render = true; requestRender(); };
  const resize = viewer.set_3d_pane_size.bind(viewer);
  viewer.set_3d_pane_size = () => {
    const pixels = Math.max(1, pane.clientWidth * pane.clientHeight);
    const budget = softwareRenderer ? 400000 : 1500000;
    viewer.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1,
      softwareRenderer ? 1 : 1.25, Math.sqrt(budget / pixels)));
    resize();
  };
  viewer.set_3d_pane_size();
}

function releaseCachedScene() {
  const geometries = new Set();
  const materials = new Set();
  for (const root of [robotRoot, surfaceRoot]) {
    root?.traverse((object) => {
      if (object.geometry) geometries.add(object.geometry);
      for (const material of [object.material].flat().filter(Boolean)) materials.add(material);
    });
    if (root) viewer?.scene.remove(root);
  }
  geometries.forEach((geometry) => geometry.dispose());
  materials.forEach((material) => material.dispose());
  robotRoot = surfaceRoot = null;
  kinematicJoints = [];
  collisionCapsules = [];
  collisionLine = null;
  displayedPose = null;
  poseAnimating = false;
}

function syncDisplayActivity() {
  if (!displayActive()) { cancelRender(); return; }
  // Resume at the latest pose; never replay a backlog of hidden animation.
  if (robotRoot) {
    poseAnimating = false;
    poseStart = [...poseTarget];
    displayJointValues(poseTarget);
  }
  viewer?.set_dirty();
}

document.addEventListener("visibilitychange", syncDisplayActivity);
window.addEventListener("pagehide", (event) => {
  suspended = true;
  cancelRender();
  if (!event.persisted) {
    destroyed = true;
    viewer?.connection?.close();
    releaseCachedScene();
    viewer?.controls?.dispose();
    viewer?.renderer.dispose();
  }
});
window.addEventListener("pageshow", () => { suspended = false; syncDisplayActivity(); });

function postToPlatform(message) {
  if (window.parent !== window && platformOrigin) {
    window.parent.postMessage(message, platformOrigin);
  }
}

function finiteVector(value, length) {
  return Array.isArray(value) && value.length === length
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

function setSceneView(view) {
  const offset = SCENE_Z[view] ?? SCENE_Z.overview;
  if (robotRoot) robotRoot.position.z = offset;
  if (surfaceRoot) surfaceRoot.position.z = offset;
  viewer?.set_dirty();
}

function applyCamera(camera, view) {
  if (!viewer?.camera || !viewer?.controls) return false;
  if (!finiteVector(camera?.position, 3) || !finiteVector(camera?.target, 3)) return false;
  setSceneView(view);
  viewer.camera.position.fromArray(camera.position.map(Number));
  if (finiteVector(camera.quaternion, 4)) {
    viewer.camera.quaternion.fromArray(camera.quaternion.map(Number));
  }
  viewer.controls.target.fromArray(camera.target.map(Number));
  const fov = Number(camera.fov);
  const zoom = Number(camera.zoom || 1);
  if (Number.isFinite(fov) && fov >= 10 && fov <= 120) viewer.camera.fov = fov;
  if (Number.isFinite(zoom) && zoom > 0 && zoom <= 20) viewer.camera.zoom = zoom;
  viewer.camera.updateProjectionMatrix();
  viewer.controls.update();
  viewer.set_dirty();
  return true;
}

function parseGlb(buffer) {
  const view = new DataView(buffer);
  if (view.byteLength < 20 || view.getUint32(0, true) !== 0x46546c67) {
    throw new Error("G2 静态模型不是有效 GLB");
  }
  if (view.getUint32(4, true) !== 2 || view.getUint32(8, true) !== view.byteLength) {
    throw new Error("G2 静态模型版本或长度无效");
  }
  let offset = 12;
  let gltf = null;
  let binary = null;
  while (offset + 8 <= view.byteLength) {
    const length = view.getUint32(offset, true);
    const type = view.getUint32(offset + 4, true);
    const start = offset + 8;
    const end = start + length;
    if (end > view.byteLength) throw new Error("G2 静态模型 chunk 越界");
    if (type === 0x4e4f534a) {
      const text = new TextDecoder().decode(new Uint8Array(buffer, start, length)).trim();
      gltf = JSON.parse(text);
    } else if (type === 0x004e4942) {
      binary = buffer.slice(start, end);
    }
    offset = end;
  }
  if (!gltf || !binary) throw new Error("G2 静态模型缺少 JSON 或 BIN chunk");
  return { gltf, binary };
}

function accessorArray(gltf, binary, accessorIndex) {
  const accessor = gltf.accessors?.[accessorIndex];
  const bufferView = gltf.bufferViews?.[accessor?.bufferView];
  if (!accessor || !bufferView || accessor.byteOffset) {
    throw new Error("G2 静态模型 accessor 不受支持");
  }
  const sizes = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };
  const components = {
    5121: Uint8Array,
    5123: Uint16Array,
    5125: Uint32Array,
    5126: Float32Array,
  };
  const Component = components[accessor.componentType];
  const width = sizes[accessor.type];
  if (!Component || !width) throw new Error("G2 静态模型 accessor 类型不受支持");
  return new Component(binary, Number(bufferView.byteOffset || 0), accessor.count * width);
}

function createMaterials(gltf) {
  return (gltf.materials || []).map((item) => {
    const factor = item.pbrMetallicRoughness?.baseColorFactor || [0.72, 0.75, 0.8, 1];
    return new MeshCat.THREE.MeshPhongMaterial({
      color: new MeshCat.THREE.Color(Number(factor[0]), Number(factor[1]), Number(factor[2])),
      shininess: 38,
      specular: 0x5a5a5a,
      reflectivity: 0.35,
    });
  });
}

function buildCachedScene(gltf, binary) {
  const nexus = gltf.asset?.extras?.nexus;
  if (nexus?.schema_version !== 1 || nexus?.robot !== "g2a" || !nexus?.display_only) {
    throw new Error("G2 静态模型 Nexus 元数据无效");
  }
  const materials = createMaterials(gltf);
  const meshObjects = (gltf.meshes || []).map((meshDefinition) => {
    const primitive = meshDefinition.primitives?.[0];
    if (!primitive || primitive.mode !== 4) throw new Error("G2 静态模型 primitive 无效");
    const positions = accessorArray(gltf, binary, primitive.attributes.POSITION);
    const indices = accessorArray(gltf, binary, primitive.indices);
    const geometry = new MeshCat.THREE.BufferGeometry();
    geometry.setAttribute("position", new MeshCat.THREE.BufferAttribute(positions, 3));
    geometry.setIndex(new MeshCat.THREE.BufferAttribute(indices, 1));
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    const mesh = new MeshCat.THREE.Mesh(geometry, materials[primitive.material]);
    mesh.name = meshDefinition.name || "geometry";
    mesh.frustumCulled = true;
    return mesh;
  });

  const objects = (gltf.nodes || []).map((node) => {
    const object = Number.isInteger(node.mesh)
      ? meshObjects[node.mesh]
      : new MeshCat.THREE.Group();
    object.name = node.name || "node";
    if (finiteVector(node.matrix, 16)) {
      new MeshCat.THREE.Matrix4().fromArray(node.matrix).decompose(
        object.position, object.quaternion, object.scale,
      );
    }
    return object;
  });
  (gltf.nodes || []).forEach((node, index) => {
    (node.children || []).forEach((child) => objects[index].add(objects[child]));
  });

  robotRoot = new MeshCat.THREE.Group();
  robotRoot.name = "robot";
  const scene = gltf.scenes?.[gltf.scene || 0];
  (scene?.nodes || []).forEach((index) => robotRoot.add(objects[index]));
  viewer.scene.add(robotRoot);

  const objectByName = new Map(objects.map((object) => [object.name, object]));
  visualJointNames = [...nexus.visual_joint_names];
  poseStart = visualJointNames.map(() => 0);
  poseTarget = [...poseStart];
  kinematicJoints = nexus.joints.map((joint) => ({
    ...joint,
    object: objectByName.get(joint.child),
    originMatrix: new MeshCat.THREE.Matrix4().fromArray(joint.origin),
    motionMatrix: new MeshCat.THREE.Matrix4(),
    resultMatrix: new MeshCat.THREE.Matrix4(),
    axisVector: new MeshCat.THREE.Vector3(...joint.axis).normalize(),
  }));
  if (visualJointNames.length !== 24 || kinematicJoints.some((joint) => !joint.object)) {
    throw new Error("G2 静态模型关节树不完整");
  }
  return nexus;
}

function addWorkSurface(workSurface = {}) {
  const height = Number.isFinite(Number(workSurface.height)) ? Number(workSurface.height) : 0.75;
  const hardMargin = Number.isFinite(Number(workSurface.hard_margin))
    ? Number(workSurface.hard_margin)
    : 0.004;
  const originValues = Array.isArray(workSurface.origin) && workSurface.origin.length === 3
    ? workSurface.origin.map(Number)
    : [0, 0, height];
  const normalValues = Array.isArray(workSurface.normal) && workSurface.normal.length === 3
    ? workSurface.normal.map(Number)
    : [0, 0, 1];
  const origin = new MeshCat.THREE.Vector3(...originValues);
  const normal = new MeshCat.THREE.Vector3(...normalValues).normalize();
  const bounds = Array.isArray(workSurface.xy_bounds) && workSurface.xy_bounds.length === 4
    ? workSurface.xy_bounds.map(Number)
    : [0.3, 1.0, -0.8, 0.8];
  const surfaceSizeX = bounds[1] - bounds[0];
  const surfaceSizeY = bounds[3] - bounds[2];
  const centerX = (bounds[0] + bounds[1]) * 0.5;
  const centerY = (bounds[2] + bounds[3]) * 0.5;
  const centerZ = origin.z
    - (normal.x * (centerX - origin.x) + normal.y * (centerY - origin.y)) / normal.z;
  const surfaceCenter = new MeshCat.THREE.Vector3(centerX, centerY, centerZ);
  const surfaceRotation = new MeshCat.THREE.Quaternion().setFromUnitVectors(
    new MeshCat.THREE.Vector3(0, 0, 1), normal,
  );
  surfaceRoot = new MeshCat.THREE.Group();
  surfaceRoot.name = "work_surface";
  const table = new MeshCat.THREE.Mesh(
    new MeshCat.THREE.BoxGeometry(surfaceSizeX, surfaceSizeY, 0.04),
    new MeshCat.THREE.MeshLambertMaterial({ color: 0x8f7968 }),
  );
  table.position.copy(surfaceCenter).addScaledVector(normal, -0.02);
  table.quaternion.copy(surfaceRotation);
  const margin = new MeshCat.THREE.Mesh(
    new MeshCat.THREE.BoxGeometry(surfaceSizeX, surfaceSizeY, hardMargin),
    new MeshCat.THREE.MeshLambertMaterial({
      color: 0xe08a34, transparent: true, opacity: 0.28, depthWrite: false,
    }),
  );
  margin.position.copy(surfaceCenter).addScaledVector(normal, hardMargin * 0.5 + 0.001);
  margin.quaternion.copy(surfaceRotation);
  surfaceRoot.add(table, margin);
  viewer.scene.add(surfaceRoot);
}

function configureViewer() {
  viewer.set_property(["Background"], "top_color", [0.96, 0.97, 0.99]);
  viewer.set_property(["Background"], "bottom_color", [0.88, 0.91, 0.95]);
  viewer.set_property(["Grid"], "visible", false);
  viewer.set_property(["Axes"], "visible", false);
  viewer.set_property(["Lights", "AmbientLight"], "intensity", 0.82);
  viewer.set_property(["Lights", "PointLightNegativeX"], "intensity", 0.58);
  viewer.set_property(["Lights", "PointLightPositiveX"], "intensity", 0.58);
  viewer.set_property(["Lights", "FillLight"], "intensity", 0.5);
  setSceneView("overview");
}

function displayJointValues(values) {
  if (displayedPose && values.every((value, index) => value === displayedPose[index])) return;
  displayedPose = [...values];
  const primary = new Map(visualJointNames.map((name, index) => [name, values[index] || 0]));
  for (const joint of kinematicJoints) {
    let value = primary.get(joint.name) || 0;
    if (joint.mimic) {
      value = (primary.get(joint.mimic.joint) || 0) * Number(joint.mimic.multiplier)
        + Number(joint.mimic.offset);
    }
    const motion = joint.motionMatrix;
    if (["revolute", "continuous"].includes(joint.type)) {
      motion.makeRotationAxis(joint.axisVector, value);
    } else if (joint.type === "prismatic") {
      motion.makeTranslation(
        joint.axisVector.x * value,
        joint.axisVector.y * value,
        joint.axisVector.z * value,
      );
    } else {
      motion.identity();
    }
    joint.resultMatrix.copy(joint.originMatrix).multiply(motion).decompose(
      joint.object.position, joint.object.quaternion, joint.object.scale,
    );
  }
  updateCollisionOverlay();
  viewer.set_dirty();
}

function capsuleObject(definition, side, frameObject) {
  const start = new MeshCat.THREE.Vector3(...definition.start.map(Number));
  const end = new MeshCat.THREE.Vector3(...definition.end.map(Number));
  const radius = Number(definition.radius);
  const direction = end.clone().sub(start);
  const length = direction.length();
  const material = new MeshCat.THREE.MeshBasicMaterial({
    color: 0x25a36f, transparent: true, opacity: 0.18, depthWrite: false,
  });
  const group = new MeshCat.THREE.Group();
  group.name = `collision_${definition.name}`;
  const cylinder = new MeshCat.THREE.Mesh(
    new MeshCat.THREE.CylinderGeometry(radius, radius, length, 14, 1, true), material,
  );
  cylinder.position.copy(start).add(end).multiplyScalar(0.5);
  cylinder.quaternion.setFromUnitVectors(
    new MeshCat.THREE.Vector3(0, 1, 0), direction.clone().normalize(),
  );
  const first = new MeshCat.THREE.Mesh(
    new MeshCat.THREE.SphereGeometry(radius, 14, 8), material,
  );
  const second = new MeshCat.THREE.Mesh(
    new MeshCat.THREE.SphereGeometry(radius, 14, 8), material,
  );
  first.position.copy(start); second.position.copy(end);
  group.add(cylinder, first, second);
  frameObject.add(group);
  return { definition, side, frameObject, group, material, start, end, radius };
}

function installCollisionOverlay(config, objectByName) {
  collisionOverlayConfig = config && [
    "cross-chain-capsules-v1",
    "cross-chain-capsules-v2",
  ].includes(config.geometry) ? config : null;
  if (!collisionOverlayConfig) return;
  collisionCapsules = [];
  for (const [side, key] of [["left", "left_capsules"], ["right", "right_capsules"]]) {
    for (const definition of collisionOverlayConfig[key] || []) {
      const frameObject = objectByName.get(definition.frame);
      if (frameObject) collisionCapsules.push(capsuleObject(definition, side, frameObject));
    }
  }
  const geometry = new MeshCat.THREE.BufferGeometry().setFromPoints([
    new MeshCat.THREE.Vector3(), new MeshCat.THREE.Vector3(),
  ]);
  collisionLine = new MeshCat.THREE.Line(
    geometry,
    new MeshCat.THREE.LineBasicMaterial({ color: 0x25a36f, transparent: true, opacity: 0.9 }),
  );
  collisionLine.name = "collision_closest_pair";
  robotRoot.add(collisionLine);
  setCollisionOverlayVisible(collisionOverlayVisible);
}

function closestSegmentPoints(firstStart, firstEnd, secondStart, secondEnd) {
  const first = firstEnd.clone().sub(firstStart);
  const second = secondEnd.clone().sub(secondStart);
  const offset = firstStart.clone().sub(secondStart);
  const aa = first.dot(first); const bb = first.dot(second); const cc = second.dot(second);
  const dd = first.dot(offset); const ee = second.dot(offset);
  const determinant = aa * cc - bb * bb;
  let firstParameter = 0; let secondParameter = 0;
  if (aa < 1e-12 && cc < 1e-12) {
    // Both capsules collapse to spheres.
  } else if (aa < 1e-12) {
    secondParameter = Math.max(0, Math.min(1, ee / cc));
  } else if (cc < 1e-12) {
    firstParameter = Math.max(0, Math.min(1, -dd / aa));
  } else if (determinant < 1e-12) {
    secondParameter = Math.max(0, Math.min(1, ee / cc));
  } else {
    firstParameter = Math.max(0, Math.min(1, (bb * ee - cc * dd) / determinant));
    secondParameter = Math.max(0, Math.min(1, (aa * ee - bb * dd) / determinant));
    firstParameter = Math.max(0, Math.min(1, (bb * secondParameter - dd) / aa));
    secondParameter = Math.max(0, Math.min(1, (bb * firstParameter + ee) / cc));
  }
  const firstPoint = firstStart.clone().addScaledVector(first, firstParameter);
  const secondPoint = secondStart.clone().addScaledVector(second, secondParameter);
  return { firstPoint, secondPoint, centerDistance: firstPoint.distanceTo(secondPoint) };
}

function latchedCollisionLevel(rawLevel, now = performance.now()) {
  if (rawLevel === "danger") {
    collisionVisualLatch = { level: rawLevel, expiresAt: now + COLLISION_VISUAL_HOLD_MS };
    return rawLevel;
  }
  if (rawLevel === "warning") {
    if (collisionVisualLatch?.level === "danger" && now < collisionVisualLatch.expiresAt) {
      return "danger";
    }
    collisionVisualLatch = { level: rawLevel, expiresAt: now + COLLISION_VISUAL_HOLD_MS };
    return rawLevel;
  }
  if (collisionVisualLatch && now < collisionVisualLatch.expiresAt) {
    return collisionVisualLatch.level;
  }
  collisionVisualLatch = null;
  return "safe";
}

function updateCollisionOverlay() {
  if (!robotRoot || !collisionOverlayConfig || !collisionCapsules.length) return;
  robotRoot.updateMatrixWorld(true);
  const left = collisionCapsules.filter((item) => item.side === "left");
  const right = collisionCapsules.filter((item) => item.side === "right");
  let closest = null;
  for (const first of left) for (const second of right) {
    const firstStart = first.start.clone().applyMatrix4(first.frameObject.matrixWorld);
    const firstEnd = first.end.clone().applyMatrix4(first.frameObject.matrixWorld);
    const secondStart = second.start.clone().applyMatrix4(second.frameObject.matrixWorld);
    const secondEnd = second.end.clone().applyMatrix4(second.frameObject.matrixWorld);
    const points = closestSegmentPoints(firstStart, firstEnd, secondStart, secondEnd);
    const separation = points.centerDistance - first.radius - second.radius;
    if (!closest || separation < closest.separation) closest = {
      first, second, separation, ...points,
    };
  }
  if (!closest) return;
  const hardMargin = Number(collisionOverlayConfig.hard_margin);
  const warningMargin = Number(collisionOverlayConfig.warning_margin);
  const rawLevel = closest.separation < hardMargin ? "danger"
    : closest.separation < warningMargin ? "warning" : "safe";
  const level = latestTelemetryStale ? "safe" : latchedCollisionLevel(rawLevel);
  const colors = { safe: 0x25a36f, warning: 0xe0a12f, danger: 0xd94040 };
  for (const item of collisionCapsules) {
    const limiting = item === closest.first || item === closest.second;
    item.material.color.setHex(limiting ? colors[level] : 0x25a36f);
    item.material.opacity = limiting ? 0.48 : 0.14;
    item.group.visible = collisionOverlayVisible;
  }
  if (collisionLine) {
    const firstLocal = robotRoot.worldToLocal(closest.firstPoint.clone());
    const secondLocal = robotRoot.worldToLocal(closest.secondPoint.clone());
    collisionLine.geometry.setFromPoints([firstLocal, secondLocal]);
    collisionLine.material.color.setHex(colors[level]);
    collisionLine.visible = collisionOverlayVisible;
  }
}

function setCollisionOverlayVisible(visible) {
  collisionOverlayVisible = visible === true;
  collisionCapsules.forEach((item) => { item.group.visible = collisionOverlayVisible; });
  if (collisionLine) collisionLine.visible = collisionOverlayVisible;
  viewer?.set_dirty();
}

function interpolatedPose(now) {
  if (now >= poseFinishedAt) return [...poseTarget];
  const raw = Math.max(0, Math.min(1, (now - poseStartedAt) / INTERPOLATION_MS));
  const amount = raw * raw * (3 - 2 * raw);
  return poseStart.map((value, index) => value + (poseTarget[index] - value) * amount);
}

function animatePose(now) {
  displayJointValues(interpolatedPose(now));
  poseAnimating = now < poseFinishedAt;
}

function updateMeasuredPose(rawValues, effectorProfile = "dahuan") {
  if (rendererMode !== "cached-browser-fk" || !finiteVector(rawValues, 24)) return;
  const values = rawValues.map(Number);
  for (const index of [7, 15]) {
    if (effectorProfile === "dahuan" || effectorProfile === "dh") {
      values[index] = Math.max(0, Math.min(0.025, 0.025 - values[index]));
    } else {
      const legacy = Math.max(-Math.PI / 4, Math.min(0, values[index]));
      values[index] = -legacy * 0.025 / (Math.PI / 4);
    }
  }
  // Display-only deadband; robot commands and collision inputs are untouched.
  if (values.every((value, index) => Math.abs(value - poseTarget[index])
      <= ([7, 15].includes(index) ? 0.00001 : 0.0001))) {
    updateCollisionOverlay();
    return;
  }
  const now = performance.now();
  poseStart = interpolatedPose(now);
  poseTarget = values;
  poseStartedAt = now;
  poseFinishedAt = now + INTERPOLATION_MS;
  poseAnimating = true;
  requestRender();
}

async function startCachedViewer(modelUrl, workSurface, collisionOverlay) {
  createViewer();
  const response = await fetch(modelUrl, { cache: "force-cache", credentials: "same-origin" });
  if (!response.ok) throw new Error(`G2 静态模型请求失败：HTTP ${response.status}`);
  const { gltf, binary } = parseGlb(await response.arrayBuffer());
  const nexus = buildCachedScene(gltf, binary);
  const objectByName = new Map();
  robotRoot.traverse((object) => { if (object.name) objectByName.set(object.name, object); });
  installCollisionOverlay(collisionOverlay, objectByName);
  addWorkSurface(workSurface);
  configureViewer();
  displayJointValues(poseTarget);
  loading.hidden = true;
  postToPlatform({
    type: "nexus-viewer-ready",
    renderer: rendererMode,
    renderFaces: nexus.render_face_count,
    cacheable: true,
    renderFps,
    softwareRenderer,
  });
}

function startMeshcatFallback(error) {
  rendererMode = "meshcat-websocket-fallback";
  cancelRender();
  releaseCachedScene();
  // Reuse the renderer, controls and resize listener instead of leaking a
  // second Viewer when cached asset loading fails.
  createViewer();
  viewer.connect(`ws://${window.location.hostname}:${meshcatPort}`);
  viewer.connection.addEventListener("open", () => {
    window.setTimeout(() => {
      loading.hidden = true;
      postToPlatform({
        type: "nexus-viewer-ready",
        renderer: rendererMode,
        cacheable: false,
        fallbackReason: String(error?.message || error),
      });
    }, 180);
  }, { once: true });
  viewer.connection.addEventListener("error", () => {
    loading.hidden = true;
    postToPlatform({
      type: "nexus-viewer-error",
      error: `缓存模型和 MeshCat 回退均失败：${String(error?.message || error)}`,
    });
  }, { once: true });
}

window.addEventListener("message", (event) => {
  if (!platformOrigin || event.origin !== platformOrigin || event.source !== window.parent) return;
  const message = event.data;
  if (!message || typeof message !== "object") return;
  if (message.type === "nexus-viewer-visibility") {
    const visible = message.visible === true;
    if (parentVisible !== visible) { parentVisible = visible; syncDisplayActivity(); }
  } else if (message.type === "nexus-camera-capture") {
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
    applyCamera(message.camera, message.view);
  } else if (message.type === "nexus-measured-joints") {
    latestTelemetryStale = message.stale === true;
    updateMeasuredPose(message.joints, message.effectorProfile);
  } else if (message.type === "nexus-collision-overlay") {
    setCollisionOverlayVisible(message.visible === true);
  }
});

async function boot() {
  const response = await fetch("/api/bootstrap", { cache: "no-store", credentials: "same-origin" });
  if (!response.ok) throw new Error(`三维渲染配置请求失败：HTTP ${response.status}`);
  const result = await response.json();
  const bootstrap = result?.data;
  const config = bootstrap?.viewer;
  if (typeof config?.model_url !== "string" || typeof config?.renderer_url !== "string") {
    throw new Error("三维渲染配置缺少带指纹的静态资源 URL");
  }
  const script = document.createElement("script");
  script.src = config.renderer_url;
  await new Promise((resolve, reject) => {
    script.addEventListener("load", resolve, { once: true });
    script.addEventListener("error", () => reject(new Error("内置三维渲染器加载失败")), { once: true });
    document.head.append(script);
  });
  await startCachedViewer(config.model_url, bootstrap.work_surface, bootstrap.collision_overlay).catch(startMeshcatFallback);
}

boot().catch((error) => {
  loading.hidden = true;
  postToPlatform({ type: "nexus-viewer-error", error: String(error?.message || error) });
});
