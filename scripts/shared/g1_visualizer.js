import * as THREE from "./three.module.js";
import { GLTFLoader } from "./GLTFLoader.js";
import { OrbitControls } from "./OrbitControls.js";

const COLORS = { neutral: 0xa8b0b8, measured: 0x32d5e8, commanded: 0xf3a62f, fault: 0xff4d5e };
const MOVED_RAD = 0.005;
const ERROR_RAD = 0.02;

function affectedLinks(manifest, pose, reference) {
  const children = new Map();
  manifest.joint_hierarchy.forEach((joint) => {
    const list = children.get(joint.parent_link) || [];
    list.push(joint.child_link);
    children.set(joint.parent_link, list);
  });
  const result = new Set();
  const add = (link) => {
    if (result.has(link)) return;
    result.add(link);
    (children.get(link) || []).forEach(add);
  };
  (pose || []).forEach((value, index) => {
    if (Math.abs(value - (reference?.[index] ?? value)) < MOVED_RAD) return;
    const joint = manifest.joint_hierarchy.find((item) => item.name === manifest.joint_names?.[index]);
    if (joint) add(joint.child_link);
  });
  return result;
}

function decorate(root, color, opacity, outlineOpacity) {
  root.traverse((node) => {
    if (!node.isMesh) return;
    node.material = new THREE.MeshStandardMaterial({
      color, transparent: true, opacity, depthWrite: opacity > 0.25, roughness: 0.88,
    });
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(node.geometry, 24),
      new THREE.LineBasicMaterial({color, transparent: true, opacity: outlineOpacity}),
    );
    edges.name = "__outline";
    node.add(edges);
  });
}

function setLayerLinks(root, links) {
  root.traverse((node) => {
    // Link nodes carry descendant joints, so hiding one would also hide a moved
    // grandchild. Hide only the link's visual mesh node.
    if (node.userData?.kind === "visual") node.visible = links === null || links.has(node.userData.link);
  });
}

function applyPose(root, manifest, pose) {
  if (!Array.isArray(pose)) return;
  manifest.joint_names.forEach((name, index) => {
    const node = root.getObjectByName(name);
    if (!node) return;
    if (!node.userData.baseQuaternion) node.userData.baseQuaternion = node.quaternion.clone();
    const axis = new THREE.Vector3(...(node.userData.axis || [1, 0, 0])).normalize();
    node.quaternion.copy(node.userData.baseQuaternion).multiply(
      new THREE.Quaternion().setFromAxisAngle(axis, Number(pose[index]) || 0),
    );
  });
}

function marker(scene, name, color, shape="sphere") {
  const geometry = shape === "diamond" ? new THREE.OctahedronGeometry(0.035) : new THREE.SphereGeometry(0.025, 16, 10);
  const item = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({color}));
  item.name = name; item.visible = false; scene.add(item); return item;
}

export async function mountG1Visualizer(options) {
  const container = options.container;
  if (!container || !window.WebGLRenderingContext) throw new Error("WebGL is unavailable in this browser");
  const response = await fetch(new URL("manifest.json", import.meta.url));
  if (!response.ok) throw new Error("generated model manifest is missing");
  const manifest = await response.json();
  const modelUrl = new URL(manifest.model, import.meta.url).href;
  const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.replaceChildren(renderer.domElement);
  renderer.domElement.setAttribute("aria-label", "Interactive full-body G1 commanded versus measured view");
  renderer.domElement.tabIndex = 0;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.02, 20);
  // URDF, robot telemetry, calibration, and workspace overlays are all Z-up.
  // Keep that shared frame instead of baking a corrective rotation into 49 meshes.
  camera.up.set(0, 0, 1);
  camera.position.set(2.0, -2.2, 1.45);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0.65); controls.enableDamping = true;
  const homeCamera = new THREE.Vector3(), homeTarget = new THREE.Vector3();
  scene.add(new THREE.HemisphereLight(0xe9f5ff, 0x26313a, 2.2));
  const key = new THREE.DirectionalLight(0xffffff, 2.0); key.position.set(2, -1, 3); scene.add(key);
  const ground = new THREE.GridHelper(3, 30, 0x41505c, 0x28343d);
  ground.rotation.x = Math.PI / 2;
  scene.add(ground);
  const gltf = await new GLTFLoader().loadAsync(modelUrl);
  const referenceRoot = gltf.scene;
  const bounds = new THREE.Box3().setFromObject(referenceRoot);
  const modelCenter = bounds.getCenter(new THREE.Vector3());
  const modelSize = bounds.getSize(new THREE.Vector3());
  ground.position.z = bounds.min.z;
  homeTarget.copy(modelCenter);
  homeCamera.copy(modelCenter).add(new THREE.Vector3(modelSize.z * 1.05, -modelSize.z * 1.25, modelSize.z * 0.35));
  controls.target.copy(homeTarget); camera.position.copy(homeCamera); controls.update();
  const measuredRoot = gltf.scene.clone(true);
  const commandedRoot = gltf.scene.clone(true);
  decorate(referenceRoot, COLORS.neutral, 0.06, 0.6);
  decorate(measuredRoot, COLORS.measured, 0.10, 0.95);
  decorate(commandedRoot, COLORS.commanded, 0.20, 0.9);
  scene.add(referenceRoot, measuredRoot, commandedRoot);
  const measuredMarker = marker(scene, "Measured object", COLORS.measured);
  const targetMarker = marker(scene, "Pregrasp target", COLORS.commanded, "diamond");
  const endEffectorMarker = marker(scene, "End effector", 0xffffff, "diamond");
  let workspaceHelper = null, supportPlane = null, cameraFrustum = null;
  const trajectoryMaterial = new THREE.LineDashedMaterial({color: COLORS.commanded, dashSize: 0.035, gapSize: 0.02});
  const trajectory = new THREE.Line(new THREE.BufferGeometry(), trajectoryMaterial); scene.add(trajectory);
  const trail = [];
  const trailLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({color: COLORS.measured, transparent: true, opacity: 0.45})); scene.add(trailLine);
  let previous = null, latest = null, latestAt = 0, hidden = document.hidden;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function updateScene(state) {
    const reference = state.reference_pose_rad;
    applyPose(referenceRoot, state, reference);
    applyPose(measuredRoot, state, state.measured_pose_rad);
    applyPose(commandedRoot, state, state.commanded_pose_rad);
    setLayerLinks(referenceRoot, null);
    setLayerLinks(measuredRoot, affectedLinks({...manifest, joint_names: state.joint_names}, state.measured_pose_rad, reference));
    setLayerLinks(commandedRoot, affectedLinks({...manifest, joint_names: state.joint_names}, state.commanded_pose_rad, reference));
    measuredRoot.visible = options.measuredToggle?.checked !== false;
    commandedRoot.visible = options.commandedToggle?.checked !== false;
    referenceRoot.visible = options.referenceToggle?.checked !== false;
    const measured = state.measured_object_xyz_m;
    measuredMarker.visible = Array.isArray(measured); if (measuredMarker.visible) measuredMarker.position.fromArray(measured);
    const target = state.pregrasp_target_xyz_m;
    targetMarker.visible = Array.isArray(target); if (targetMarker.visible) targetMarker.position.fromArray(target);
    const endEffector = state.end_effector_xyz_m;
    endEffectorMarker.visible = Array.isArray(endEffector); if (endEffectorMarker.visible) endEffectorMarker.position.fromArray(endEffector);
    const points = (state.predicted_trajectory_xyz_m || []).filter(Array.isArray).map((point) => new THREE.Vector3(...point));
    trajectory.geometry.setFromPoints(points); trajectory.computeLineDistances();
    if (Array.isArray(measured)) {
      trail.push({at: performance.now(), point: new THREE.Vector3(...measured)});
      while (trail.length && performance.now() - trail[0].at > 2000) trail.shift();
      trailLine.geometry.setFromPoints(trail.map((item) => item.point));
    }
    const bounds = state.workspace_bounds;
    if (bounds?.minimum && bounds?.maximum && !workspaceHelper) {
      workspaceHelper = new THREE.Box3Helper(new THREE.Box3(new THREE.Vector3(...bounds.minimum), new THREE.Vector3(...bounds.maximum)), 0x8796a3);
      workspaceHelper.material.transparent = true; workspaceHelper.material.opacity = 0.65; scene.add(workspaceHelper);
    }
    if (state.support_plane && !supportPlane) {
      supportPlane = new THREE.Mesh(new THREE.PlaneGeometry(1.4, 1.4), new THREE.MeshBasicMaterial({color:0x8796a3, transparent:true, opacity:0.10, side:THREE.DoubleSide, wireframe:true}));
      const normal = new THREE.Vector3(...state.support_plane.normal).normalize();
      supportPlane.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1), normal);
      supportPlane.position.copy(normal.multiplyScalar(-Number(state.support_plane.offset || 0)));
      supportPlane.name = "Support plane"; scene.add(supportPlane);
    }
    if (state.camera && !cameraFrustum) {
      const intrinsics = state.camera.intrinsics, depth = 0.45;
      const corners = [[0,0],[intrinsics.width,0],[intrinsics.width,intrinsics.height],[0,intrinsics.height]].map(([u,v]) => new THREE.Vector3((u-intrinsics.ppx)/intrinsics.fx*depth,(v-intrinsics.ppy)/intrinsics.fy*depth,depth));
      const origin = new THREE.Vector3(); const vertices=[];
      corners.forEach((corner) => vertices.push(origin,corner));
      for(let i=0;i<4;i++) vertices.push(corners[i],corners[(i+1)%4]);
      cameraFrustum = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(vertices), new THREE.LineDashedMaterial({color:0xb9c2c9,dashSize:0.025,gapSize:0.015}));
      const transform = state.camera.optical_to_torso;
      if (transform?.translation) cameraFrustum.position.fromArray(transform.translation);
      if (transform?.rotation) {
        const r=transform.rotation, matrix=new THREE.Matrix4().set(r[0][0],r[0][1],r[0][2],0,r[1][0],r[1][1],r[1][2],0,r[2][0],r[2][1],r[2][2],0,0,0,0,1);
        cameraFrustum.quaternion.setFromRotationMatrix(matrix);
      }
      cameraFrustum.computeLineDistances(); cameraFrustum.name="Camera frustum"; scene.add(cameraFrustum);
    }
    measuredRoot.getObjectsByProperty("name", "__joint_ring").forEach((ring) => ring.removeFromParent());
    const selected = options.jointSelect?.value || state.selected_joint;
    const selectedIndex = state.joint_names?.indexOf(selected);
    const selectedError = selectedIndex >= 0 ? Math.abs((state.measured_pose_rad?.[selectedIndex] ?? 0) - (state.commanded_pose_rad?.[selectedIndex] ?? 0)) : 0;
    const errorJoints = (state.joint_names || []).filter((name,index) => Math.abs((state.measured_pose_rad?.[index] ?? 0)-(state.commanded_pose_rad?.[index] ?? 0)) > ERROR_RAD);
    const ringNames = new Set([...(state.faulted_joints || []), ...errorJoints, ...(selected ? [selected] : [])]);
    ringNames.forEach((name) => {
      const joint = measuredRoot.getObjectByName(name); if (!joint) return;
      const fault = (state.faulted_joints || []).includes(name) || errorJoints.includes(name) || (name === selected && selectedError > ERROR_RAD);
      const ring = new THREE.Mesh(new THREE.TorusGeometry(0.045,0.006,8,24),new THREE.MeshBasicMaterial({color:fault ? COLORS.fault : 0xffffff}));
      ring.name="__joint_ring"; ring.rotation.x=Math.PI/2; joint.add(ring);
    });
    options.onState?.(state);
  }

  async function poll() {
    try {
      const stateUrl = typeof options.stateUrl === "function" ? options.stateUrl() : options.stateUrl;
      const stateResponse = await fetch(stateUrl, {cache: "no-store", headers: options.headers?.() || {}});
      if (!stateResponse.ok) throw new Error(`state HTTP ${stateResponse.status}`);
      const payload = await stateResponse.json();
      const state = payload.visualization || payload.bridge?.visualization || payload;
      state.joint_names ||= manifest.joint_hierarchy.filter((joint) => joint.type !== "fixed").map((joint) => joint.name);
      previous = latest; latest = state; latestAt = performance.now(); updateScene(state);
      options.onAvailability?.(state.available ? (state.fresh ? "live" : "stale") : "unavailable");
    } catch (error) { options.onAvailability?.("unavailable", error); }
  }
  const interval = window.setInterval(poll, 100);
  poll();
  function resize() {
    const width = Math.max(container.clientWidth, 1), height = Math.max(container.clientHeight, 1);
    renderer.setSize(width, height, false); camera.aspect = width / height; camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(container); resize();
  document.addEventListener("visibilitychange", () => { hidden = document.hidden; });
  renderer.domElement.addEventListener("keydown", (event) => {
    if (event.key.toLowerCase() === "r") { camera.position.copy(homeCamera); controls.target.copy(homeTarget); controls.update(); }
  });
  options.resetButton?.addEventListener("click", () => { camera.position.copy(homeCamera); controls.target.copy(homeTarget); controls.update(); });
  [options.referenceToggle, options.measuredToggle, options.commandedToggle].forEach((item) => item?.addEventListener("change", () => latest && updateScene(latest)));
  options.jointSelect?.addEventListener("change", () => latest && updateScene(latest));
  function frame(now) {
    requestAnimationFrame(frame); if (hidden) return;
    if (!reduced && previous && latest && now - latestAt < 100) {
      const alpha = Math.min(1, (now - latestAt) / 100);
      const mix = (a, b) => Array.isArray(a) && Array.isArray(b) ? a.map((value, i) => value + ((b[i] ?? value) - value) * alpha) : b;
      applyPose(measuredRoot, latest, mix(previous.measured_pose_rad, latest.measured_pose_rad));
      applyPose(commandedRoot, latest, mix(previous.commanded_pose_rad, latest.commanded_pose_rad));
    }
    controls.update(); renderer.render(scene, camera);
  }
  requestAnimationFrame(frame);
  return {destroy() { clearInterval(interval); renderer.dispose(); }};
}

export { MOVED_RAD, ERROR_RAD };
