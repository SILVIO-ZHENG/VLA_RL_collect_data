// Deterministic browser-lifecycle tests. No HTTP, robot, or model connection.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const root = path.resolve(__dirname, '..');

function fixture() {
  let now = 0, nextId = 0;
  const timers = new Map(), handlers = new Map(), elements = new Map(), images = [];
  const on = (scope) => (name, fn) => handlers.set(`${scope}:${name}`, fn);
  const element = () => ({ hidden: false, open: false, style: {}, dataset: {}, children: [],
    classList: { add() {}, remove() {}, toggle() {} }, addEventListener() {},
    querySelector() { return this; }, querySelectorAll() { return []; },
    replaceChildren(...children) { this.children = children; }, append(...children) { this.children.push(...children); },
    setAttribute() {}, close() {}, textContent: '',
  });
  const get = (key) => { if (!elements.has(key)) elements.set(key, element()); return elements.get(key); };
  const schedule = (fn, delay = 0) => { const id = ++nextId; timers.set(id, { fn, at: now + delay }); return id; };
  const advance = (duration) => {
    const end = now + duration;
    for (let count = 0; ; count++) {
      assert.ok(count < 20000, 'unbounded timer loop');
      const first = [...timers].sort((a, b) => a[1].at - b[1].at)[0];
      if (!first || first[1].at > end) break;
      now = first[1].at; timers.delete(first[0]); first[1].fn(now);
    }
    now = end;
  };
  class Image {
    constructor() { images.push(this); }
    set src(value) { this.source = value; }
    get src() { return this.source; }
    removeAttribute() { this.source = ''; }
    decode() { return Promise.resolve(); }
  }
  const document = { hidden: false, addEventListener: on('document'), querySelector: get,
    getElementById: get, querySelectorAll: () => [], createElement: element };
  const window = { location: {search: '', hostname: 'localhost', origin: 'http://localhost'},
    addEventListener: on('window'), setTimeout: schedule, clearTimeout: (id) => timers.delete(id),
    localStorage: {getItem: () => null}, devicePixelRatio: 1 };
  window.parent = window;
  const context = vm.createContext({document, window, location: {protocol: 'http:', host: 'localhost'},
    URLSearchParams, URL, Image, AbortController, DOMException, performance: {now: () => now},
    requestAnimationFrame: (fn) => schedule(fn, 16), cancelAnimationFrame: (id) => timers.delete(id),
    fetch: () => { throw new Error('Unexpected network access'); }, console });
  const run = (code) => vm.runInContext(code, context);
  return {context, run, advance, document, images, timers, handlers, get};
}

function viewerFixture() {
  const f = fixture();
  const source = fs.readFileSync(path.join(root, 'src/nexus_deploy/viewer/static/viewer.js'), 'utf8');
  f.run(source.slice(0, source.lastIndexOf('boot().catch(')));
  f.run(`globalThis.renderCount = 0;
    viewer = {needs_render:false, set_dirty(){this.needs_render=true;requestRender()},
      render(){renderCount++;this.needs_render=false}};
    visualJointNames=Array.from({length:24},(_,i)=>String(i));
    poseStart=Array(24).fill(0); poseTarget=[...poseStart]; robotRoot={};
    globalThis.joints=Array(24).fill(0); joints[7]=.025;joints[15]=.025;`);
  return f;
}

test('identical 8 Hz joints stop rendering after one finite interpolation', () => {
  const f = viewerFixture();
  f.run('joints[0]=.2');
  for (let i=0; i<80; i++) { f.run('updateMeasuredPose(joints)'); f.advance(125); }
  assert.ok(f.run('renderCount') <= 5);
  assert.equal(f.timers.size, 0);
});

test('moving pose is frame capped; hidden/offscreen renders nothing and resumes latest pose', () => {
  const f = viewerFixture();
  for (let i=0; i<40; i++) { f.run(`joints[0]=${i/100};updateMeasuredPose(joints)`); f.advance(125); }
  assert.ok(f.run('renderCount') <= 101);
  const count = f.run('renderCount');
  f.document.hidden = true;
  f.handlers.get('document:visibilitychange')();
  f.run('joints[0]=.9;updateMeasuredPose(joints)'); f.advance(10000);
  assert.equal(f.run('renderCount'), count);
  assert.equal(f.timers.size, 0);
  f.document.hidden = false;
  f.handlers.get('document:visibilitychange')(); f.advance(200);
  assert.equal(f.run('displayedPose[0]'), .9);
  assert.equal(f.run('renderCount'), count+1);
  f.run('parentVisible=false;syncDisplayActivity();joints[0]=1;updateMeasuredPose(joints)');
  f.advance(10000);
  assert.equal(f.run('renderCount'), count+1);
});

test('collision visual latch stays active for one second and renews while unsafe', () => {
  const f = fixture();
  const source = fs.readFileSync(path.join(root, 'src/nexus_deploy/viewer/static/viewer.js'), 'utf8');
  const start = source.indexOf('function latchedCollisionLevel');
  const end = source.indexOf('\nfunction updateCollisionOverlay', start);
  f.run('const COLLISION_VISUAL_HOLD_MS=1000;let collisionVisualLatch=null;');
  f.run(source.slice(start, end));
  assert.equal(f.run('latchedCollisionLevel("danger", 0)'), 'danger');
  assert.equal(f.run('latchedCollisionLevel("safe", 999)'), 'danger');
  assert.equal(f.run('latchedCollisionLevel("danger", 900)'), 'danger');
  assert.equal(f.run('latchedCollisionLevel("safe", 1800)'), 'danger');
  assert.equal(f.run('latchedCollisionLevel("safe", 1901)'), 'safe');
  assert.equal(f.run('latchedCollisionLevel("warning", 2000)'), 'warning');
  assert.equal(f.run('latchedCollisionLevel("safe", 2999)'), 'warning');
  assert.equal(f.run('latchedCollisionLevel("safe", 3000)'), 'safe');
});

test('session collision banner remains visible while telemetry stays unsafe', () => {
  const f = fixture();
  const source = fs.readFileSync(path.join(root, 'src/nexus_deploy/static/session.js'), 'utf8');
  const start = source.indexOf('function updateSafetyNotice');
  const end = source.indexOf('\nfunction renderTwin', start);
  f.run(`let noticeNow=0;Date.now=()=>noticeNow;
    const SAFETY_NOTICE_DURATION_MS={collision:2000,warning:1200};
    const state={safetyNotice:null,safetyNoticeSequence:null,safetyHazardKey:null};
    const dom={collision_banner:document.querySelector('banner'),
      collision_title:document.querySelector('title'),collision_detail:document.querySelector('detail')};`);
  f.run(source.slice(start, end));
  const call = (collision) => f.run(`updateSafetyNotice({sequence:1,safety:{reason:'unsafe'},
    collision:${collision},proximityWarning:false,intervention:false,distanceMm:0,
    distanceLabel:'0 mm',hardMarginMm:3,warningMarginMm:5})`);
  call(true);
  assert.equal(f.run('state.safetyNotice.expiresAt'), 2000);
  f.run('noticeNow=1900'); call(true);
  assert.equal(f.run('state.safetyNotice.expiresAt'), 3900);
  f.run('noticeNow=3000'); call(true);
  assert.equal(f.get('banner').hidden, false);
  assert.equal(f.run('state.safetyNotice.expiresAt'), 5000);
  f.run('noticeNow=4001'); call(false);
  assert.equal(f.get('banner').hidden, false);
  f.run('noticeNow=5001'); call(false);
  assert.equal(f.get('banner').hidden, true);
});

function alignmentFixture() {
  const f = fixture();
  const sockets = [];
  class WebSocket {
    static CONNECTING = 0; static OPEN = 1; static CLOSED = 3;
    constructor() { sockets.push(this); this.readyState=1; }
    close() { this.readyState=3; this.onclose?.(); }
  }
  f.context.WebSocket=WebSocket;
  const source = fs.readFileSync(path.join(root, 'src/nexus_deploy/alignment/static/alignment.js'), 'utf8');
  f.run(source.replace(/\nbootstrap\(\);\s*$/, ''));
  f.run('state.session={id:"offline-test",state:"ADJUSTING"};globalThis.drawCount=0;'
    + 'render=()=>drawCount++;refreshFrame=()=>{}');
  return {...f, sockets};
}

test('alignment repeated connect and stale close cannot start a reconnect storm', () => {
  const f = alignmentFixture();
  for(let i=0; i<10; i++) f.run('connectEvents()');
  assert.equal(f.sockets.length, 1);
  const staleClose = f.sockets[0].onclose;
  f.sockets[0].close(); f.advance(1000);
  assert.equal(f.sockets.length, 2);
  staleClose(); f.advance(10000);
  assert.equal(f.sockets.length, 2);
  f.run('reset()'); f.advance(10000);
  assert.equal(f.timers.size, 0);
  assert.equal(f.sockets[1].onclose, null);
});

test('alignment coalesces updates at 4 Hz and keeps only latest hidden snapshot', () => {
  const f = alignmentFixture();
  f.run('connectEvents()');
  for(let i=0;i<80;i++) {
    f.sockets[0].onmessage({data:JSON.stringify({id:'offline-test',state:'ADJUSTING',frame_sequence:i})});
    f.advance(120);
  }
  assert.ok(f.run('drawCount') <= 40);
  f.document.hidden=true; f.handlers.get('document:visibilitychange')();
  const count=f.run('drawCount');
  for(let i=80;i<160;i++) {
    f.sockets[0].onmessage({data:JSON.stringify({id:'offline-test',state:'ADJUSTING',frame_sequence:i})});
    f.advance(120);
  }
  assert.equal(f.run('drawCount'),count);
  assert.equal(f.sockets.length,1); // Session stays connected; no stop API.
  f.document.hidden=false;f.handlers.get('document:visibilitychange')();f.advance(260);
  assert.equal(f.run('state.session.frame_sequence'),159);
  assert.equal(f.run('drawCount'),count+1);
});

test('alignment session replacement closes old socket without reconnecting it', () => {
  const f = alignmentFixture();
  f.run('connectEvents()');
  const oldClose = f.sockets[0].onclose;
  f.run('state.session={id:"new-session",state:"ADJUSTING"};connectEvents()');
  assert.equal(f.sockets.length, 2);
  assert.equal(f.sockets[0].readyState, 3);
  oldClose();f.advance(10000);
  assert.equal(f.sockets.length, 2);
});

test('slow inference images use one batch and coalesce to latest request', async () => {
  const f=fixture();
  f.run('const state={destroyed:false,camerasVisible:true,cameraSwapGeneration:0,cameraBatch:null};'
    + 'const dom={camera_grid:document.querySelector("grid")};');
  const source=fs.readFileSync(path.join(root,'src/nexus_deploy/static/session.js'),'utf8');
  f.run(source.slice(source.indexOf('function preloadInferenceImage'), source.indexOf('function renderMetricCells')));
  for(let i=0;i<100;i++) f.run(`scheduleInferenceFrameSwap([0,1,2].map(id=>({id,frame_id:"${i}",image_url:"/api/frame/${i}/"+id})))`);
  assert.equal(f.images.length,3);
  for(const image of [...f.images]) image.onload();
  await new Promise(setImmediate);
  assert.equal(f.images.length,6); // Skip 98 intermediate batches.
  assert.ok(f.images[3].src.includes('/99/'));
  f.run('cancelInferenceImages()');await new Promise(setImmediate);
  assert.equal(f.run('state.cameraBatch'),null);
  assert.ok(f.images.slice(3).every(image=>!image.src && image.onload===null));
  assert.equal(f.timers.size,0);
});

test('inference timeout releases pending batch and hidden page loads nothing', async () => {
  const f=fixture();
  f.run('const state={destroyed:false,camerasVisible:true,cameraSwapGeneration:0,cameraBatch:null};const dom={};');
  const source=fs.readFileSync(path.join(root,'src/nexus_deploy/static/session.js'),'utf8');
  f.run(source.slice(source.indexOf('function preloadInferenceImage'), source.indexOf('function renderMetricCells')));
  f.run('scheduleInferenceFrameSwap([{id:0,frame_id:1,image_url:"/api/slow"}])');
  f.advance(6000);await new Promise(setImmediate);
  assert.equal(f.run('state.cameraBatch'),null);
  assert.ok(!f.images[0].src);
  f.document.hidden=true;
  f.run('scheduleInferenceFrameSwap([{id:0,frame_id:2,image_url:"/api/hidden"}])');
  assert.equal(f.images.length,1);
});
