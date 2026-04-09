/**
 * engine.js — Lunar Simulation Engine (Pure WebGL2 + WebXR)
 *
 * KTX2/UASTC texture loading via Basis Universal WASM transcoder,
 * tangent-space normal mapping, Lambertian diffuse lighting.
 */

// ═══════════════════════════════════════════════════════════════
// 1. CONFIGURATION
// ═══════════════════════════════════════════════════════════════
const CFG = {
    sphereRadius: 1.0,
    sphereSegments: 64,
    spherePos: [0.0, 0.0, -3.0],
    sunDir: [0.6, 0.2, -0.5],      // normalized in shader — dramatic terminator
    assets: 'assets/',
    transcoderPath: 'lib/',
};

// ═══════════════════════════════════════════════════════════════
// 2. MATH UTILITIES
// ═══════════════════════════════════════════════════════════════
function modelMatrix(time, tx, ty, tz) {
    const a = time * 0.00015;
    const c = Math.cos(a), s = Math.sin(a);
    return new Float32Array([c,0,s,0, 0,1,0,0, -s,0,c,0, tx,ty,tz,1]);
}

function perspectiveMatrix(fov, aspect, near, far) {
    const f = 1 / Math.tan(fov / 2), m = new Float32Array(16);
    m[0] = f / aspect; m[5] = f;
    m[10] = (far + near) / (near - far); m[11] = -1;
    m[14] = (2 * far * near) / (near - far);
    return m;
}

function lookAtMatrix(eye, target, up) {
    let zx = eye[0] - target[0];
    let zy = eye[1] - target[1];
    let zz = eye[2] - target[2];
    const zLen = Math.hypot(zx, zy, zz) || 1.0;
    zx /= zLen; zy /= zLen; zz /= zLen;

    let xx = up[1] * zz - up[2] * zy;
    let xy = up[2] * zx - up[0] * zz;
    let xz = up[0] * zy - up[1] * zx;
    const xLen = Math.hypot(xx, xy, xz) || 1.0;
    xx /= xLen; xy /= xLen; xz /= xLen;

    const yx = zy * xz - zz * xy;
    const yy = zz * xx - zx * xz;
    const yz = zx * xy - zy * xx;

    return new Float32Array([
        xx, yx, zx, 0,
        xy, yy, zy, 0,
        xz, yz, zz, 0,
        -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
        -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
        -(zx * eye[0] + zy * eye[1] + zz * eye[2]),
        1,
    ]);
}

const IDENTITY = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);

/** Matches perspectiveMatrix: point sprites scale with perspective + viewport (space / satellite motion). */
function starPointScaleForViewport(viewportHeight, projYScale) {
    return (viewportHeight * 0.5) * projYScale;
}

// ═══════════════════════════════════════════════════════════════
// 3. SPHERE GEOMETRY (positions, normals, UVs, tangents)
// ═══════════════════════════════════════════════════════════════
function createSphere(radius, seg) {
    const pos = [], nrm = [], uv = [], tan = [], idx = [];
    for (let lat = 0; lat <= seg; lat++) {
        const th = (lat * Math.PI) / seg;
        const sinTh = Math.sin(th), cosTh = Math.cos(th);
        for (let lon = 0; lon <= seg; lon++) {
            const ph = (lon * 2 * Math.PI) / seg;
            const sinPh = Math.sin(ph), cosPh = Math.cos(ph);
            const nx = cosPh * sinTh, ny = cosTh, nz = sinPh * sinTh;
            pos.push(radius * nx, radius * ny, radius * nz);
            nrm.push(nx, ny, nz);
            uv.push(lon / seg, lat / seg);
            // Tangent = dP/dphi normalized = (-sinPhi, 0, cosPhi)
            tan.push(sinTh > 0.001 ? -sinPh : 1, 0, sinTh > 0.001 ? cosPh : 0, 1.0);
        }
    }
    for (let lat = 0; lat < seg; lat++)
        for (let lon = 0; lon < seg; lon++) {
            const a = lat * (seg + 1) + lon, b = a + seg + 1;
            idx.push(a, b, a + 1, b, b + 1, a + 1);
        }
    return {
        positions: new Float32Array(pos), normals: new Float32Array(nrm),
        uvs: new Float32Array(uv), tangents: new Float32Array(tan),
        indices: new Uint16Array(idx), indexCount: idx.length,
    };
}

// ═══════════════════════════════════════════════════════════════
// 4. SHADER SOURCES
// ═══════════════════════════════════════════════════════════════
const VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;
layout(location=2) in vec2 aUV;
layout(location=3) in vec4 aTan;

uniform mat4 uModel, uView, uProj;

out vec2 vUV;
out vec3 vWorldPos;
out mat3 vTBN;

void main(){
    vec4 wp = uModel * vec4(aPos,1.0);
    vWorldPos = wp.xyz;
    vUV = aUV;

    mat3 nm = mat3(uModel);
    vec3 N = normalize(nm * aNorm);
    vec3 T = normalize(nm * aTan.xyz);
    T = normalize(T - dot(T,N)*N);           // Gram-Schmidt
    vec3 B = cross(N,T) * aTan.w;            // handedness

    vTBN = mat3(T, B, N);
    gl_Position = uProj * uView * wp;
}`;

const FS = `#version 300 es
precision highp float;

in vec2 vUV;
in vec3 vWorldPos;
in mat3 vTBN;

uniform sampler2D uAlbedo;
uniform sampler2D uNormal;
uniform vec3  uSunDir;
uniform float uTexReady;      // 1.0 = textured, 0.0 = fallback grid

out vec4 oColor;

void main(){
    vec3 albedo; vec3 N;

    if(uTexReady > 0.5){
        albedo = texture(uAlbedo, vUV).rgb;
        vec3 nTS = texture(uNormal, vUV).rgb * 2.0 - 1.0;
        N = normalize(vTBN * nTS);
    } else {
        albedo = vec3(0.65,0.68,0.72);
        N = normalize(vTBN[2]);
        // wireframe grid for depth reference
        vec3 sp = vWorldPos*6.0;
        vec3 gd = abs(fract(sp)-0.5);
        float g = step(0.47, max(gd.x, max(gd.y, gd.z)));
        albedo = mix(albedo, vec3(0.1,0.75,0.9), g);
    }

    // Lambertian diffuse
    float NdotL = max(dot(N, uSunDir), 0.0);
    vec3 c = albedo * (0.025 + NdotL);

    // Reinhard tonemap + gamma
    c = c / (c + 1.0);
    c = pow(c, vec3(1.0/2.2));

    oColor = vec4(c, 1.0);
}`;

const VS_STAR = `#version 300 es
precision highp float;
layout(location=0) in vec4 aStarData; // x=RA, y=Dec, z=Mag, w=Kelvin

uniform mat4 uModel, uView, uProj;
// (viewport height) / (2 * tan(fovY/2)) — perspective-correct point diameter in pixels
uniform float uPointScale;

out float vIntensity;
out vec3  vStarColor;

// Attempt 6500K-normalized blackbody → RGB (Tanner Helland approx)
vec3 kelvinToRGB(float K) {
    float t = K / 100.0;
    float r, g, b;
    if (t <= 66.0) {
        r = 1.0;
        g = clamp((99.4708 * log(t) - 161.1196) / 255.0, 0.0, 1.0);
    } else {
        r = clamp(329.698727 * pow(t - 60.0, -0.1332047592) / 255.0, 0.0, 1.0);
        g = clamp(288.1221695 * pow(t - 60.0, -0.0755148492) / 255.0, 0.0, 1.0);
    }
    if (t >= 66.0) {
        b = 1.0;
    } else if (t <= 19.0) {
        b = 0.0;
    } else {
        b = clamp((138.5177 * log(t - 10.0) - 305.0448) / 255.0, 0.0, 1.0);
    }
    return vec3(r, g, b);
}

void main(){
    float ra  = aStarData.x;
    float dec = aStarData.y;
    float mag = aStarData.z;
    float K   = aStarData.w;

    // Distant inertial sky: fixed RA/Dec on a large sphere (satellite / space POV).
    float radius = 50.0;
    float x = radius * cos(dec) * cos(ra);
    float y = radius * sin(dec);
    float z = radius * cos(dec) * sin(ra);

    vec4 wp = uModel * vec4(x, y, z, 1.0);
    vec4 clip = uProj * uView * wp;
    gl_Position = clip;

    // Pogson law: flux ∝ 10^(-0.4 m); relative brightness vs ~mag 4 reference
    float m = clamp(mag, -1.5, 8.0);
    float flux = pow(10.0, -0.4 * (m - 4.0));
    float intensity = clamp(flux * 0.62, 0.04, 1.0);

    float w = max(abs(clip.w), 1e-4);
    // Unresolved point sources: 1–2 screen pixels max (no Airy disk / spikes in this pass).
    float px = intensity > 0.7 ? 2.0 : 1.0;
    gl_PointSize = clamp(px * uPointScale / w, 1.0, 2.0);

    vIntensity = intensity;
    // Human vision in this scene: subtle temperature tint, mostly near-white stars.
    vec3 color = kelvinToRGB(clamp(K, 2000.0, 40000.0));
    vStarColor = mix(vec3(1.0), color, 0.35);
}`;

const FS_STAR = `#version 300 es
precision highp float;

in float vIntensity;
in vec3  vStarColor;
out vec4 oColor;

void main(){
    // Real appearance: unresolved disk — a single sharp dot (no bloom, spikes, or PSF).
    float d = length(gl_PointCoord - vec2(0.5));
    if (d > 0.5) discard;

    float L = min(vIntensity * 1.05, 1.0);
    oColor = vec4(vStarColor * L, 1.0);
}`;

// ── RA/Dec string → radians conversion ──────────────────────
// RA format from Yale BSC JSON: "00h 05m 09.9s"
function parseRA(str) {
    const m = str.match(/([\d.]+)h\s*([\d.]+)m\s*([\d.]+)s/);
    if (!m) return 0;
    const hours = parseFloat(m[1]) + parseFloat(m[2]) / 60 + parseFloat(m[3]) / 3600;
    return hours * (Math.PI / 12);  // hours → radians (24h = 2π)
}

// Dec format from Yale BSC JSON: "+45° 13′ 45″"  or  "-05° 42′ 27″"
function parseDec(str) {
    const m = str.match(/([+-]?)(\d+)°\s*(\d+)′\s*([\d.]+)″/);
    if (!m) return 0;
    const sign = m[1] === '-' ? -1 : 1;
    const deg = parseFloat(m[2]) + parseFloat(m[3]) / 60 + parseFloat(m[4]) / 3600;
    return sign * deg * (Math.PI / 180);  // degrees → radians
}

// Mock fallback (used while the catalog is loading)
function generateMockStars(count) {
    const data = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) {
        const u = Math.random(), v = Math.random();
        data[i * 4]     = 2 * Math.PI * u;                          // RA
        data[i * 4 + 1] = Math.asin(1.0 - 2.0 * v);                // Dec
        data[i * 4 + 2] = -1.5 + Math.pow(Math.random(), 3) * 7.5; // Mag
        data[i * 4 + 3] = 3000 + Math.random() * 27000;             // Kelvin
    }
    return data;
}

// ── Yale Bright Star Catalog loader ─────────────────────────
const BSC_URL = 'https://raw.githubusercontent.com/brettonw/YaleBrightStarCatalog/master/bsc5-short.json';

async function fetchYaleBSC() {
    log('↓ Fetching Yale Bright Star Catalog…');
    try {
        const res = await fetch(BSC_URL);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const catalog = await res.json();

        // Filter to entries that have all required fields
        const valid = catalog.filter(s => s.RA && s.Dec && s.V !== undefined);

        const data = new Float32Array(valid.length * 4);
        for (let i = 0; i < valid.length; i++) {
            data[i * 4]     = parseRA(valid[i].RA);
            data[i * 4 + 1] = parseDec(valid[i].Dec);
            data[i * 4 + 2] = parseFloat(valid[i].V);
            data[i * 4 + 3] = valid[i].K ? parseFloat(valid[i].K) : 6500;
        }

        // Hot-swap the GPU buffer
        gl.bindBuffer(gl.ARRAY_BUFFER, starBuf);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);
        starCount = valid.length;

        log(`✓ Yale BSC loaded — ${starCount} stars`);
    } catch (e) {
        log('⚠ BSC fetch failed, keeping mock stars: ' + e.message);
    }
}

// ═══════════════════════════════════════════════════════════════
// 5. COMPRESSED FORMAT DETECTION & KTX2 LOADING
// ═══════════════════════════════════════════════════════════════

function detectFormats(gl) {
    const f = {};
    let ext;
    if ((ext = gl.getExtension('WEBGL_compressed_texture_astc')))
        f.astc = { gl: ext.COMPRESSED_RGBA_ASTC_4x4_KHR, basis: 10, name: 'ASTC' };
    if ((ext = gl.getExtension('EXT_texture_compression_bptc')))
        f.bptc = { gl: ext.COMPRESSED_RGBA_BPTC_UNORM_EXT, basis: 6, name: 'BC7' };
    if ((ext = gl.getExtension('WEBGL_compressed_texture_s3tc')))
        f.s3tc = { gl: ext.COMPRESSED_RGBA_S3TC_DXT5_EXT, basis: 3, name: 'S3TC/DXT5' };
    return f;
}

function pickTarget(formats) {
    return formats.astc || formats.bptc || formats.s3tc || null;
}

let basisMod = null;

async function initTranscoder() {
    if (typeof window.BASIS === 'undefined') {
        log('⚠ Basis transcoder script not found – KTX2 disabled');
        return false;
    }
    try {
        basisMod = await window.BASIS({
            locateFile: (f) => CFG.transcoderPath + f,
        });
        basisMod.initializeBasis();
        log('✓ Basis WASM transcoder ready');
        return true;
    } catch (e) {
        log('✗ Transcoder init failed: ' + e.message);
        return false;
    }
}

async function loadKTX2(gl, url, target) {
    if (!basisMod) return null;
    log(`  ↓ Fetching ${url}`);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const buf = new Uint8Array(await res.arrayBuffer());

    // Use KTX2File API if available, else BasisFile
    const Cls = basisMod.KTX2File || basisMod.BasisFile;
    const isKTX2 = !!basisMod.KTX2File;
    const file = new Cls(buf);
    if (!file.isValid()) { file.close(); file.delete(); throw new Error('Invalid file'); }

    const w = isKTX2 ? file.getWidth() : file.getImageWidth(0, 0);
    const h = isKTX2 ? file.getHeight() : file.getImageHeight(0, 0);
    const levels = isKTX2 ? file.getLevels() : file.getNumLevels(0);

    if (!file.startTranscoding()) { file.close(); file.delete(); throw new Error('Transcode init failed'); }

    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);

    if (target) {
        // Upload compressed mip levels
        for (let lv = 0; lv < levels; lv++) {
            const sz = isKTX2
                ? file.getImageTranscodedSizeInBytes(lv, 0, 0, target.basis)
                : file.getImageTranscodedSizeInBytes(0, lv, target.basis);
            const dst = new Uint8Array(sz);
            const ok = isKTX2
                ? file.transcodeImage(dst, lv, 0, 0, target.basis, 0, -1, -1)
                : file.transcodeImage(dst, 0, lv, target.basis, 0, 0);
            if (!ok) { log(`  ✗ Transcode failed at mip ${lv}`); break; }
            const mw = Math.max(1, w >> lv), mh = Math.max(1, h >> lv);
            gl.compressedTexImage2D(gl.TEXTURE_2D, lv, target.gl, mw, mh, 0, dst);
        }
        log(`  ✓ ${url} → ${target.name}  ${w}×${h}  (${levels} mips)`);
    } else {
        // RGBA fallback — level 0 only
        const basisRGBA = 13; // cTFRGBA32
        const sz = isKTX2
            ? file.getImageTranscodedSizeInBytes(0, 0, 0, basisRGBA)
            : file.getImageTranscodedSizeInBytes(0, 0, basisRGBA);
        const dst = new Uint8Array(sz);
        isKTX2
            ? file.transcodeImage(dst, 0, 0, 0, basisRGBA, 0, -1, -1)
            : file.transcodeImage(dst, 0, 0, basisRGBA, 0, 0);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, dst);
        gl.generateMipmap(gl.TEXTURE_2D);
        log(`  ✓ ${url} → RGBA8 fallback  ${w}×${h}`);
    }

    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, levels > 1 && target ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    file.close(); file.delete();
    return tex;
}

// 5c. PNG fallback
function loadPNGTexture(gl, url) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            const tex = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, tex);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
            gl.generateMipmap(gl.TEXTURE_2D);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            log(`  ✓ PNG loaded: ${url}  ${img.width}×${img.height}`);
            resolve(tex);
        };
        img.onerror = () => { log(`  ✗ PNG failed: ${url}`); resolve(null); };
        img.src = url;
    });
}

// ═══════════════════════════════════════════════════════════════
// 6. ENGINE STATE
// ═══════════════════════════════════════════════════════════════
let gl, program, vao, idxCount;
let uModel, uView, uProj, uSunDir, uTexReady, uAlbedoLoc, uNormalLoc;
let albedoTex = null, normalTex = null, texturesLoaded = false;
let xrSession = null, xrRefSpace = null, raf = null;
let starProgram, starVao, starCount, starBuf, uStarModel, uStarView, uStarProj, uStarPointScale;
let audioCtx = null, humOsc = null, humGain = null;
let hudRoot = null, hudAltitudeEl = null, hudCoordsEl = null;
let uiContainer = null;
let flightModeBtn = null;
let flightActive = false;
const cameraPosition = new Float32Array([0.0, 0.0, 0.0]);
let pitch = 0.0;
let yaw = Math.PI;
let previousFrameTime = 0.0;
const keyState = {
    KeyW: false, KeyA: false, KeyS: false, KeyD: false,
    ArrowUp: false, ArrowLeft: false, ArrowDown: false, ArrowRight: false,
};
const NON_VR_FLY_SPEED = 2.0;
const MOUSE_LOOK_SENSITIVITY = 0.002;
const MAX_PITCH = (89 * Math.PI) / 180;

// Locomotion state (WebXR thumbstick flight)
const locomotion = { x: 0, y: 0, z: 0 };   // accumulated world offset
const MOVE_SPEED = 0.04;                    // metres per frame at full stick

// ── Easter Egg: Cinematic Dolly State ────────────────────────
const easterEggAudio = new Audio('thousand_years.mp3');
easterEggAudio.preload = 'auto';

const cinematic = {
    active:    false,   // true while the camera dolly is in progress
    finished:  false,   // true after the sequence has fully played
    startTime: 0,
    duration:  4000,    // 4 seconds in ms
    startPos:  new Float32Array(3),
    endPos:    new Float32Array(3),
    startYaw:  0,
    startPitch:0,
};

function log(msg) {
    console.log(msg);
    const el = document.getElementById('status-text');
    if (el) el.textContent = msg;
}

function initSpacecraftAudio() {
    if (audioCtx) return;
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;

    audioCtx = new AudioCtx();
    humOsc = audioCtx.createOscillator();
    const humLowpass = audioCtx.createBiquadFilter();
    humGain = audioCtx.createGain();

    humOsc.type = 'sine';
    humOsc.frequency.setValueAtTime(40, audioCtx.currentTime);
    humLowpass.type = 'lowpass';
    humLowpass.frequency.setValueAtTime(80, audioCtx.currentTime);
    humGain.gain.setValueAtTime(0.08, audioCtx.currentTime);

    humOsc.connect(humLowpass);
    humLowpass.connect(humGain);
    humGain.connect(audioCtx.destination);
    humOsc.start();
}

// ═══════════════════════════════════════════════════════════════
// 7. INIT
// ═══════════════════════════════════════════════════════════════
function compileShader(src, type) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        console.error(gl.getShaderInfoLog(s)); gl.deleteShader(s); return null;
    }
    return s;
}

async function initEngine() {
    const canvas = document.getElementById('gl-canvas');
    uiContainer = document.getElementById('ui-container');
    flightModeBtn = document.getElementById('flight-mode-button');
    hudRoot = document.getElementById('vr-hud');
    hudAltitudeEl = document.getElementById('hud-altitude');
    hudCoordsEl = document.getElementById('hud-coords');
    gl = canvas.getContext('webgl2', { xrCompatible: true, antialias: true });
    if (!gl) { log('WebGL2 not supported.'); return; }

    // --- Shaders ---
    const vs = compileShader(VS, gl.VERTEX_SHADER);
    const fs = compileShader(FS, gl.FRAGMENT_SHADER);
    program = gl.createProgram();
    gl.attachShader(program, vs); gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error(gl.getProgramInfoLog(program)); return;
    }

    uModel    = gl.getUniformLocation(program, 'uModel');
    uView     = gl.getUniformLocation(program, 'uView');
    uProj     = gl.getUniformLocation(program, 'uProj');
    uSunDir   = gl.getUniformLocation(program, 'uSunDir');
    uTexReady = gl.getUniformLocation(program, 'uTexReady');
    uNormalLoc = gl.getUniformLocation(program, 'uNormal');
    uAlbedoLoc = gl.getUniformLocation(program, 'uAlbedo');

    // --- Star Shaders ---
    const vsStar = compileShader(VS_STAR, gl.VERTEX_SHADER);
    const fsStar = compileShader(FS_STAR, gl.FRAGMENT_SHADER);
    starProgram = gl.createProgram();
    gl.attachShader(starProgram, vsStar); gl.attachShader(starProgram, fsStar);
    gl.linkProgram(starProgram);
    if (!gl.getProgramParameter(starProgram, gl.LINK_STATUS)) {
        console.error(gl.getProgramInfoLog(starProgram)); return;
    }

    uStarModel = gl.getUniformLocation(starProgram, 'uModel');
    uStarView = gl.getUniformLocation(starProgram, 'uView');
    uStarProj = gl.getUniformLocation(starProgram, 'uProj');
    uStarPointScale = gl.getUniformLocation(starProgram, 'uPointScale');

    // --- Geometry ---
    const sphere = createSphere(CFG.sphereRadius, CFG.sphereSegments);
    idxCount = sphere.indexCount;

    vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const mkBuf = (data, loc, size) => {
        const b = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    };
    mkBuf(sphere.positions, 0, 3);
    mkBuf(sphere.normals,   1, 3);
    mkBuf(sphere.uvs,       2, 2);
    mkBuf(sphere.tangents,  3, 4);

    const ib = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, sphere.indices, gl.STATIC_DRAW);
    gl.bindVertexArray(null);

    // --- Star Geometry (mock, replaced async by real catalog) ---
    starCount = 1000;
    const starData = generateMockStars(starCount);
    starVao = gl.createVertexArray();
    gl.bindVertexArray(starVao);
    starBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, starBuf);
    gl.bufferData(gl.ARRAY_BUFFER, starData, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.enable(gl.BLEND);

    canvas.addEventListener('click', () => {
        if (flightActive && document.pointerLockElement !== canvas) {
            canvas.requestPointerLock();
        }
    });

    document.addEventListener('pointerlockchange', () => {
        if (document.pointerLockElement !== canvas && flightActive) {
            flightActive = false;
            if (uiContainer) uiContainer.style.display = 'flex';
            if (hudRoot) hudRoot.style.display = 'none';
            document.getElementById('dpad-container').style.display = 'none';
        }
    });

    document.addEventListener('mousemove', (e) => {
        if (document.pointerLockElement !== canvas || xrSession) return;
        if (cinematic.active || cinematic.finished) return;
        yaw -= e.movementX * MOUSE_LOOK_SENSITIVITY;
        pitch -= e.movementY * MOUSE_LOOK_SENSITIVITY;
        if (pitch > MAX_PITCH) pitch = MAX_PITCH;
        if (pitch < -MAX_PITCH) pitch = -MAX_PITCH;
    });

    const movementKeys = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowUp', 'ArrowLeft', 'ArrowDown', 'ArrowRight']);
    window.addEventListener('keydown', (e) => {
        if (!movementKeys.has(e.code)) return;
        keyState[e.code] = true;
        e.preventDefault();
    });
    window.addEventListener('keyup', (e) => {
        if (!movementKeys.has(e.code)) return;
        keyState[e.code] = false;
        e.preventDefault();
    });

    const bindDPadButton = (buttonId, keyCode) => {
        const btn = document.getElementById(buttonId);
        if (!btn) return;
        const press = (e) => {
            keyState[keyCode] = true;
            btn.classList.add('active');
            e.preventDefault();
        };
        const release = (e) => {
            keyState[keyCode] = false;
            btn.classList.remove('active');
            e.preventDefault();
        };
        btn.addEventListener('pointerdown', press);
        btn.addEventListener('pointerup', release);
        btn.addEventListener('pointercancel', release);
        btn.addEventListener('pointerleave', release);
    };
    bindDPadButton('dpad-up', 'KeyW');
    bindDPadButton('dpad-down', 'KeyS');
    bindDPadButton('dpad-left', 'KeyA');
    bindDPadButton('dpad-right', 'KeyD');

    document.getElementById('dpad-container').style.display = 'none';

    if (flightModeBtn) {
        flightModeBtn.addEventListener('click', () => {
            flightActive = true;
            previousFrameTime = 0;
            if (uiContainer) uiContainer.style.display = 'none';
            if (hudRoot) hudRoot.style.display = 'block';
            document.getElementById('dpad-container').style.display = 'grid';
            initSpacecraftAudio();
            canvas.requestPointerLock();
        });
    }

    // ── Mission Briefing Overlay Listener ────────────────────
    const missionBriefing = document.getElementById('mission-briefing');
    const engageBtn = document.getElementById('engage-button');
    if (missionBriefing && engageBtn) {
        engageBtn.addEventListener('click', () => {
            initSpacecraftAudio();
            if (window.matchMedia("(pointer: fine)").matches) {
                canvas.requestPointerLock();
                flightActive = true;
            } else {
                flightActive = true;
                document.getElementById('dpad-container').style.display = 'grid';
            }
            missionBriefing.classList.add('fade-out');
            if (uiContainer) uiContainer.style.display = 'none';
            if (hudRoot) hudRoot.style.display = 'block';
        });
    }

    // ── Easter Egg: Cmd/Ctrl + J listener ────────────────────
    window.addEventListener('keydown', (e) => {
        // Trigger only on Cmd+J (Mac) or Ctrl+J (Win/Linux)
        if (e.code === 'KeyJ' && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            if (cinematic.active || cinematic.finished) return;
            startCinematicDolly();
        }
        // Escape dismisses the overlay + stops audio after reveal
        if (e.code === 'Escape' && cinematic.finished) {
            dismissCinematic();
        }
    });

    // --- Start 2D fallback loop immediately (shows grid sphere) ---
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    raf = requestAnimationFrame(render2D);

    // --- Load textures in background ---
    loadAllTextures();

    // --- Load real star catalog in background ---
    fetchYaleBSC();

    // --- WebXR ---
    checkXR();
}

async function loadAllTextures() {
    log('Initializing texture pipeline…');
    const formats = detectFormats(gl);
    const target = pickTarget(formats);
    log(`GPU formats: ${Object.values(formats).map(f=>f.name).join(', ') || 'none (RGBA fallback)'}`);

    const transcoderOk = await initTranscoder();

    // Try KTX2 first, fall back to PNG
    if (transcoderOk) {
        try {
            albedoTex = await loadKTX2(gl, CFG.assets + 'lunar_albedo.ktx2', target);
            normalTex = await loadKTX2(gl, CFG.assets + 'lunar_normal.ktx2', target);
        } catch (e) { log('KTX2 failed, trying PNG… ' + e.message); }
    }
    if (!albedoTex) albedoTex = await loadPNGTexture(gl, CFG.assets + 'lunar_albedo.png');
    if (!normalTex) normalTex = await loadPNGTexture(gl, CFG.assets + 'lunar_normal.png');

    texturesLoaded = !!(albedoTex && normalTex);
    log(texturesLoaded ? '✓ All textures loaded — normal mapping active' : '⚠ Textures missing — using fallback');
}

// ═══════════════════════════════════════════════════════════════
// 8. WEBXR
// ═══════════════════════════════════════════════════════════════
async function checkXR() {
    const btn = document.getElementById('vr-mode-button');
    if (navigator.xr) {
        try {
            if (await navigator.xr.isSessionSupported('immersive-vr')) {
                btn.disabled = false;
                btn.addEventListener('click', enterVR);
            } else { log('Immersive VR not supported on this hardware.'); }
        } catch (e) { log('WebXR query failed.'); }
    } else { log('WebXR unavailable (need HTTPS or localhost).'); }
}

async function enterVR() {
    try {
        initSpacecraftAudio();
        xrSession = await navigator.xr.requestSession('immersive-vr', {
            optionalFeatures: ['dom-overlay'],
            domOverlay: { root: hudRoot || document.getElementById('vr-hud') },
        });
        xrSession.addEventListener('end', exitVR);
        await gl.makeXRCompatible();
        xrSession.updateRenderState({ baseLayer: new XRWebGLLayer(xrSession, gl) });
        xrRefSpace = await xrSession.requestReferenceSpace('local');
        if (uiContainer) uiContainer.style.display = 'none';
        if (hudRoot) hudRoot.style.display = 'block';
        if (raf !== null) cancelAnimationFrame(raf);
        raf = xrSession.requestAnimationFrame(renderXR);
    } catch (e) { console.error('VR session failed:', e); }
}

function exitVR() {
    xrSession = null;
    flightActive = false;
    previousFrameTime = 0;
    if (uiContainer) uiContainer.style.display = 'flex';
    if (hudRoot) hudRoot.style.display = 'none';
    document.getElementById('dpad-container').style.display = 'none';
    if (audioCtx) {
        audioCtx.close();
        audioCtx = null;
        humOsc = null;
        humGain = null;
    }
    resizeCanvas();
    raf = requestAnimationFrame(render2D);
}

// ═══════════════════════════════════════════════════════════════
// 9. RENDER HELPERS
// ═══════════════════════════════════════════════════════════════
function bindScene(time) {
    gl.useProgram(program);
    gl.bindVertexArray(vao);

    const [tx, ty, tz] = CFG.spherePos;
    gl.uniformMatrix4fv(uModel, false, modelMatrix(time, tx, ty, tz));
    gl.uniform3f(uSunDir, ...normalize3(CFG.sunDir));
    gl.uniform1f(uTexReady, texturesLoaded ? 1.0 : 0.0);

    // Bind textures
    if (texturesLoaded) {
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, albedoTex);
        gl.uniform1i(uAlbedoLoc, 0);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, normalTex);
        gl.uniform1i(uNormalLoc, 1);
    }
}

function normalize3([x, y, z]) {
    const l = Math.sqrt(x*x + y*y + z*z);
    return [x/l, y/l, z/l];
}

// ═══════════════════════════════════════════════════════════════
// 10. RENDER LOOPS
// ═══════════════════════════════════════════════════════════════
function render2D(time) {
    if (xrSession) return;
    const dt = previousFrameTime > 0 ? (time - previousFrameTime) / 1000 : 0;
    previousFrameTime = time;

    const cosPitch = Math.cos(pitch);
    const sinPitch = Math.sin(pitch);
    const cosYaw = Math.cos(yaw);
    const sinYaw = Math.sin(yaw);
    const forwardX = cosPitch * sinYaw;
    const forwardY = sinPitch;
    const forwardZ = cosPitch * cosYaw;
    const rightX = cosYaw;
    const rightY = 0;
    const rightZ = -sinYaw;

    const moveForward = (keyState.KeyW || keyState.ArrowUp ? 1 : 0) - (keyState.KeyS || keyState.ArrowDown ? 1 : 0);
    const moveRight = (keyState.KeyD || keyState.ArrowRight ? 1 : 0) - (keyState.KeyA || keyState.ArrowLeft ? 1 : 0);
    const moveLen = Math.hypot(moveForward, moveRight);
    if (moveLen > 0 && !cinematic.active && !cinematic.finished) {
        const speed = (NON_VR_FLY_SPEED * dt) / moveLen;
        cameraPosition[0] += (forwardX * moveForward + rightX * moveRight) * speed;
        cameraPosition[1] += (forwardY * moveForward + rightY * moveRight) * speed;
        cameraPosition[2] += (forwardZ * moveForward + rightZ * moveRight) * speed;
    }

    const lookTarget = [
        cameraPosition[0] + forwardX,
        cameraPosition[1] + forwardY,
        cameraPosition[2] + forwardZ,
    ];
    const viewMatrix = lookAtMatrix(cameraPosition, lookTarget, [0, 1, 0]);

    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.clearColor(0.01, 0.01, 0.02, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const proj = perspectiveMatrix(Math.PI / 3, gl.canvas.width / gl.canvas.height, 0.1, 100);

    // --- Draw Stars (opaque dots — standard blend, no additive bloom) ---
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(starProgram);
    gl.bindVertexArray(starVao);
    gl.uniformMatrix4fv(uStarProj, false, proj);
    gl.uniformMatrix4fv(uStarView, false, viewMatrix);
    gl.uniformMatrix4fv(uStarModel, false, IDENTITY);
    gl.uniform1f(uStarPointScale, starPointScaleForViewport(gl.canvas.height, proj[5]));

    gl.depthMask(false);
    gl.drawArrays(gl.POINTS, 0, starCount);
    gl.depthMask(true);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    // --- Draw Moon ---
    bindScene(time);
    gl.uniformMatrix4fv(uProj, false, proj);
    gl.uniformMatrix4fv(uView, false, viewMatrix);
    gl.drawElements(gl.TRIANGLES, idxCount, gl.UNSIGNED_SHORT, 0);

    if (flightActive) {
        const [moonX, moonY, moonZ] = CFG.spherePos;
        const adx = cameraPosition[0] - moonX;
        const ady = cameraPosition[1] - moonY;
        const adz = cameraPosition[2] - moonZ;
        const altitude = Math.sqrt(adx * adx + ady * ady + adz * adz) - CFG.sphereRadius;
        if (hudAltitudeEl) hudAltitudeEl.textContent = altitude.toFixed(2);
        if (hudCoordsEl) hudCoordsEl.textContent = `${cameraPosition[0].toFixed(2)}  ${cameraPosition[1].toFixed(2)}  ${cameraPosition[2].toFixed(2)}`;
    }

    // ── Easter Egg: Cinematic camera dolly update ─────────
    if (cinematic.active) {
        const elapsed = time - cinematic.startTime;
        const t = Math.min(elapsed / cinematic.duration, 1.0);
        // Smoothstep: 3t² − 2t³  (ease in-out)
        const s = t * t * (3.0 - 2.0 * t);

        cameraPosition[0] = cinematic.startPos[0] + (cinematic.endPos[0] - cinematic.startPos[0]) * s;
        cameraPosition[1] = cinematic.startPos[1] + (cinematic.endPos[1] - cinematic.startPos[1]) * s;
        cameraPosition[2] = cinematic.startPos[2] + (cinematic.endPos[2] - cinematic.startPos[2]) * s;

        if (t >= 1.0) {
            cinematic.active = false;
            cinematic.finished = true;
            onCinematicComplete();
        }
    }

    raf = requestAnimationFrame(render2D);
}

function renderXR(time, frame) {
    raf = frame.session.requestAnimationFrame(renderXR);

    // ── Thumbstick locomotion ────────────────────────────────
    const session = frame.session;
    for (const src of session.inputSources) {
        if (!src.gamepad) continue;
        const gp = src.gamepad;
        // Standard mapping: axes[2]=thumbstick X, axes[3]=thumbstick Y
        const ax = gp.axes.length > 2 ? gp.axes[2] : 0;
        const ay = gp.axes.length > 3 ? gp.axes[3] : 0;

        // Apply deadzone
        const dx = Math.abs(ax) > 0.15 ? ax : 0;
        const dy = Math.abs(ay) > 0.15 ? ay : 0;

        if (dx !== 0 || dy !== 0) {
            // Get the viewer orientation to move relative to gaze
            const tempPose = frame.getViewerPose(xrRefSpace);
            if (tempPose) {
                const m = tempPose.transform.matrix; // 4×4 column-major
                // Forward  = -Z column (col 2 negated)
                const fwdX = -m[8], fwdY = -m[9], fwdZ = -m[10];
                // Right    = +X column (col 0)
                const rgtX =  m[0], rgtY =  m[1], rgtZ =  m[2];

                // pushing thumbstick forward gives negative dy
                locomotion.x += (rgtX * dx - fwdX * dy) * MOVE_SPEED;
                locomotion.y += (rgtY * dx - fwdY * dy) * MOVE_SPEED;
                locomotion.z += (rgtZ * dx - fwdZ * dy) * MOVE_SPEED;
            }
        }
    }

    // Create an offset reference space that incorporates locomotion
    const offsetTransform = new XRRigidTransform(
        { x: -locomotion.x, y: -locomotion.y, z: -locomotion.z, w: 1 },
        { x: 0, y: 0, z: 0, w: 1 }
    );
    const movedRefSpace = xrRefSpace.getOffsetReferenceSpace(offsetTransform);

    const pose = frame.getViewerPose(movedRefSpace);
    if (!pose) return;

    const layer = session.renderState.baseLayer;
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    gl.clearColor(0.01, 0.01, 0.02, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    for (const view of pose.views) {
        const vp = layer.getViewport(view);
        gl.viewport(vp.x, vp.y, vp.width, vp.height);

        // --- Draw Stars (inertial sky; head / craft motion via view matrix) ---
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.useProgram(starProgram);
        gl.bindVertexArray(starVao);
        gl.uniformMatrix4fv(uStarProj, false, view.projectionMatrix);
        gl.uniformMatrix4fv(uStarView, false, view.transform.inverse.matrix);
        gl.uniformMatrix4fv(uStarModel, false, IDENTITY);
        const py = view.projectionMatrix[5];
        gl.uniform1f(uStarPointScale, starPointScaleForViewport(vp.height, py));

        gl.depthMask(false);
        gl.drawArrays(gl.POINTS, 0, starCount);
        gl.depthMask(true);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        // --- Draw Moon ---
        bindScene(time);
        gl.uniformMatrix4fv(uProj, false, view.projectionMatrix);
        gl.uniformMatrix4fv(uView, false, view.transform.inverse.matrix);
        gl.drawElements(gl.TRIANGLES, idxCount, gl.UNSIGNED_SHORT, 0);
    }

    const cameraX = pose.transform.position.x + locomotion.x;
    const cameraY = pose.transform.position.y + locomotion.y;
    const cameraZ = pose.transform.position.z + locomotion.z;
    const [moonX, moonY, moonZ] = CFG.spherePos;
    const dx = cameraX - moonX;
    const dy = cameraY - moonY;
    const dz = cameraZ - moonZ;
    const altitude = Math.sqrt(dx * dx + dy * dy + dz * dz) - CFG.sphereRadius;

    if (hudAltitudeEl) hudAltitudeEl.textContent = altitude.toFixed(2);
    if (hudCoordsEl) hudCoordsEl.textContent = `${cameraX.toFixed(2)}  ${cameraY.toFixed(2)}  ${cameraZ.toFixed(2)}`;
}

function resizeCanvas() {
    if (xrSession) return;
    const c = document.getElementById('gl-canvas');
    if (c.width !== window.innerWidth || c.height !== window.innerHeight) {
        c.width = window.innerWidth; c.height = window.innerHeight;
    }
}

// ═══════════════════════════════════════════════════════════════
// 11. EASTER EGG — CINEMATIC DOLLY + REVEAL
// ═══════════════════════════════════════════════════════════════

function startCinematicDolly() {
    // Compute the current forward look-vector from yaw/pitch
    const cosPitch = Math.cos(pitch);
    const sinPitch = Math.sin(pitch);
    const cosYaw   = Math.cos(yaw);
    const sinYaw   = Math.sin(yaw);
    const fwdX = cosPitch * sinYaw;
    const fwdY = sinPitch;
    const fwdZ = cosPitch * cosYaw;

    // Vector from camera to moon center
    const toMoonX = CFG.spherePos[0] - cameraPosition[0];
    const toMoonY = CFG.spherePos[1] - cameraPosition[1];
    const toMoonZ = CFG.spherePos[2] - cameraPosition[2];
    const distToMoon = Math.sqrt(toMoonX * toMoonX + toMoonY * toMoonY + toMoonZ * toMoonZ);

    // Travel 20% of the remaining distance along the forward look-vector
    const dollyDist = distToMoon * 0.2;

    cinematic.startPos[0] = cameraPosition[0];
    cinematic.startPos[1] = cameraPosition[1];
    cinematic.startPos[2] = cameraPosition[2];
    cinematic.endPos[0]   = cameraPosition[0] + fwdX * dollyDist;
    cinematic.endPos[1]   = cameraPosition[1] + fwdY * dollyDist;
    cinematic.endPos[2]   = cameraPosition[2] + fwdZ * dollyDist;
    cinematic.startYaw    = yaw;
    cinematic.startPitch  = pitch;

    // Freeze normal controls
    cinematic.active   = true;
    cinematic.finished = false;
    cinematic.startTime = performance.now();

    // Dim the HUD during the cinematic
    if (hudRoot) hudRoot.style.opacity = '0.3';

    // Reset all movement keys so the ship doesn't drift
    for (const k of Object.keys(keyState)) keyState[k] = false;

    log('🎬 Cinematic sequence initiated…');
}

function onCinematicComplete() {
    // Fade in the overlay
    const overlay = document.getElementById('dj-overlay');
    if (overlay) {
        // Force a reflow so the CSS animation keyframes restart cleanly
        overlay.classList.remove('visible');
        void overlay.offsetWidth;
        overlay.classList.add('visible');
    }

    // Play the audio track
    easterEggAudio.currentTime = 0;
    easterEggAudio.play().catch((err) => {
        console.warn('Easter egg audio blocked by browser autoplay policy:', err.message);
    });

    // Auto-fade text after 10 seconds (audio keeps playing)
    cinematic._dismissTimer = setTimeout(() => {
        if (cinematic.finished) {
            const ov = document.getElementById('dj-overlay');
            if (ov) ov.classList.remove('visible');
            if (hudRoot) hudRoot.style.opacity = '1';
            cinematic.finished = false;
            cinematic._dismissTimer = null;
        }
    }, 10000);

    log('🌙 The moon sings for you.');
}

function dismissCinematic() {
    // Cancel auto-dismiss timer if user pressed Escape early
    if (cinematic._dismissTimer) {
        clearTimeout(cinematic._dismissTimer);
        cinematic._dismissTimer = null;
    }

    // Fade out the overlay
    const overlay = document.getElementById('dj-overlay');
    if (overlay) overlay.classList.remove('visible');

    // Stop the audio gracefully
    easterEggAudio.pause();
    easterEggAudio.currentTime = 0;

    // Restore HUD
    if (hudRoot) hudRoot.style.opacity = '1';

    // Reset cinematic state so it can be re-triggered
    cinematic.active   = false;
    cinematic.finished = false;

    log('Ready — flight controls restored.');
}

// ═══════════════════════════════════════════════════════════════
// BOOT
// ═══════════════════════════════════════════════════════════════
window.onload = initEngine;
