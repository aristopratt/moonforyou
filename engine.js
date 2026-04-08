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
    spherePos: [0.0, 1.5, -3.0],
    sunDir: [0.5, 0.3, 0.8],       // normalized in shader
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

const IDENTITY = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);

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

function log(msg) {
    console.log(msg);
    const el = document.getElementById('status-text');
    if (el) el.textContent = msg;
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
    uAlbedoLoc = gl.getUniformLocation(program, 'uAlbedo');
    uNormalLoc = gl.getUniformLocation(program, 'uNormal');

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

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);

    // --- Start 2D fallback loop immediately (shows grid sphere) ---
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    raf = requestAnimationFrame(render2D);

    // --- Load textures in background ---
    loadAllTextures();

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
    const btn = document.getElementById('vr-button');
    if (navigator.xr) {
        try {
            if (await navigator.xr.isSessionSupported('immersive-vr')) {
                btn.style.display = 'block';
                btn.addEventListener('click', enterVR);
            } else { log('Immersive VR not supported on this hardware.'); }
        } catch (e) { log('WebXR query failed.'); }
    } else { log('WebXR unavailable (need HTTPS or localhost).'); }
}

async function enterVR() {
    try {
        xrSession = await navigator.xr.requestSession('immersive-vr');
        xrSession.addEventListener('end', exitVR);
        await gl.makeXRCompatible();
        xrSession.updateRenderState({ baseLayer: new XRWebGLLayer(xrSession, gl) });
        xrRefSpace = await xrSession.requestReferenceSpace('local');
        document.getElementById('ui-container').style.display = 'none';
        if (raf !== null) cancelAnimationFrame(raf);
        raf = xrSession.requestAnimationFrame(renderXR);
    } catch (e) { console.error('VR session failed:', e); }
}

function exitVR() {
    xrSession = null;
    document.getElementById('ui-container').style.display = 'flex';
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
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.clearColor(0.01, 0.01, 0.02, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    bindScene(time);
    gl.uniformMatrix4fv(uProj, false, perspectiveMatrix(Math.PI / 3, gl.canvas.width / gl.canvas.height, 0.1, 100));
    gl.uniformMatrix4fv(uView, false, IDENTITY);
    gl.drawElements(gl.TRIANGLES, idxCount, gl.UNSIGNED_SHORT, 0);

    raf = requestAnimationFrame(render2D);
}

function renderXR(time, frame) {
    raf = frame.session.requestAnimationFrame(renderXR);
    const pose = frame.getViewerPose(xrRefSpace);
    if (!pose) return;

    const layer = frame.session.renderState.baseLayer;
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    gl.clearColor(0.01, 0.01, 0.02, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    bindScene(time);

    for (const view of pose.views) {
        const vp = layer.getViewport(view);
        gl.viewport(vp.x, vp.y, vp.width, vp.height);
        gl.uniformMatrix4fv(uProj, false, view.projectionMatrix);
        gl.uniformMatrix4fv(uView, false, view.transform.inverse.matrix);
        gl.drawElements(gl.TRIANGLES, idxCount, gl.UNSIGNED_SHORT, 0);
    }
}

function resizeCanvas() {
    if (xrSession) return;
    const c = document.getElementById('gl-canvas');
    if (c.width !== window.innerWidth || c.height !== window.innerHeight) {
        c.width = window.innerWidth; c.height = window.innerHeight;
    }
}

// ═══════════════════════════════════════════════════════════════
// BOOT
// ═══════════════════════════════════════════════════════════════
window.onload = initEngine;
