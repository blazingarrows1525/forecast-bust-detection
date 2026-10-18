// Forecast Bust Intelligence: the landing's night orbit.
//
// India raised on the model's own 0.25-degree grid over a plotting globe. One
// column per grid cell; each cell carries its subdivision's served reading for
// one issue date and one lead day. Every height comes from /api/risk-cube and
// every cell's subdivision from /api/voxel-grid (largest overlap wins, cells
// are never blended). Nothing is interpolated between cells or lead days: a
// lead change swaps to the archive's own values for that day.
//
// The globe is a plotting sphere (ocean, graticule, limb). It carries no other
// land because the product has no data for any other land.
//
// Frames are drawn only while something moves (the intro, a drag, a hover
// change); there is no free-running loop. Without WebGL the same cells are
// drawn flat on a 2-D canvas, with the same hover, links and lead rail.
"use strict";

window.FBIOrbit = (() => {
  const D2R = Math.PI / 180;
  const R = 10;                 // globe radius, scene units
  const H0 = 0.035;             // the land plate every cell stands on
  const H_PER_P = 0.85;         // column height per unit of probability
  const H_REFUSED = H0 + 0.35 * H_PER_P;   // a fixed height, hatched: unknown, never low
  // Cells meet edge to edge: a visible gap between thousands of thin columns
  // aliases into vertical moire at landing-page scale. The grid still shows,
  // as the stair-stepped subdivision rims.
  const GAP = 0.995;
  const BANDS = [0.05, 0.10, 0.20, 0.35];
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const code = (L) => "D" + String(L).padStart(2, "0");
  const pct = (v) => (v * 100).toFixed(1) + "%";
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const bandOf = (p) => (p < BANDS[0] ? 0 : p < BANDS[1] ? 1 : p < BANDS[2] ? 2 : p < BANDS[3] ? 3 : 4);

  // ------------------------------------------------------------ the data
  // The grid decoded once; per lead, one reading per grid region.
  function prepare(grid) {
    const bin = atob(grid.ids_b64), ids = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) ids[i] = bin.charCodeAt(i);
    const cells = [], at = new Int32Array(grid.ny * grid.nx).fill(-1);
    for (let iy = 0; iy < grid.ny; iy++) {
      for (let ix = 0; ix < grid.nx; ix++) {
        const v = ids[iy * grid.nx + ix];
        if (v === grid.nodata) continue;
        at[iy * grid.nx + ix] = cells.length;
        cells.push({ iy, ix, ri: v, lat: grid.lat0 + iy * grid.dlat, lon: grid.lon0 + ix * grid.dlon });
      }
    }
    // Region edges on the grid: where a cell meets a different subdivision
    // (inner) or nothing at all (coast). Stair-stepped on purpose: they are
    // the model's cells, not a smoothed polygon.
    const edges = [];
    const cellAt = (iy, ix) => (iy < 0 || ix < 0 || iy >= grid.ny || ix >= grid.nx) ? -1 : at[iy * grid.nx + ix];
    cells.forEach((c, k) => {
      const hl = grid.dlat / 2, hw = grid.dlon / 2;
      const sides = [
        [0, 1, [c.lat - hl, c.lon + hw], [c.lat + hl, c.lon + hw]],   // east
        [1, 0, [c.lat + hl, c.lon - hw], [c.lat + hl, c.lon + hw]],   // north
        [0, -1, [c.lat - hl, c.lon - hw], [c.lat + hl, c.lon - hw]],  // west
        [-1, 0, [c.lat - hl, c.lon - hw], [c.lat - hl, c.lon + hw]],  // south
      ];
      sides.forEach(([dy, dx, a, b], s) => {
        const n = cellAt(c.iy + dy, c.ix + dx);
        if (n >= 0 && cells[n].ri === c.ri) return;
        // Inner edges are listed once (east and north); coast edges always.
        if (n >= 0 && s >= 2) return;
        edges.push({ k, n, a, b });
      });
    });
    const rings = (grid.outline || []).map((flat) => {
      const pts = [];
      for (let i = 0; i + 1 < flat.length; i += 2) pts.push([flat[i + 1], flat[i]]);
      return pts;
    }).sort((p, q) => q.length - p.length);
    return { grid, cells, at, edges, rings, regionIds: grid.region_ids };
  }

  // Reading per grid region for one lead: the archive's values, or an
  // explicit state when there are none. Null stays null; it is never zero.
  function readings(data, cube, lead) {
    const i = cube.lead_days.indexOf(lead);
    const byId = new Map(cube.regions.map((r) => [r.region_id, r]));
    return data.regionIds.map((rid) => {
      const r = byId.get(rid);
      if (!r || i < 0) return { rid, name: null, state: "none" };
      const status = r.status[i], p = r.p[i];
      const base = { rid, name: r.region, lead, init: cube.init_date, valid: r.valid_date[i],
        baseline: r.baseline[i], actual: r.actual[i], lon: r.lon, lat: r.lat };
      if (status === "OK" && typeof p === "number") return { ...base, state: "scored", p };
      if (status === "OUT_OF_DISTRIBUTION") return { ...base, state: "refused" };
      return { ...base, state: "none" };
    });
  }

  const heightOf = (rd) => rd.state === "scored" ? H0 + rd.p * H_PER_P
    : rd.state === "refused" ? H_REFUSED : H0;

  // The station each subdivision is plotted at: its cell nearest the
  // representative point the API serves, so the plot sits inside the region.
  function stationCells(data, cube) {
    const byId = new Map(cube.regions.map((r) => [r.region_id, r]));
    const best = new Map();
    data.cells.forEach((c, k) => {
      const r = byId.get(data.regionIds[c.ri]);
      if (!r || !isFinite(r.lon) || !isFinite(r.lat)) return;
      const d = (c.lon - r.lon) ** 2 + (c.lat - r.lat) ** 2;
      const b = best.get(c.ri);
      if (!b || d < b.d) best.set(c.ri, { k, d });
    });
    return best;
  }

  // Ranked by served reading, highest first; refusals rank above every
  // scored reading, because unknown is elevated, never quiet.
  function ranked(rds) {
    const key = (r) => r.state === "refused" ? 2 : r.state === "scored" ? r.p : -1;
    return rds.map((r, ri) => ({ ri, r })).filter((x) => x.r.state !== "none")
      .sort((a, b) => key(b.r) - key(a.r));
  }

  function qualityTier(webgl) {
    const asked = new URLSearchParams(location.search).get("quality");
    if (["high", "balanced", "low", "2d"].includes(asked)) return asked;
    if (!webgl) return "2d";
    const cores = navigator.hardwareConcurrency || 8, mem = navigator.deviceMemory || 8;
    if (cores <= 4 || mem <= 4) return "low";
    if (Math.min(screen.width, screen.height) < 520) return "balanced";
    return "high";
  }
  const TIERS = {
    high: { dpr: 2, aa: true, stars: 700, seg: [128, 96] },
    balanced: { dpr: 1.5, aa: true, stars: 380, seg: [96, 72] },
    low: { dpr: 1, aa: false, stars: 0, seg: [64, 48] },
  };

  function webglAvailable() {
    try {
      const c = document.createElement("canvas");
      return !!(window.WebGLRenderingContext && (c.getContext("webgl2") || c.getContext("webgl")));
    } catch (e) { return false; }
  }

  // ------------------------------------------------------- station plots
  // A plotted station, as on a synoptic chart: the mark at the column's top,
  // the served reading upper left, the baseline lower left, the lead upper
  // right, the subdivision lower right. Each links into the command centre on
  // exactly that reading.
  function stationHTML(rd, href) {
    const band = rd.state === "scored" ? bandOf(rd.p) : null;
    const mark = rd.state === "refused"
      ? `<span class="stn-mark hatch" aria-hidden="true"></span>`
      : `<span class="stn-mark" style="background:var(--c${band})" aria-hidden="true"></span>`;
    const served = rd.state === "refused" ? "refused" : pct(rd.p);
    const label = rd.state === "refused"
      ? `${rd.name}, ${code(rd.lead)}: refused, not scored. Open in the command centre.`
      : `${rd.name}, ${code(rd.lead)}: ${pct(rd.p)} served, baseline ${pct(rd.baseline)}. Open in the command centre.`;
    const obs = rd.actual === 1 ? "busted" : rd.actual === 0 ? "held" : "not observed";
    return `<a class="stn" href="${href}" aria-label="${esc(label)} Observed: ${obs}.">` +
      `<span class="stn-served sign">${served}</span>${mark}<span class="stn-lead sign">${code(rd.lead)}</span>` +
      `<span class="stn-base sign">${rd.state === "refused" ? "no reading" : "base " + pct(rd.baseline)}</span>` +
      `<span class="stn-obs sign${rd.actual === 1 ? " bust" : ""}">${obs}</span>` +
      `<span class="stn-name">${esc(rd.name)}</span></a>`;
  }

  // Station plots go in the void above the land, never over it: a row in
  // west-to-east order of their stations, each tied to its station by a leader.
  // Without room above, they stack beside the land; failing that, around the
  // station itself.
  function placeInVoid(items, land, w, h, boxW, boxH, top, minX = 8, maxN = 3) {
    const gap = 12;
    // As many plots as the free band holds, highest readings first. A station
    // deep in the south would need a leader across the whole land to reach a
    // plot above it, so it yields its place to the next reading instead.
    const room = Math.min(maxN, Math.max(1, Math.floor((w - 8 - minX + gap) / (boxW + gap))));
    const reach = land ? (it) => it.y - land.y0 <= 0.4 * (land.y1 - land.y0) : () => true;
    const order = items.filter(reach).slice(0, room).sort((a, b) => a.x - b.x);
    if (land && land.y0 - boxH - 26 >= top) {
      const y = land.y0 - boxH - 26;
      let xs = order.map((it) => Math.min(w - boxW - 8, Math.max(minX, it.x - boxW / 2)));
      for (let i = 1; i < xs.length; i++) xs[i] = Math.max(xs[i], xs[i - 1] + boxW + gap);
      const over = xs.length ? xs[xs.length - 1] + boxW + 8 - w : 0;
      if (over > 0) xs = xs.map((x) => Math.max(minX, x - over));
      return order.map((it, i) => ({ ...it, box: { x: xs[i], y, w: boxW, h: boxH } }));
    }
    if (land && land.x1 + 24 + boxW <= w - 8) {
      let y = Math.max(top, land.y0);
      return order.sort((a, b) => a.y - b.y).map((it) => {
        const box = { x: land.x1 + 24, y: Math.max(y, it.y - boxH / 2), w: boxW, h: boxH };
        y = box.y + boxH + gap;
        return box.y + boxH > h - 8 ? { ...it, box: null } : { ...it, box };
      });
    }
    return placeLabels(items, w, h, boxW, boxH);
  }

  // Greedy placement: each label tries four quadrants around its station and
  // takes the first that overlaps nothing already placed and stays on screen.
  function placeLabels(items, w, h, boxW, boxH) {
    const placed = [];
    const hit = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
    return items.map((it) => {
      const tries = [[18, -boxH - 14], [18, 16], [-boxW - 18, -boxH - 14], [-boxW - 18, 16]];
      for (const [dx, dy] of tries) {
        const box = { x: it.x + dx, y: it.y + dy, w: boxW, h: boxH };
        if (box.x < 6 || box.y < 6 || box.x + box.w > w - 6 || box.y + box.h > h - 6) continue;
        if (placed.some((p) => hit(p, box))) continue;
        placed.push(box);
        return { ...it, box };
      }
      return { ...it, box: null };
    });
  }

  // A leader from the station to the nearest point of its plot's frame, and
  // the station itself as a plotted ring.
  function leaderSVG(it, boxW, boxH) {
    const pin = `<circle cx="${it.x.toFixed(1)}" cy="${it.y.toFixed(1)}" r="4" class="pin"/>`;
    if (!it.box) return pin;
    const bx = Math.min(Math.max(it.x, it.box.x + 8), it.box.x + boxW - 8);
    const by = it.box.y + boxH < it.y ? it.box.y + boxH : it.box.y > it.y ? it.box.y : it.y;
    return `<path d="M${it.x.toFixed(1)} ${it.y.toFixed(1)} L${bx.toFixed(1)} ${by.toFixed(1)}" class="leader"/>` + pin;
  }

  // ======================================================== WebGL scene
  function mountGL(host, data, cube, opts) {
    const THREE = window.THREE;
    const tier = TIERS[opts.tier] || TIERS.high;
    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: tier.aa, powerPreference: "high-performance" });
    } catch (e) { return null; }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, tier.dpr));
    const canvas = renderer.domElement;
    canvas.className = "orbit-canvas";
    host.prepend(canvas);

    const col = (n) => new THREE.Color(css(n));
    const scene = new THREE.Scene();
    scene.background = col("--ground");
    const camera = new THREE.PerspectiveCamera(30, 1, 0.05, 600);

    const dir = (lat, lon) => new THREE.Vector3(
      Math.cos(lat * D2R) * Math.cos(lon * D2R), Math.sin(lat * D2R), -Math.cos(lat * D2R) * Math.sin(lon * D2R));
    const east = (lon) => new THREE.Vector3(-Math.sin(lon * D2R), 0, -Math.cos(lon * D2R));

    // India's middle, and the frame the camera works in around it.
    const C_LAT = 22.6, C_LON = 82.4;
    const T = dir(C_LAT, C_LON).multiplyScalar(R);
    const UP = dir(C_LAT, C_LON), EAST = east(C_LON), NORTH = new THREE.Vector3().crossVectors(UP, EAST);
    // One low key light from the south-west, above the horizon.
    const SUN = UP.clone().multiplyScalar(0.62).addScaledVector(NORTH, -0.55).addScaledVector(EAST, -0.48).normalize();

    // ---- globe: a plotting sphere, its graticule every ten degrees
    const globeMat = new THREE.ShaderMaterial({
      uniforms: {
        uSun: { value: SUN }, uFade: { value: 0 },
        uOcean: { value: col("--globe-ocean") }, uDeep: { value: col("--globe-deep") },
        uGrat: { value: col("--globe-grat") }, uLimb: { value: col("--globe-limb") },
      },
      vertexShader: `
        varying vec3 vW; varying vec3 vP; varying vec3 vV;
        void main(){
          vec4 w = modelMatrix * vec4(position, 1.0);
          vW = normalize(mat3(modelMatrix) * normal); vP = position;
          vV = cameraPosition - w.xyz;
          gl_Position = projectionMatrix * viewMatrix * w;
        }`,
      fragmentShader: `
        uniform vec3 uSun, uOcean, uDeep, uGrat, uLimb; uniform float uFade;
        varying vec3 vW; varying vec3 vP; varying vec3 vV;
        void main(){
          vec3 n = normalize(vW);
          float lit = smoothstep(-0.35, 0.75, dot(n, uSun));
          vec3 c = mix(uDeep, uOcean, lit);
          vec3 p = normalize(vP);
          float lat = degrees(asin(clamp(p.y, -1.0, 1.0)));
          float lon = degrees(atan(-p.z, p.x));
          float dl = abs(fract(lat / 10.0 + 0.5) - 0.5) * 10.0;
          float dn = abs(fract(lon / 10.0 + 0.5) - 0.5) * 10.0;
          float wl = max(fwidth(lat), 1e-4), wn = max(fwidth(lon), 1e-4);
          float g = max(1.0 - smoothstep(0.0, 1.3 * wl, dl), 1.0 - smoothstep(0.0, 1.3 * wn, dn));
          c = mix(c, uGrat, g * (0.25 + 0.5 * lit));
          float fr = pow(1.0 - max(dot(n, normalize(vV)), 0.0), 2.6);
          c += uLimb * fr * 0.42;
          gl_FragColor = vec4(c * uFade, 1.0);
        }`,
    });
    globeMat.extensions = { derivatives: true };
    const globe = new THREE.Mesh(new THREE.SphereGeometry(R, tier.seg[0], tier.seg[1]), globeMat);
    scene.add(globe);

    // ---- the limb: a thin shell of scattered light round the planet's edge
    const limbMat = new THREE.ShaderMaterial({
      uniforms: { uLimb: { value: col("--globe-limb") }, uFade: { value: 0 } },
      vertexShader: `
        varying vec3 vN; varying vec3 vV;
        void main(){
          vec4 w = modelMatrix * vec4(position, 1.0);
          vN = normalize(mat3(modelMatrix) * normal); vV = normalize(cameraPosition - w.xyz);
          gl_Position = projectionMatrix * viewMatrix * w;
        }`,
      fragmentShader: `
        uniform vec3 uLimb; uniform float uFade;
        varying vec3 vN; varying vec3 vV;
        void main(){
          // Back faces of a slightly larger shell: brightest where the shell
          // meets the planet's edge, falling away into space.
          float i = pow(clamp(0.66 - dot(vN, vV), 0.0, 1.0), 3.0);
          gl_FragColor = vec4(uLimb * i * 0.95 * uFade, 1.0);
        }`,
      side: THREE.BackSide, blending: THREE.AdditiveBlending, transparent: true, depthWrite: false,
    });
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(R * 1.045, tier.seg[0], tier.seg[1]), limbMat));

    // ---- a few distant stars: a rear layer that never moves on its own
    let stars = null;
    if (tier.stars) {
      let seed = 1601;
      const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
      const pos = new Float32Array(tier.stars * 3);
      for (let i = 0; i < tier.stars; i++) {
        const u = rnd() * 2 - 1, a = rnd() * Math.PI * 2, s = Math.sqrt(1 - u * u);
        pos.set([s * Math.cos(a) * 260, u * 260, s * Math.sin(a) * 260], i * 3);
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      stars = new THREE.Points(g, new THREE.PointsMaterial({
        color: col("--ink-3"), size: 1.3, sizeAttenuation: false, transparent: true, opacity: 0 }));
      scene.add(stars);
    }

    // ---- the cells: one instanced mesh, every height and colour from the archive
    const N = data.cells.length;
    const box = new THREE.BoxGeometry(1, 1, 1);
    box.translate(0, 0.5, 0);
    const aColor = new Float32Array(N * 3), aFlag = new Float32Array(N),
      aRegion = new Float32Array(N), aDelay = new Float32Array(N);
    data.cells.forEach((c, k) => { aRegion[k] = c.ri; });
    box.setAttribute("aColor", new THREE.InstancedBufferAttribute(aColor, 3));
    box.setAttribute("aFlag", new THREE.InstancedBufferAttribute(aFlag, 1));
    box.setAttribute("aRegion", new THREE.InstancedBufferAttribute(aRegion, 1));
    box.setAttribute("aDelay", new THREE.InstancedBufferAttribute(aDelay, 1));
    const cellMat = new THREE.ShaderMaterial({
      uniforms: {
        uT: { value: 0 }, uRise: { value: 0.55 }, uHover: { value: -1 }, uSun: { value: SUN },
        uHatchInk: { value: col("--hatch-ink") }, uHatchBg: { value: col("--hatch-bg") },
        uHot: { value: col("--accent") }, uShow: { value: 0 },
      },
      vertexShader: `
        attribute vec3 aColor; attribute float aFlag; attribute float aRegion; attribute float aDelay;
        uniform float uT, uRise, uHover;
        varying vec3 vColor; varying vec3 vN; varying vec3 vW; varying float vFlag; varying float vHot; varying float vY;
        void main(){
          // The column rises once, as it is plotted; afterwards it stands at
          // the archive's value and lead changes swap it without a tween.
          float r = clamp((uT - aDelay) / uRise, 0.0, 1.0);
          r = 1.0 - pow(1.0 - r, 3.0);
          vec3 p = position;
          p.y *= mix(0.04, 1.0, r);
          vec4 w = modelMatrix * instanceMatrix * vec4(p, 1.0);
          vW = w.xyz; vY = position.y;
          vN = normalize(mat3(modelMatrix) * mat3(instanceMatrix) * normal);
          vColor = aColor; vFlag = aFlag; vHot = 1.0 - step(0.5, abs(aRegion - uHover));
          gl_Position = projectionMatrix * viewMatrix * w;
        }`,
      fragmentShader: `
        uniform vec3 uSun, uHatchInk, uHatchBg, uHot; uniform float uShow;
        varying vec3 vColor; varying vec3 vN; varying vec3 vW; varying float vFlag; varying float vHot; varying float vY;
        void main(){
          vec3 n = normalize(vN);
          vec3 base = vColor;
          if (vFlag > 0.5) {
            float s = step(0.5, fract((vW.x + vW.y - vW.z) * 7.0));
            base = mix(uHatchBg, uHatchInk, s);
          }
          // One low key light: tops face the sky and catch it, sides fall
          // into shade, so a column reads as a column.
          float diff = max(dot(n, uSun), 0.0);
          float top = max(dot(n, normalize(vW)), 0.0);
          vec3 c = base * (0.52 + 0.62 * diff) * mix(0.6, 1.1, top);
          c = mix(c, uHot, vHot * 0.26);
          gl_FragColor = vec4(c * uShow, 1.0);
        }`,
    });
    const cells = new THREE.InstancedMesh(box, cellMat, N);
    cells.frustumCulled = false;
    scene.add(cells);

    // Each cell's frame on the sphere: x east, y up, z south. Sizes are the
    // cell's true extent on the globe, shrunk by GAP so the grid shows.
    const frames = data.cells.map((c) => {
      const up = dir(c.lat, c.lon), e = east(c.lon), s = new THREE.Vector3().crossVectors(e, up);
      return { up, e, s, pos: up.clone().multiplyScalar(R),
        w: data.grid.dlon * D2R * R * Math.cos(c.lat * D2R) * GAP, d: data.grid.dlat * D2R * R * GAP };
    });
    const m4 = new THREE.Matrix4(), scl = new THREE.Matrix4();

    // ---- coast: the outline rings, drawn progressively in the intro
    const coastPos = [];
    data.rings.forEach((ring) => {
      for (let i = 0; i + 1 < ring.length; i++) {
        const a = dir(ring[i][0], ring[i][1]).multiplyScalar(R + H0 + 0.004);
        const b = dir(ring[i + 1][0], ring[i + 1][1]).multiplyScalar(R + H0 + 0.004);
        coastPos.push(a.x, a.y, a.z, b.x, b.y, b.z);
      }
    });
    const coastGeo = new THREE.BufferGeometry();
    coastGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(coastPos), 3));
    const coastMat = new THREE.LineBasicMaterial({ color: col("--ink"), transparent: true, opacity: 0.85 });
    const coast = new THREE.LineSegments(coastGeo, coastMat);
    coastGeo.setDrawRange(0, 0);
    scene.add(coast);

    // ---- subdivision rims: the grid's own region edges at each cell's top
    const inner = data.edges.filter((e) => e.n >= 0);
    const rimGeo = new THREE.BufferGeometry();
    rimGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(inner.length * 6), 3));
    const rimMat = new THREE.LineBasicMaterial({ color: col("--ink"), transparent: true, opacity: 0 });
    const rims = new THREE.LineSegments(rimGeo, rimMat);
    scene.add(rims);

    // ---- the hovered subdivision's outline, drawn over everything
    const hotGeo = new THREE.BufferGeometry();
    hotGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3));
    const hot = new THREE.LineSegments(hotGeo, new THREE.LineBasicMaterial({
      color: col("--focus"), transparent: true, depthTest: false }));
    hot.renderOrder = 10; hot.visible = false;
    scene.add(hot);

    let rds = [], heights = new Float32Array(N), lead = opts.lead;
    const stations = stationCells(data, cube);

    function applyLead(L) {
      lead = L;
      rds = readings(data, cube, L);
      const order = ranked(rds);
      const rankOf = new Map(order.map((x, i) => [x.ri, i]));
      const ramp = [0, 1, 2, 3, 4].map((i) => col(`--c${i}`)), land = col("--land"), none = col("--land-none");
      data.cells.forEach((c, k) => {
        const rd = rds[c.ri], f = frames[k], h = heightOf(rd);
        heights[k] = h;
        m4.makeBasis(f.e, f.up, f.s);
        scl.makeScale(f.w, h, f.d);
        m4.multiply(scl).setPosition(f.pos);
        cells.setMatrixAt(k, m4);
        const cc = rd.state === "scored" ? ramp[bandOf(rd.p)] : rd.state === "refused" ? land : none;
        aColor.set([cc.r, cc.g, cc.b], k * 3);
        aFlag[k] = rd.state === "refused" ? 1 : 0;
        // Plotted in rank order: the highest reading rises first.
        aDelay[k] = rankOf.has(c.ri) ? 0.05 * rankOf.get(c.ri) : 0;
      });
      cells.instanceMatrix.needsUpdate = true;
      ["aColor", "aFlag", "aDelay"].forEach((n) => { box.attributes[n].needsUpdate = true; });
      const rp = rimGeo.attributes.position.array;
      inner.forEach((e, i) => {
        const h = Math.max(heights[e.k], heights[e.n]) + 0.003;
        const a = dir(e.a[0], e.a[1]).multiplyScalar(R + h), b = dir(e.b[0], e.b[1]).multiplyScalar(R + h);
        rp.set([a.x, a.y, a.z, b.x, b.y, b.z], i * 6);
      });
      rimGeo.attributes.position.needsUpdate = true;
      rimGeo.computeBoundingSphere();
      if (hoverRi >= 0) outlineRegion(hoverRi);
      invalidate();
    }

    let hoverRi = -1;
    function outlineRegion(ri) {
      const segs = data.edges.filter((e) => data.cells[e.k].ri === ri || (e.n >= 0 && data.cells[e.n].ri === ri));
      const arr = new Float32Array(segs.length * 6);
      segs.forEach((e, i) => {
        const h = Math.max(heights[e.k], e.n >= 0 ? heights[e.n] : 0) + 0.006;
        const a = dir(e.a[0], e.a[1]).multiplyScalar(R + h), b = dir(e.b[0], e.b[1]).multiplyScalar(R + h);
        arr.set([a.x, a.y, a.z, b.x, b.y, b.z], i * 6);
      });
      hotGeo.setAttribute("position", new THREE.BufferAttribute(arr, 3));
      hotGeo.computeBoundingSphere();
      hot.visible = true;
    }
    function setHover(ri) {
      if (ri === hoverRi) return;
      hoverRi = ri;
      cellMat.uniforms.uHover.value = ri;
      if (ri < 0) hot.visible = false; else outlineRegion(ri);
      invalidate();
    }

    // ---- camera: an orbit about India's middle, from the south
    // look: 0 frames the whole planet, 1 frames India; the intro carries it
    // from one to the other.
    const view = { el: 0.84, az: -0.1, dist: 10.6, look: 1 };
    const intro = { el: 0.5, az: 0.7, dist: 62, look: 0 };
    const cam = { ...view };
    let offsetFrac = 0;
    function placeCamera() {
      const p = T.clone().multiplyScalar(cam.look)
        .addScaledVector(NORTH, -Math.cos(cam.el) * Math.cos(cam.az) * cam.dist)
        .addScaledVector(EAST, Math.cos(cam.el) * Math.sin(cam.az) * cam.dist)
        .addScaledVector(UP, Math.sin(cam.el) * cam.dist);
      camera.position.copy(p);
      camera.up.copy(NORTH.clone().multiplyScalar(Math.sin(cam.el)).addScaledVector(UP, Math.cos(cam.el)).normalize());
      camera.lookAt(T.clone().multiplyScalar(cam.look));
    }

    let W = 1, Hh = 1;
    function resize() {
      W = host.clientWidth || 1; Hh = host.clientHeight || 1;
      renderer.setSize(W, Hh, false);
      camera.aspect = W / Hh;
      // On a wide screen the text owns the left; India is framed right of centre.
      // On a phone the scene sits under the text, so India is lifted toward
      // the top of its band rather than left floating over empty void.
      offsetFrac = W >= 960 ? 0.23 : 0;
      if (offsetFrac) camera.setViewOffset(W, Hh, -W * offsetFrac, 0, W, Hh);
      else camera.setViewOffset(W, Hh, 0, Hh * 0.13, W, Hh);
      camera.updateProjectionMatrix();
      invalidate();
    }

    // ---- the intro: named states, each a stop, cancellable at any point
    const PH = { orbit: [0, 2.4], coast: [1.0, 2.5], rims: [2.3, 2.9], plot: [2.6, 4.4] };
    const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
    const span = (t, [a, b]) => Math.min(1, Math.max(0, (t - a) / (b - a)));
    let t0 = performance.now(), introDone = false;
    function stepIntro(now) {
      const t = (now - t0) / 1000;
      const f = ease(span(t, PH.orbit));
      cam.el = intro.el + (view.el - intro.el) * f;
      cam.az = intro.az + (view.az - intro.az) * f;
      cam.dist = intro.dist + (view.dist - intro.dist) * f;
      cam.look = intro.look + (view.look - intro.look) * f;
      const fade = span(t, [0, 0.9]);
      globeMat.uniforms.uFade.value = fade; limbMat.uniforms.uFade.value = fade;
      if (stars) stars.material.opacity = 0.55 * fade;
      coastGeo.setDrawRange(0, Math.floor(coastPos.length / 3 * span(t, PH.coast) / 2) * 2);
      rimMat.opacity = 0.3 * span(t, PH.rims);
      cellMat.uniforms.uShow.value = span(t, [PH.coast[0] + 0.6, PH.rims[0]]);
      cells.visible = cellMat.uniforms.uShow.value > 0;
      cellMat.uniforms.uT.value = Math.max(0, t - PH.plot[0]);
      opts.onPhase && opts.onPhase(t >= PH.plot[1] ? "plotted" : t >= PH.rims[0] ? "boundaries"
        : t >= PH.coast[0] ? "outline" : "orbit");
      if (t >= PH.plot[1] + 0.2) finishIntro();
    }
    function finishIntro() {
      if (introDone) return;
      introDone = true;
      Object.assign(cam, view);
      globeMat.uniforms.uFade.value = 1; limbMat.uniforms.uFade.value = 1;
      if (stars) stars.material.opacity = 0.55;
      coastGeo.setDrawRange(0, coastPos.length / 3);
      rimMat.opacity = 0.3;
      cellMat.uniforms.uShow.value = 1;
      cells.visible = true;
      cellMat.uniforms.uT.value = 1e3;
      opts.onPhase && opts.onPhase("plotted");
      opts.onReady && opts.onReady();
      invalidate();
    }

    // ---- drag to turn, with inertia; wheel and vertical swipes stay the page's
    let vel = { az: 0, el: 0 }, drag = null;
    canvas.addEventListener("pointerdown", (e) => {
      if (!introDone) finishIntro();
      if (e.pointerType === "touch") return;      // touch scrolls the page; the rail and plots work by tap
      drag = { x: e.clientX, y: e.clientY, moved: 0 };
      try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
    });
    canvas.addEventListener("pointermove", (e) => {
      if (drag) {
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        drag.x = e.clientX; drag.y = e.clientY; drag.moved += Math.abs(dx) + Math.abs(dy);
        vel.az = -dx * 0.0042; vel.el = dy * 0.0032;
        nudge(vel.az, vel.el);
        return;
      }
      pointerAt = e; invalidate();
    });
    const endDrag = (e) => {
      if (!drag) return;
      const click = drag.moved < 5;
      drag = null;
      try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
      if (click) { pick(e, true); vel = { az: 0, el: 0 }; } else if (!opts.reduced) coast_();
    };
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);
    canvas.addEventListener("pointerleave", () => { if (!drag) { pointerAt = null; setHover(-1); opts.onHover && opts.onHover(null); } });
    canvas.addEventListener("click", (e) => { if (e.pointerType === "touch" || e.detail === 0) pick(e, true); });
    function nudge(daz, del) {
      cam.az = Math.max(-0.75, Math.min(0.55, cam.az + daz));
      cam.el = Math.max(0.36, Math.min(1.15, cam.el + del));
      invalidate();
    }
    let coasting = false;
    function coast_() { coasting = true; invalidate(); }

    // ---- picking: one raycast per frame against the cells actually drawn
    const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
    let pointerAt = null;
    function cellUnder(e) {
      const r = canvas.getBoundingClientRect();
      ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      ray.setFromCamera(ndc, camera);
      const hit = ray.intersectObject(cells, false)[0];
      return hit ? hit.instanceId : -1;
    }
    function pick(e, open) {
      if (!introDone) return;
      const k = cellUnder(e);
      if (k < 0) return;
      const rd = rds[data.cells[k].ri];
      if (open && rd.state !== "none" && opts.onPick) opts.onPick(rd);
    }
    function hoverFrame() {
      if (!pointerAt || !introDone) return;
      const e = pointerAt; pointerAt = null;
      const k = cellUnder(e);
      const ri = k >= 0 ? data.cells[k].ri : -1;
      setHover(ri);
      canvas.style.cursor = ri >= 0 && rds[ri].state !== "none" ? "pointer" : drag ? "grabbing" : "grab";
      opts.onHover && opts.onHover(ri >= 0 ? rds[ri] : null, e.clientX, e.clientY);
    }

    // ---- station plots: HTML over the canvas, placed from the projected tops
    const plotLayer = opts.plots, lineLayer = opts.leaders;
    // The land's extent on screen, from the coast, so plots can avoid it.
    const coastProbe = [];
    data.rings.forEach((ring) => ring.forEach((p, i) => { if (i % 3 === 0) coastProbe.push(dir(p[0], p[1]).multiplyScalar(R)); }));
    function landBox() {
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      const v = new THREE.Vector3();
      coastProbe.forEach((p) => {
        v.copy(p).project(camera);
        const x = (v.x * 0.5 + 0.5) * W, y = (-v.y * 0.5 + 0.5) * Hh;
        x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
      });
      return isFinite(x0) ? { x0, y0, x1, y1 } : null;
    }
    function plotStations() {
      if (!plotLayer) return;
      const show = introDone ? ranked(rds).slice(0, 8) : [];
      const items = show.map(({ ri, r }) => {
        const st = stations.get(ri);
        if (!st) return null;
        const f = frames[st.k];
        const v = f.pos.clone().addScaledVector(f.up, heights[st.k]).project(camera);
        if (v.z > 1) return null;
        return { ri, r, x: (v.x * 0.5 + 0.5) * W, y: (-v.y * 0.5 + 0.5) * Hh };
      }).filter(Boolean);
      const boxW = W < 640 ? 168 : 190, boxH = 76;
      const laid = placeInVoid(items, landBox(), W, Hh, boxW, boxH, opts.top || 72, opts.minX ? opts.minX() : 8,
        W < 640 ? 1 : (opts.nStations || 3));
      plotLayer.innerHTML = laid.map((it) => it.box
        ? `<div class="stn-wrap" style="left:${it.box.x.toFixed(1)}px;top:${it.box.y.toFixed(1)}px;width:${boxW}px">${stationHTML(it.r, opts.href(it.r))}</div>`
        : "").join("");
      if (lineLayer) {
        lineLayer.setAttribute("viewBox", `0 0 ${W} ${Hh}`);
        lineLayer.innerHTML = laid.map((it) => leaderSVG(it, boxW, boxH)).join("");
      }
    }

    // ---- frames on demand
    let needs = true, raf = 0, visible = true, alive = true, lastKey = "";
    function invalidate() { needs = true; if (!raf && alive) raf = requestAnimationFrame(frame); }
    function frame(now) {
      raf = 0;
      if (!alive) return;
      if (!introDone) { stepIntro(now); needs = true; }
      if (coasting && !drag) {
        vel.az *= 0.9; vel.el *= 0.9;
        nudge(vel.az, vel.el);
        if (Math.abs(vel.az) + Math.abs(vel.el) < 1e-4) coasting = false;
      }
      hoverFrame();
      if (needs && visible && !document.hidden) {
        needs = false;
        placeCamera();
        renderer.render(scene, camera);
        const key = `${cam.el.toFixed(4)}|${cam.az.toFixed(4)}|${cam.dist.toFixed(3)}|${cam.look.toFixed(3)}|${W}|${Hh}|${lead}|${introDone}`;
        if (key !== lastKey) { lastKey = key; plotStations(); }
      }
      if (!introDone || coasting || drag) raf = requestAnimationFrame(frame);
    }

    // Nothing renders while the hero is off screen or the tab is hidden.
    const io = "IntersectionObserver" in window ? new IntersectionObserver((es) => {
      visible = es.some((e) => e.isIntersecting);
      if (visible) invalidate();
    }) : null;
    if (io) io.observe(host);
    const onVis = () => { if (!document.hidden) invalidate(); };
    document.addEventListener("visibilitychange", onVis);
    const ro = "ResizeObserver" in window ? new ResizeObserver(resize) : null;
    if (ro) ro.observe(host); else window.addEventListener("resize", resize);

    function dispose() {
      alive = false;
      if (raf) cancelAnimationFrame(raf);
      if (io) io.disconnect();
      if (ro) ro.disconnect();
      document.removeEventListener("visibilitychange", onVis);
      scene.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) o.material.dispose();
      });
      renderer.dispose();
      canvas.remove();
    }
    addEventListener("pagehide", dispose, { once: true });

    applyLead(lead);
    resize();
    if (opts.still) finishIntro(); else { t0 = performance.now(); invalidate(); }
    return {
      kind: "webgl",
      setLead: (L) => { applyLead(L); },
      skip: finishIntro,
      get reading() { return rds; },
      info: () => ({ calls: renderer.info.render.calls, triangles: renderer.info.render.triangles,
        geometries: renderer.info.memory.geometries, textures: renderer.info.memory.textures, cells: N }),
      dispose,
    };
  }

  // ===================================================== flat fallback
  // The same cells on a 2-D canvas, equirectangular at India's middle
  // latitude. Used when WebGL is missing, when three.js cannot load, or when
  // the quality tier is 2d. Same readings, hover, links and lead rail.
  function mountFlat(host, data, cube, opts) {
    const canvas = document.createElement("canvas");
    canvas.className = "orbit-canvas flat";
    host.prepend(canvas);
    const g = canvas.getContext("2d");
    const grid = data.grid, kx = Math.cos(22 * D2R);
    let W = 1, Hh = 1, s = 1, ox = 0, oy = 0, rds = [], lead = opts.lead, hoverRi = -1;
    const X = (lon) => ox + (lon - grid.lon0) * kx * s, Y = (lat) => oy - (lat - grid.lat0) * s;
    let hatch = null;
    function pattern() {
      const c = document.createElement("canvas"); c.width = c.height = 8;
      const h = c.getContext("2d");
      h.fillStyle = css("--hatch-bg"); h.fillRect(0, 0, 8, 8);
      h.strokeStyle = css("--hatch-ink"); h.lineWidth = 2;
      h.beginPath(); h.moveTo(-2, 10); h.lineTo(10, -2); h.stroke();
      return g.createPattern(c, "repeat");
    }
    function size() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = host.clientWidth || 1; Hh = host.clientHeight || 1;
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(Hh * dpr);
      canvas.style.width = W + "px"; canvas.style.height = Hh + "px";
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      // On a wide screen the text owns the left 57%, as it does over the globe.
      const left = W >= 960 ? W * 0.57 : 16, avail = W - left - 24;
      const spanLon = grid.nx * grid.dlon * kx, spanLat = grid.ny * grid.dlat;
      s = Math.min(avail / spanLon, (Hh - 120) / spanLat);
      ox = left + (avail - spanLon * s) / 2; oy = 40 + spanLat * s;
      draw();
    }
    function draw() {
      g.fillStyle = css("--ground"); g.fillRect(0, 0, W, Hh);
      g.strokeStyle = css("--rule"); g.lineWidth = 1;
      for (let lon = 70; lon <= 95; lon += 5) { g.beginPath(); g.moveTo(X(lon), Y(grid.lat0)); g.lineTo(X(lon), Y(grid.lat0 + grid.ny * grid.dlat)); g.stroke(); }
      for (let lat = 10; lat <= 35; lat += 5) { g.beginPath(); g.moveTo(X(grid.lon0), Y(lat)); g.lineTo(X(grid.lon0 + grid.nx * grid.dlon), Y(lat)); g.stroke(); }
      hatch = hatch || pattern();
      const ramp = [0, 1, 2, 3, 4].map((i) => css(`--c${i}`)), cw = grid.dlon * kx * s * GAP, ch = grid.dlat * s * GAP;
      data.cells.forEach((c) => {
        const rd = rds[c.ri];
        g.fillStyle = rd.state === "scored" ? ramp[bandOf(rd.p)] : rd.state === "refused" ? hatch : css("--land-none");
        g.fillRect(X(c.lon) - cw / 2, Y(c.lat) - ch / 2, cw, ch);
        if (c.ri === hoverRi) { g.fillStyle = "rgba(195,208,245,.28)"; g.fillRect(X(c.lon) - cw / 2, Y(c.lat) - ch / 2, cw, ch); }
      });
      g.strokeStyle = css("--ink"); g.lineWidth = 1.1;
      data.rings.forEach((ring) => {
        g.beginPath();
        ring.forEach(([lat, lon], i) => (i ? g.lineTo(X(lon), Y(lat)) : g.moveTo(X(lon), Y(lat))));
        g.stroke();
      });
      plot();
    }
    const stations = stationCells(data, cube);
    function plot() {
      if (!opts.plots) return;
      const items = ranked(rds).slice(0, 8).map(({ ri, r }) => {
        const st = stations.get(ri); if (!st) return null;
        const c = data.cells[st.k];
        return { ri, r, x: X(c.lon), y: Y(c.lat) };
      }).filter(Boolean);
      const land = { x0: X(grid.lon0), x1: X(grid.lon0 + grid.nx * grid.dlon), y0: Y(grid.lat0 + grid.ny * grid.dlat), y1: Y(grid.lat0) };
      const boxW = W < 640 ? 168 : 190, boxH = 76, laid = placeInVoid(items, land, W, Hh, boxW, boxH, opts.top || 72,
        opts.minX ? opts.minX() : 8, W < 640 ? 1 : (opts.nStations || 3));
      opts.plots.innerHTML = laid.map((it) => it.box
        ? `<div class="stn-wrap" style="left:${it.box.x.toFixed(1)}px;top:${it.box.y.toFixed(1)}px;width:${boxW}px">${stationHTML(it.r, opts.href(it.r))}</div>` : "").join("");
      if (opts.leaders) {
        opts.leaders.setAttribute("viewBox", `0 0 ${W} ${Hh}`);
        opts.leaders.innerHTML = laid.map((it) => leaderSVG(it, boxW, boxH)).join("");
      }
    }
    function cellAtPoint(e) {
      const r = canvas.getBoundingClientRect();
      const lon = grid.lon0 + (e.clientX - r.left - ox) / (kx * s), lat = grid.lat0 + (oy - (e.clientY - r.top)) / s;
      const ix = Math.round((lon - grid.lon0) / grid.dlon), iy = Math.round((lat - grid.lat0) / grid.dlat);
      if (ix < 0 || iy < 0 || ix >= grid.nx || iy >= grid.ny) return -1;
      return data.at[iy * grid.nx + ix];
    }
    canvas.addEventListener("mousemove", (e) => {
      const k = cellAtPoint(e), ri = k >= 0 ? data.cells[k].ri : -1;
      if (ri !== hoverRi) { hoverRi = ri; draw(); }
      canvas.style.cursor = ri >= 0 && rds[ri].state !== "none" ? "pointer" : "";
      opts.onHover && opts.onHover(ri >= 0 ? rds[ri] : null, e.clientX, e.clientY);
    });
    canvas.addEventListener("mouseleave", () => { hoverRi = -1; draw(); opts.onHover && opts.onHover(null); });
    canvas.addEventListener("click", (e) => {
      const k = cellAtPoint(e); if (k < 0) return;
      const rd = rds[data.cells[k].ri];
      if (rd.state !== "none" && opts.onPick) opts.onPick(rd);
    });
    const ro = "ResizeObserver" in window ? new ResizeObserver(size) : null;
    if (ro) ro.observe(host); else window.addEventListener("resize", size);
    function setLead(L) { lead = L; rds = readings(data, cube, L); draw(); }
    setLead(lead);
    size();
    opts.onPhase && opts.onPhase("plotted");
    opts.onReady && opts.onReady();
    return { kind: "flat", setLead, skip() {}, get reading() { return rds; },
      info: () => ({ cells: data.cells.length }), dispose() { if (ro) ro.disconnect(); canvas.remove(); } };
  }

  // ------------------------------------------------------------- loader
  // three.js is fetched only when it will be used, after the page is up, so
  // the headline never waits on 600 KB of renderer.
  function loadThree() {
    if (window.THREE) return Promise.resolve(window.THREE);
    return new Promise((ok, fail) => {
      const s = document.createElement("script");
      s.src = "/vendor/three.min.js"; s.async = true;
      s.onload = () => (window.THREE ? ok(window.THREE) : fail(new Error("three.js did not define THREE")));
      s.onerror = () => fail(new Error("three.js failed to load"));
      document.head.appendChild(s);
    });
  }

  async function mount(host, grid, cube, opts) {
    const data = prepare(grid);
    const tier = qualityTier(webglAvailable());
    host.dataset.quality = tier;
    if (tier !== "2d") {
      try {
        await loadThree();
        const gl = mountGL(host, data, cube, { ...opts, tier });
        if (gl) return gl;
      } catch (e) { /* the flat chart below is the same data */ }
      host.dataset.quality = "2d";
    }
    return mountFlat(host, data, cube, opts);
  }

  return { mount, readings, ranked, bandOf, prepare };
})();
