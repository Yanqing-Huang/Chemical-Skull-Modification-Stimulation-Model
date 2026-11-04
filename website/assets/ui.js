/* UI: parameter sliders, WebGL 3D layered-slab view, ECharts figures, stats. */
(function () {
  'use strict';

  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();

  var DEF = {
    freq: 20, reagent: 'edta', cMain: 0.1, tSoak: 30,
    hOuter: 0.43, hDiploe: 0.20, hInner: 0.17,
    epsOuter: 0.08, epsDiploe: 0.30, epsInner: 0.10,
    tauOuter: 4, tauDiploe: 2, tauInner: 4,
    qOuter: 11000, qDiploe: 6000, qInner: 10000,
    kAcid: 1e-7, kEdta: 5e-6,
    dAcid: 1e-9, dEdta: 6e-10, dUrea: 1.4e-9,
    kf: 1e-5, alpha: 3.2e-5,
    sigmaFluid: 1.5, sigmaMatrix: 1e-4, archieM: 1.5
  };
  var params = Object.assign({}, DEF);
  var sim = null;
  var colorMode = 'calcium';

  /* ---------- slider wiring ---------- */
  var SLIDERS = [
    ['freq', 'v-freq', function (v) { return v.toFixed(0) + ' Hz'; }, function (v) { return v; }],
    ['cMain', 'v-cmain', function (v) { return v.toFixed(2) + ' mol/L'; }, function (v) { return v; }],
    ['tSoak', 'v-tsoak', function (v) { return v.toFixed(0) + ' min'; }, function (v) { return v; }],
    ['hOuter', 'v-ho', function (v) { return v.toFixed(2) + ' mm'; }, function (v) { return v; }],
    ['hDiploe', 'v-hd', function (v) { return v.toFixed(2) + ' mm'; }, function (v) { return v; }],
    ['hInner', 'v-hi', function (v) { return v.toFixed(2) + ' mm'; }, function (v) { return v; }],
    ['epsOuter', 'v-eo', function (v) { return v.toFixed(2); }, function (v) { return v; }],
    ['epsDiploe', 'v-ed', function (v) { return v.toFixed(2); }, function (v) { return v; }],
    ['epsInner', 'v-ei', function (v) { return v.toFixed(2); }, function (v) { return v; }],
    ['tauOuter', 'v-to', function (v) { return v.toFixed(1); }, function (v) { return v; }],
    ['tauDiploe', 'v-td', function (v) { return v.toFixed(1); }, function (v) { return v; }],
    ['tauInner', 'v-ti', function (v) { return v.toFixed(1); }, function (v) { return v; }],
    ['kEdta', 'v-kedta', function (v) { return v.toExponential(1) + ' m³/(mol·s)'; }, function (v) { return Math.pow(10, v); }],
    ['kAcid', 'v-kacid', function (v) { return v.toExponential(1) + ' m³/(mol·s)'; }, function (v) { return Math.pow(10, v); }],
    ['sigmaFluid', 'v-sf', function (v) { return v.toFixed(2) + ' S/m'; }, function (v) { return v; }],
    ['archieM', 'v-am', function (v) { return v.toFixed(2); }, function (v) { return v; }]
  ];
  SLIDERS.forEach(function (cfg) {
    var el = document.getElementById('in-' + cfg[0]);
    var out = document.getElementById(cfg[1]);
    function sync() {
      var raw = parseFloat(el.value);
      params[cfg[0]] = cfg[3](raw);
      out.textContent = cfg[2](params[cfg[0]]);
    }
    el.addEventListener('input', sync);
    sync();
  });
  document.getElementById('in-reagent').addEventListener('change', function () {
    params.reagent = this.value;
    var hint = { edta: 'EDTA：螯合脱钙（主试剂）', acetic: '乙酸：酸溶解脱钙', urea: '尿素：溶胀，不耗钙', seq: '50% 乙酸预处理 5 min → 当前浓度 EDTA' };
    document.getElementById('reagent-hint').textContent = hint[this.value];
  });

  /* ---------- WebGL 3D layered slab ---------- */
  var canvas = document.getElementById('gl');
  var gl = canvas.getContext('webgl');
  var rotY = -0.6, rotX = 0.45, zoom = 1.0, dragging = false, px = 0, py = 0;
  canvas.addEventListener('mousedown', function (e) { dragging = true; px = e.clientX; py = e.clientY; });
  window.addEventListener('mouseup', function () { dragging = false; });
  window.addEventListener('mousemove', function (e) {
    if (!dragging) return;
    rotY += (e.clientX - px) * 0.008; rotX += (e.clientY - py) * 0.008;
    rotX = Math.max(-1.4, Math.min(1.4, rotX));
    px = e.clientX; py = e.clientY; draw();
  });
  canvas.addEventListener('wheel', function (e) {
    e.preventDefault(); zoom = Math.max(0.5, Math.min(2.5, zoom * (e.deltaY > 0 ? 0.92 : 1.08))); draw();
  }, { passive: false });

  var prog = null;
  function shader(type, src) {
    var s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s;
  }
  if (gl) {
    var vs = shader(gl.VERTEX_SHADER,
      'attribute vec3 aP; attribute vec3 aC; uniform mat4 uM; varying vec3 vC;' +
      'void main(){ vC=aC; gl_Position=uM*vec4(aP,1.0); }');
    var fs = shader(gl.FRAGMENT_SHADER,
      'precision mediump float; varying vec3 vC; void main(){ gl_FragColor=vec4(vC,1.0); }');
    prog = gl.createProgram();
    gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
    gl.enable(gl.DEPTH_TEST);
  }

  function matMul(a, b) {
    var r = new Array(16).fill(0);
    for (var i = 0; i < 4; i++) for (var j = 0; j < 4; j++)
      for (var k = 0; k < 4; k++) r[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
    return r;
  }
  function rotMatY(a) { return [Math.cos(a), 0, -Math.sin(a), 0, 0, 1, 0, 0, Math.sin(a), 0, Math.cos(a), 0, 0, 0, 0, 1]; }
  function rotMatX(a) { return [1, 0, 0, 0, 0, Math.cos(a), Math.sin(a), 0, 0, -Math.sin(a), Math.cos(a), 0, 0, 0, 0, 1]; }
  function persp(fov, asp, n, f) {
    var t = 1 / Math.tan(fov / 2);
    return [t / asp, 0, 0, 0, 0, t, 0, 0, 0, 0, (f + n) / (n - f), -1, 0, 0, 2 * f * n / (n - f), 0];
  }
  function translate(x, y, z) { return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]; }

  function fieldColors() {
    /* per-cell color by selected mode, mapped blue->orange */
    var g = sim.grid, q = sim.qFinal;
    var vals = [];
    if (colorMode === 'calcium') {
      for (var i = 0; i < g.n; i++) vals.push(q[i] / g.q0[i]);
    } else if (colorMode === 'porosity') {
      var ep = window.porosityOf(g, params, q);
      for (i = 0; i < g.n; i++) vals.push(ep[i] / 0.6);
    } else {
      var cSnap = sim.final.snaps[sim.final.snaps.length - 1].c;
      var Cb = params.cMain * 1000; /* solver uses mol/m³ */
      for (i = 0; i < g.n; i++) vals.push(Math.min(1, cSnap[i] / Cb));
    }
    return vals.map(function (v) {
      v = Math.max(0, Math.min(1, v));
      return [0.25 + 0.75 * v, 0.45 + 0.25 * v, 0.85 - 0.6 * v];
    });
  }

  function draw() {
    if (!gl || !sim) return;
    var w = canvas.width = canvas.clientWidth * (window.devicePixelRatio || 1);
    var h = canvas.height = canvas.clientHeight * (window.devicePixelRatio || 1);
    gl.viewport(0, 0, w, h);
    gl.clearColor(1, 1, 1, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    var g = sim.grid;
    var cols = fieldColors();
    var total = g.H[0] + g.H[1] + g.H[2];
    var verts = [], cdata = [];
    var LX = 6, LY = 3; /* lateral extent (exaggerated vs thickness for visibility) */
    var zOff = -total / 2;
    var layerStart = [0, g.H[0], g.H[0] + g.H[1]];
    for (var L = 0; L < 3; L++) {
      /* mean color of layer */
      var r = 0, gg = 0, b = 0, cnt = 0;
      for (var i = 0; i < g.n; i++) if (g.li[i] === L) { r += cols[i][0]; gg += cols[i][1]; b += cols[i][2]; cnt++; }
      r /= cnt; gg /= cnt; b /= cnt;
      var z0 = zOff + layerStart[L], z1 = z0 + g.H[L];
      var corners = [
        [-LX, -LY, z0], [LX, -LY, z0], [LX, LY, z0], [-LX, LY, z0],
        [-LX, -LY, z1], [LX, -LY, z1], [LX, LY, z1], [-LX, LY, z1]
      ];
      var quads = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]];
      /* gradient across thickness within layer: sample cells */
      quads.forEach(function (qd) {
        var idx = [qd[0], qd[1], qd[2], qd[0], qd[2], qd[3]];
        idx.forEach(function (vi) {
          var c = corners[vi];
          /* color interpolation by z position within layer */
          var frac = (c[2] - z0) / (z1 - z0);
          var shade = 0.85 + 0.15 * frac;
          verts.push(c[0], c[1], c[2]);
          cdata.push(r * shade, gg * shade, b * shade);
        });
      });
    }
    var vb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vb);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.STATIC_DRAW);
    var cb = gl.createBuffer();
    var aP = gl.getAttribLocation(prog, 'aP');
    var aC = gl.getAttribLocation(prog, 'aC');
    gl.useProgram(prog);
    gl.enableVertexAttribArray(aP);
    gl.vertexAttribPointer(aP, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, cb);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(cdata), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(aC);
    gl.vertexAttribPointer(aC, 3, gl.FLOAT, false, 0, 0);
    var M = matMul(persp(0.7, w / h, 0.1, 100), matMul(translate(0, 0, -12 / zoom), matMul(rotMatX(rotX), rotMatY(rotY))));
    gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uM'), false, new Float32Array(M));
    gl.drawArrays(gl.TRIANGLES, 0, verts.length / 3);
    /* thickness labels via 2D overlay are omitted; legend shown in HTML */
  }

  /* ---------- charts ---------- */
  var chProfile = echarts.init(document.getElementById('chart-profile'), null, { renderer: 'svg' });
  var chTime = echarts.init(document.getElementById('chart-time'), null, { renderer: 'svg' });
  var chFreq = echarts.init(document.getElementById('chart-freq'), null, { renderer: 'svg' });
  [chProfile, chTime, chFreq].forEach(function (ch) {
    window.addEventListener('resize', function () { ch.resize(); });
  });

  function updateCharts() {
    var g = sim.grid, q = sim.qFinal;
    var xs = g.x.map(function (v) { return +v.toFixed(3); });
    chProfile.setOption({
      animation: false,
      tooltip: { trigger: 'axis', appendToBody: true },
      legend: { textStyle: { color: muted }, top: 0 },
      grid: { left: 56, right: 56, top: 34, bottom: 34 },
      xAxis: { type: 'category', data: xs, name: '厚度位置 x (mm)', nameLocation: 'middle', nameGap: 26, axisLabel: { color: muted }, axisLine: { lineStyle: { color: rule } } },
      yAxis: [
        { type: 'value', name: 'q/q₀', max: 1.05, axisLabel: { color: muted }, splitLine: { lineStyle: { color: rule } } },
        { type: 'value', name: '孔隙率 ε', axisLabel: { color: muted }, splitLine: { show: false } }
      ],
      series: [
        { name: '固相钙剩余', type: 'line', data: q.map(function (v, i) { return +(v / g.q0[i]).toFixed(4); }), lineStyle: { color: accent }, itemStyle: { color: accent }, symbol: 'none' },
        { name: '孔隙率', type: 'line', yAxisIndex: 1, data: window.porosityOf(g, params, q).map(function (v) { return +v.toFixed(4); }), lineStyle: { color: accent2, type: 'dashed' }, itemStyle: { color: accent2 }, symbol: 'none' }
      ]
    });

    var snaps = sim.final.snaps;
    var caT = snaps.map(function (sn) {
      var tot = 0, nat = 0;
      for (var i = 0; i < g.n; i++) { tot += sn.q[i] * g.dx[i]; nat += g.q0[i] * g.dx[i]; }
      return [+(sn.t / 60).toFixed(2), +(100 * tot / nat).toFixed(3)];
    });
    chTime.setOption({
      animation: false,
      tooltip: { trigger: 'axis', appendToBody: true },
      grid: { left: 56, right: 20, top: 34, bottom: 34 },
      xAxis: { type: 'value', name: '浸泡时间 (min)', nameLocation: 'middle', nameGap: 26, axisLabel: { color: muted }, axisLine: { lineStyle: { color: rule } } },
      yAxis: { type: 'value', name: '全颅骨钙剩余 (%)', axisLabel: { color: muted }, splitLine: { lineStyle: { color: rule } }, min: function (v) { return Math.floor(v.min - 1); }, max: 100 },
      series: [{ name: '钙剩余', type: 'line', data: caT, lineStyle: { color: accent }, itemStyle: { color: accent }, symbol: 'none', areaStyle: { color: accent, opacity: 0.08 } }]
    });

    var freqs = [];
    for (var f = 1; f <= 200; f += 4) freqs.push(f);
    var Rf = window.resistanceVsFreq(sim.R1, freqs);
    chFreq.setOption({
      animation: false,
      tooltip: { trigger: 'axis', appendToBody: true, formatter: function (ps) { var p2 = ps[0]; return p2.value[0] + ' Hz<br/>阻抗: ' + p2.value[1].toFixed(1) + ' Ω·mm'; } },
      grid: { left: 64, right: 20, top: 34, bottom: 34 },
      xAxis: { type: 'value', name: '频率 (Hz)', nameLocation: 'middle', nameGap: 26, max: 200, axisLabel: { color: muted }, axisLine: { lineStyle: { color: rule } } },
      yAxis: { type: 'value', name: '阻抗 (Ω·mm, 任意单位)', axisLabel: { color: muted }, splitLine: { lineStyle: { color: rule } } },
      series: [{ name: '阻抗-频率', type: 'line', data: freqs.map(function (f, i) { return [f, +Rf[i].toFixed(1)]; }), lineStyle: { color: accent2 }, itemStyle: { color: accent2 }, symbol: 'none',
        markLine: { silent: true, symbol: 'none', lineStyle: { color: muted, type: 'dashed' }, label: { color: muted, formatter: '当前频率' }, data: [{ xAxis: params.freq }] } }
      ]
    });
  }

  /* ---------- stats & diagnostics ---------- */
  function fmt(v, d) { return Number(v).toFixed(d); }
  function updateStats() {
    document.getElementById('st-ca').textContent = fmt(100 * sim.caWhole, 2) + '%';
    document.getElementById('st-ca-o').textContent = fmt(100 * sim.caOuter, 2) + '%';
    document.getElementById('st-r').textContent = fmt(sim.R1 / sim.R0, 3);
    document.getElementById('st-phi').textContent = fmt(sim.diag[0].phi, 2);
    var names = ['外板', '板障', '内板'];
    var html = '';
    sim.diag.forEach(function (d, i) {
      html += '<tr><td>' + names[i] + '</td><td>' + fmt(d.phi, 2) + '</td><td>' + fmt(d.eta, 3) + '</td><td>' + fmt(d.bi, 3) + '</td></tr>';
    });
    document.getElementById('diag-body').innerHTML = html;
  }

  /* ---------- run ---------- */
  function run() {
    sim = window.runParamSim(params);
    draw(); updateCharts(); updateStats();
  }
  document.getElementById('run-btn').addEventListener('click', run);
  document.getElementById('reset-btn').addEventListener('click', function () {
    params = Object.assign({}, DEF);
    SLIDERS.forEach(function (cfg) {
      var el = document.getElementById('in-' + cfg[0]);
      el.value = el.getAttribute('data-def');
      var raw = parseFloat(el.value);
      params[cfg[0]] = cfg[3](raw);
      document.getElementById(cfg[1]).textContent = cfg[2](params[cfg[0]]);
    });
    document.getElementById('in-reagent').value = DEF.reagent;
    document.getElementById('in-reagent').dispatchEvent(new Event('change'));
    run();
  });
  document.querySelectorAll('.color-btn').forEach(function (b) {
    b.addEventListener('click', function () {
      document.querySelectorAll('.color-btn').forEach(function (x) { x.classList.remove('active'); });
      b.classList.add('active');
      colorMode = b.getAttribute('data-mode');
      draw();
    });
  });

  run();
})();
