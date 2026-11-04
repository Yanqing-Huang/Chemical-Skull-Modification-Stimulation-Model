/* Numerical engine: 1D finite-volume diffusion-reaction across three skull layers.
   Solves eps*dc/dt = d(De dc/dx)/dx - nu*kr*q*c with Robin outer BC, zero-flux inner BC.
   Matches the Python reference solver in the repository root. */
(function () {
  'use strict';

  function thomas(lo, dg, up, b) {
    var n = dg.length, i;
    var cp = new Array(n), dp = new Array(n);
    cp[0] = up[0] / dg[0]; dp[0] = b[0] / dg[0];
    for (i = 1; i < n; i++) {
      var m = dg[i] - lo[i] * cp[i - 1];
      cp[i] = up[i] / m;
      dp[i] = (b[i] - lo[i] * dp[i - 1]) / m;
    }
    var x = new Array(n);
    x[n - 1] = dp[n - 1];
    for (i = n - 2; i >= 0; i--) x[i] = dp[i] - cp[i] * x[i + 1];
    return x;
  }

  function buildGrid(p) {
    var H = [p.hOuter, p.hDiploe, p.hInner];
    var NC = [40, 24, 20];
    var n = NC[0] + NC[1] + NC[2];
    var dx = new Array(n), li = new Array(n), x = new Array(n);
    var tau = [p.tauOuter, p.tauDiploe, p.tauInner];
    var eps0 = [p.epsOuter, p.epsDiploe, p.epsInner];
    var q0v = [p.qOuter, p.qDiploe, p.qInner];
    var tauA = new Array(n), epsA = new Array(n), q0 = new Array(n);
    var idx = 0, pos = 0;
    for (var L = 0; L < 3; L++) {
      var w = H[L] / NC[L];
      for (var k = 0; k < NC[L]; k++) {
        dx[idx] = w; li[idx] = L; x[idx] = pos + w / 2; pos += w;
        tauA[idx] = tau[L]; epsA[idx] = eps0[L]; q0[idx] = q0v[L];
        idx++;
      }
    }
    return { n: n, dx: dx, li: li, x: x, H: H, tau: tauA, eps0: epsA, q0: q0 };
  }

  function runStage(g, p, st, q, samples) {
    var n = g.n;
    var dt = st.T / 160;
    var steps = Math.max(1, Math.round(st.T / dt));
    dt = st.T / steps;
    var a = new Array(n).fill(0);
    var snapEvery = Math.max(1, Math.floor(steps / samples));
    var snaps = [{ t: 0, c: a.slice(), q: q.slice() }];
    var kf = p.kf, alpha = p.alpha;
    for (var s = 0; s < steps; s++) {
      var eps = new Array(n), De = new Array(n), i;
      for (i = 0; i < n; i++) {
        eps[i] = Math.max(1e-8, g.eps0[i] + alpha * (g.q0[i] - q[i]));
        De[i] = st.Db * eps[i] / g.tau[i];
      }
      var cOldArr = a.map(function (v, i2) { return v / eps[i2]; });
      var lo = new Array(n).fill(0), dg = new Array(n), up = new Array(n).fill(0), rhs = new Array(n);
      for (i = 0; i < n; i++) {
        dg[i] = eps[i] / dt + st.nu * st.kr * q[i];
        rhs[i] = eps[i] * cOldArr[i] / dt;
        if (i > 0) {
          var Df = 2 * De[i - 1] * De[i] / (De[i - 1] + De[i]);
          var coef = Df / (0.5 * (g.dx[i - 1] + g.dx[i]) * g.dx[i]);
          lo[i] = -coef; dg[i] += coef;
        }
        if (i < n - 1) {
          var Df2 = 2 * De[i] * De[i + 1] / (De[i] + De[i + 1]);
          var coef2 = Df2 / (0.5 * (g.dx[i] + g.dx[i + 1]) * g.dx[i]);
          up[i] = -coef2; dg[i] += coef2;
        }
      }
      var R = 1 / kf + g.dx[0] / (2 * De[0]);
      var bcoef = 1 / (R * g.dx[0]);
      dg[0] += bcoef; rhs[0] += bcoef * st.Cb;
      var cNewArr = thomas(lo, dg, up, rhs);
      for (i = 0; i < n; i++) {
        var cNew = Math.max(0, cNewArr[i]);
        q[i] = Math.max(0, q[i] - st.kr * cNew * q[i] * dt);
        a[i] = cNew * eps[i];
      }
      if ((s + 1) % snapEvery === 0 || s === steps - 1) {
        snaps.push({ t: (s + 1) * dt, c: a.map(function (v, i2) { return v / Math.max(1e-8, g.eps0[i2] + alpha * (g.q0[i2] - q[i2])); }), q: q.slice() });
      }
    }
    return { snaps: snaps, qFinal: q };
  }

  function porosity(g, p, q) {
    return q.map(function (v, i) { return g.eps0[i] + p.alpha * (g.q0[i] - v); });
  }

  function resistance(g, p, q) {
    /* Layer-series resistance from thickness-weighted mean porosity (Python parity). */
    var R = 0;
    for (var L = 0; L < 3; L++) {
      var epsSum = 0, wSum = 0, thickness = 0;
      for (var i = 0; i < g.n; i++) {
        if (g.li[i] !== L) continue;
        var e = g.eps0[i] + p.alpha * (g.q0[i] - q[i]);
        epsSum += e * g.dx[i]; wSum += g.dx[i]; thickness += g.dx[i];
      }
      var epsMean = epsSum / wSum;
      var sigma = p.sigmaMatrix + p.sigmaFluid * Math.pow(epsMean, p.archieM);
      R += thickness / sigma;
    }
    return R;
  }

  /* p: user params; reagent: 'edta'|'acetic'|'urea'|'seq'.
     Unit consistency: grid in mm, so diffusivities are converted m²/s -> mm²/s (x1e6),
     film coefficient m/s -> mm/s (x1e3), concentrations mol/L -> mol/m³ (x1000). */
  window.runParamSim = function (p) {
    var g = buildGrid(p);
    var R0 = resistance(g, p, g.q0);
    var q0 = g.q0.slice();
    var pv = Object.assign({}, p, { kf: p.kf * 1e3 });
    var stages = {};
    function st(CbMolL, Db, kr, T) {
      return { Cb: CbMolL * 1000, Db: Db * 1e6, kr: kr, nu: 1, T: T };
    }
    if (p.reagent === 'seq') {
      stages.acid = runStage(g, pv, st(8.74, p.dAcid, p.kAcid, 300), q0.slice(), 12);
      stages.edta = runStage(g, pv, st(p.cMain, p.dEdta, p.kEdta, p.tSoak * 60), stages.acid.qFinal.slice(), 80);
    } else if (p.reagent === 'edta') {
      stages.main = runStage(g, pv, st(p.cMain, p.dEdta, p.kEdta, p.tSoak * 60), q0.slice(), 80);
    } else if (p.reagent === 'acetic') {
      stages.main = runStage(g, pv, st(p.cMain, p.dAcid, p.kAcid, p.tSoak * 60), q0.slice(), 80);
    } else { /* urea: swelling, no calcium consumption */
      stages.main = runStage(g, pv, st(p.cMain, p.dUrea, 0, p.tSoak * 60), q0.slice(), 80);
    }
    var finalStage = p.reagent === 'seq' ? stages.edta : stages.main;
    var qFinal = finalStage.qFinal;
    var R1 = resistance(g, p, qFinal);
    /* whole-skull and outer-layer calcium fractions */
    var tot = 0, nat = 0, oTot = 0, oNat = 0;
    for (var i = 0; i < g.n; i++) {
      tot += qFinal[i] * g.dx[i]; nat += g.q0[i] * g.dx[i];
      if (g.li[i] === 0) { oTot += qFinal[i] * g.dx[i]; oNat += g.q0[i] * g.dx[i]; }
    }
    /* Thiele / Biot for the main reagent */
    var krMain = p.reagent === 'acetic' ? p.kAcid : (p.reagent === 'urea' ? 0 : p.kEdta);
    var DbMain = p.reagent === 'acetic' ? p.dAcid : (p.reagent === 'urea' ? p.dUrea : p.dEdta);
    var epsL = [p.epsOuter, p.epsDiploe, p.epsInner];
    var qL = [p.qOuter, p.qDiploe, p.qInner];
    var tauL = [p.tauOuter, p.tauDiploe, p.tauInner];
    var HL = [p.hOuter, p.hDiploe, p.hInner];
    var diag = [];
    for (var L = 0; L < 3; L++) {
      var De = DbMain * 1e6 * epsL[L] / tauL[L]; /* mm²/s */
      var phi = HL[L] * Math.sqrt(krMain * qL[L] / De);
      diag.push({
        layer: L, phi: phi,
        eta: phi > 1e-9 ? Math.tanh(phi) / phi : 1,
        bi: p.kf * 1e3 * HL[L] / De
      });
    }
    return { grid: g, stages: stages, final: finalStage, qFinal: qFinal, R0: R0, R1: R1,
             caWhole: tot / nat, caOuter: oTot / oNat, diag: diag };
  };

  /* Illustrative resistance-vs-frequency curve: resistive plateau below ~100 Hz
     (per Wu thesis), uncalibrated capacitive roll-off above. */
  window.resistanceVsFreq = function (R1, freqs) {
    return freqs.map(function (f) {
      var roll = f <= 100 ? 1 : 1 / Math.sqrt(1 + Math.pow(f / 100, 2) * 0.5);
      return R1 * roll;
    });
  };

  window.porosityOf = porosity;
})();
