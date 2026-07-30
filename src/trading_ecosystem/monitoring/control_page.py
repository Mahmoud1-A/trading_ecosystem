"""Interactive trade control panel HTML (served at /control)."""

CONTROL_HTML = r"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>TE — لوحة تحكم التداول</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet"/>
  <style>
    :root {
      --bg0: #0c1116;
      --bg1: #141b22;
      --bg2: #1c2630;
      --line: #2a3642;
      --text: #e7edf2;
      --muted: #8b9aab;
      --accent: #d4a24c;
      --accent-dim: #9a7433;
      --ok: #3d9b74;
      --bad: #c45c5c;
      --warn: #d4a24c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "IBM Plex Sans", sans-serif;
      color: var(--text);
      background:
        radial-gradient(900px 420px at 85% -10%, rgba(212,162,76,.12), transparent 55%),
        radial-gradient(700px 380px at 0% 100%, rgba(61,155,116,.08), transparent 50%),
        linear-gradient(165deg, #0a0e12 0%, var(--bg0) 45%, #101820 100%);
    }
    .wrap { max-width: 1180px; margin: 0 auto; padding: 1.5rem 1.25rem 3rem; }
    header {
      display: flex; flex-wrap: wrap; gap: 1rem; justify-content: space-between;
      align-items: flex-end; margin-bottom: 1.5rem;
      border-bottom: 1px solid var(--line); padding-bottom: 1rem;
    }
    .brand { font-size: 1.55rem; font-weight: 700; letter-spacing: -.02em; }
    .brand span { color: var(--accent); }
    .sub { color: var(--muted); font-size: .92rem; margin-top: .35rem; max-width: 36rem; }
    nav a {
      color: var(--muted); text-decoration: none; margin-inline-start: 1rem; font-size: .9rem;
    }
    nav a:hover { color: var(--accent); }
    .panel {
      background: color-mix(in srgb, var(--bg1) 92%, transparent);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 1rem 1.1rem;
      margin-bottom: 1rem;
    }
    .panel h2 {
      margin: 0 0 .85rem; font-size: 1rem; font-weight: 600;
      color: var(--accent); letter-spacing: .02em; text-transform: uppercase;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: .75rem;
    }
    .metric {
      background: var(--bg2);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: .75rem .85rem;
    }
    .metric div { color: var(--muted); font-size: .75rem; margin-bottom: .25rem; }
    .metric strong { font-size: 1.15rem; font-weight: 600; font-variant-numeric: tabular-nums; }
    .warn { color: var(--warn); }
    .bad { color: var(--bad); }
    .ok { color: var(--ok); }
    .row { display: flex; flex-wrap: wrap; gap: .6rem; align-items: center; }
    button, .btn {
      font-family: inherit; cursor: pointer; border: 1px solid var(--line);
      background: var(--bg2); color: var(--text); border-radius: 8px;
      padding: .55rem 1rem; font-weight: 600; font-size: .9rem;
    }
    button.primary { background: var(--accent); color: #1a1206; border-color: var(--accent-dim); }
    button.danger { background: #3a1f1f; border-color: #6a3030; color: #f0cfcf; }
    button:disabled { opacity: .45; cursor: not-allowed; }
    button:hover:not(:disabled) { filter: brightness(1.08); }
    input[type=password], input[type=text] {
      font-family: "IBM Plex Mono", monospace;
      background: var(--bg0); border: 1px solid var(--line); color: var(--text);
      border-radius: 8px; padding: .55rem .75rem; min-width: 220px; font-size: .85rem;
    }
    .pill {
      display: inline-flex; align-items: center; gap: .4rem;
      padding: .25rem .65rem; border-radius: 999px; font-size: .8rem; font-weight: 600;
      border: 1px solid var(--line); background: var(--bg2);
    }
    .dot { width: .55rem; height: .55rem; border-radius: 50%; background: var(--muted); }
    .dot.on { background: var(--ok); box-shadow: 0 0 0 3px rgba(61,155,116,.2); }
    .dot.off { background: var(--bad); }
    .dot.unk { background: var(--warn); }
    table { width: 100%; border-collapse: collapse; font-size: .85rem; }
    th, td { text-align: start; padding: .45rem .35rem; border-bottom: 1px solid var(--line); }
    th { color: var(--muted); font-weight: 500; }
    code, .mono { font-family: "IBM Plex Mono", monospace; font-size: .8rem; }
    pre.log {
      margin: 0; max-height: 280px; overflow: auto; direction: ltr; text-align: left;
      background: #070a0d; border: 1px solid var(--line); border-radius: 8px;
      padding: .75rem; font-family: "IBM Plex Mono", monospace; font-size: .72rem;
      line-height: 1.45; color: #b8c6d4; white-space: pre-wrap;
    }
    .flash {
      min-height: 1.25rem; color: var(--muted); font-size: .85rem; margin-top: .6rem;
    }
    .flash.err { color: var(--bad); }
    .flash.ok { color: var(--ok); }
    .two {
      display: grid; grid-template-columns: 1.2fr .8fr; gap: 1rem;
    }
    @media (max-width: 860px) {
      .two { grid-template-columns: 1fr; }
      input[type=password] { min-width: 0; width: 100%; }
    }
    ul.dec { margin: 0; padding: 0; list-style: none; max-height: 260px; overflow: auto; }
    ul.dec li {
      border-bottom: 1px solid var(--line); padding: .45rem 0;
      font-family: "IBM Plex Mono", monospace; font-size: .72rem; direction: ltr; text-align: left;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <div class="brand">TE <span>Control</span> — لوحة التداول</div>
        <p class="sub">تشغيل/إيقاف المسار الورقي، مراقبة الأسهم وقرارات Master Risk Manager. الاكتشاف يبقى على اللابتوب.</p>
      </div>
      <nav>
        <a href="/control">لوحة التحكم</a>
        <a href="/">المراقب الكامل</a>
        <a href="/health">Health</a>
      </nav>
    </header>
    <p class="sub" style="margin:-.5rem 0 1rem">
      هذه هي الواجهة. لا تفتح روابط JSON الخام — البيانات تُعرض هنا تلقائياً.
    </p>

    <section class="panel">
      <h2>التداول الورقي</h2>
      <div class="row" style="margin-bottom:.85rem">
        <span class="pill"><span id="dotPaper" class="dot unk"></span> Paper: <strong id="paperActive">…</strong></span>
        <span class="pill"><span id="dotMon" class="dot unk"></span> Monitor: <strong id="monActive">…</strong></span>
        <span class="pill mono" id="hostHint">…</span>
      </div>
      <div class="row">
        <input id="token" type="password" placeholder="CONTROL_TOKEN" autocomplete="off"/>
        <button type="button" onclick="saveToken()">حفظ التوكن</button>
        <button class="primary" type="button" onclick="act('start')">تشغيل التداول</button>
        <button class="danger" type="button" onclick="act('stop')">إيقاف</button>
        <button type="button" onclick="act('restart')">إعادة تشغيل</button>
        <button type="button" onclick="refreshAll()">تحديث</button>
      </div>
      <div id="flash" class="flash">أدخل CONTROL_TOKEN من ملف .env ثم احفظه في المتصفح.</div>
    </section>

    <section class="panel">
      <h2>الاكتشاف — اختيار المحفظة / تشغيل</h2>
      <p class="sub" style="color:var(--muted);font-size:.85rem;margin:0 0 .8rem">
        التداول يستخدم محفظة <strong>ETF المجمّعة الأخيرة</strong>.
        يمكنك تشغيل اكتشاف منفصل لمحفظة <strong>العملات الرقمية</strong> دون الكتابة فوق كمّ التداول.
      </p>
      <div class="row" style="margin-bottom:.85rem">
        <span class="pill"><span id="dotDisc" class="dot unk"></span> Discovery: <strong id="discActive">…</strong></span>
        <span class="pill mono" id="discUniverse">universe: —</span>
        <span class="pill mono" id="discGen">gen: —</span>
      </div>
      <div class="row">
        <select id="universeSelect" style="font-family:inherit;background:var(--bg0);color:var(--text);border:1px solid var(--line);border-radius:8px;padding:.55rem .75rem;min-width:220px"></select>
        <button type="button" onclick="selectUniverse()">تعيين المحفظة</button>
        <button class="primary" type="button" onclick="discAct('start')">تشغيل الاكتشاف</button>
        <button class="danger" type="button" onclick="discAct('stop')">إيقاف الاكتشاف</button>
      </div>
      <div id="discFlash" class="flash"></div>
      <div class="grid" style="margin-top:.85rem" id="universeCards"></div>
    </section>

    <section class="panel">
      <h2>حالة الحساب / المخاطر</h2>
      <div class="grid" id="riskGrid"></div>
    </section>

    <section class="panel">
      <h2>Paper Trading</h2>
      <div class="grid" id="paperGrid"></div>
      <p class="sub" style="margin:.8rem 0 0;color:var(--muted);font-size:.85rem">
        Live sleeve: <code id="sleeveInfo">—</code>
      </p>
    </section>

    <div class="two">
      <section class="panel">
        <h2>قرارات المخاطر الأخيرة</h2>
        <ul class="dec" id="decisions"><li>…</li></ul>
      </section>
      <section class="panel">
        <h2>سياسة MRM (Prop)</h2>
        <table>
          <thead><tr><th>القاعدة</th><th>القيمة</th></tr></thead>
          <tbody id="propBody"></tbody>
        </table>
      </section>
    </div>

    <section class="panel">
      <h2>تنبيهات</h2>
      <table>
        <thead><tr><th>الوقت</th><th>المستوى</th><th>الرسالة</th></tr></thead>
        <tbody id="alertsBody"></tbody>
      </table>
    </section>

    <section class="panel">
      <h2>سجل te-paper</h2>
      <pre class="log" id="logs">(تحميل…)</pre>
    </section>
  </div>

  <script>
    const TOKEN_KEY = "te_control_token";
    const $ = (id) => document.getElementById(id);

    function token() {
      return localStorage.getItem(TOKEN_KEY) || $("token").value.trim();
    }
    function saveToken() {
      const t = $("token").value.trim();
      if (t) localStorage.setItem(TOKEN_KEY, t);
      flash("تم حفظ التوكن في هذا المتصفح.", "ok");
    }
    function flash(msg, kind) {
      const el = $("flash");
      el.textContent = msg;
      el.className = "flash" + (kind ? " " + kind : "");
    }
    function pct(v) {
      if (v === null || v === undefined || v === "") return "—";
      const n = Number(v);
      if (Number.isNaN(n)) return String(v);
      return (Math.abs(n) <= 1.5 ? n * 100 : n).toFixed(2) + "%";
    }
    function money(v) {
      if (v === null || v === undefined) return "—";
      return Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 });
    }
    function setDot(id, running) {
      const el = $(id);
      el.className = "dot " + (running === true ? "on" : running === false ? "off" : "unk");
    }
    function metric(label, value, cls) {
      return `<div class="metric"><div>${label}</div><strong class="${cls||""}">${value}</strong></div>`;
    }
    async function api(path, opts={}) {
      const headers = Object.assign({}, opts.headers || {});
      const t = token();
      if (t) headers["X-Control-Token"] = t;
      const res = await fetch(path, Object.assign({}, opts, { headers }));
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || data.error || res.statusText);
      return data;
    }
    function renderStatus(s) {
      const p = s.services?.paper || {};
      const m = s.services?.monitor || {};
      $("paperActive").textContent = p.active || "—";
      $("monActive").textContent = m.active || "—";
      setDot("dotPaper", p.running);
      setDot("dotMon", m.running);
      $("hostHint").textContent = s.control?.systemd
        ? "systemd · Droplet"
        : "محلي · start/stop عبر CLI";

      const dc = s.discovery_control || {};
      const discRun = !!dc.process_running || dc.monitor_status === "running";
      $("discActive").textContent = discRun ? (dc.monitor_status || "running") : (dc.monitor_status || "idle");
      setDot("dotDisc", discRun);
      $("discUniverse").textContent = "universe: " + (dc.active_universe || "—");
      $("discGen").textContent = "gen: " + (dc.generation ?? "—");
      const sel = $("universeSelect");
      const cur = sel.value;
      sel.innerHTML = (dc.universes || []).map(u =>
        `<option value="${u.id}" ${u.active ? "selected" : ""}>${u.label_en || u.id} (${u.count})</option>`
      ).join("");
      if (cur) sel.value = cur;
      $("universeCards").innerHTML = (dc.universes || []).map(u =>
        metric(
          (u.active ? "● " : "") + (u.label || u.id),
          (u.count || 0) + " رموز" + (u.trade_default ? " · تداول" : " · بحث")
        )
      ).join("");
      if (!dc.allowed) {
        $("discFlash").textContent = "تشغيل الاكتشاف معطّل على هذا السيرفر — شغّله من اللابتوب.";
        $("discFlash").className = "flash";
      }

      const snap = s.snapshot || {};
      const ddWarn = (snap.daily_dd_pct || 0) > 3 ? "warn" : "";
      const tdWarn = (snap.total_dd_pct || 0) > 7 ? "warn" : "";
      $("riskGrid").innerHTML = [
        metric("Equity", money(snap.equity)),
        metric("Daily DD", snap.daily_dd_pct ?? "—", ddWarn),
        metric("Total DD", snap.total_dd_pct ?? "—", tdWarn),
        metric("Halted", String(snap.halted), snap.halted ? "bad" : "ok"),
        metric("Daily Halt", String(snap.daily_halted), snap.daily_halted ? "bad" : ""),
        metric("Last bar", snap.last_bar_ts || "—"),
      ].join("");

      const paper = snap.paper || {};
      const sim = paper.sim || {};
      const alp = paper.alpaca || {};
      $("paperGrid").innerHTML = [
        metric("Alpaca Equity", money(alp.equity)),
        metric("Alpaca Ann", pct(alp.ann_return)),
        metric("Alpaca Positions", alp.positions ?? "—"),
        metric("Alpaca Updated", alp.updated_at || "—"),
        metric("Sim Equity", money(sim.equity)),
        metric("Sim Ann", pct(sim.ann_return)),
        metric("Sim Positions", sim.positions ?? "—"),
        metric("Sim Updated", sim.updated_at || "—"),
      ].join("");

      const sleeve = s.sleeve || {};
      $("sleeveInfo").textContent = sleeve.exists
        ? `${sleeve.count} strategies · ${sleeve.path}`
        : "missing live sleeve";

      const dec = (snap.recent_decisions || []).slice(0, 25);
      $("decisions").innerHTML = dec.length
        ? dec.map(d => `<li>${escapeHtml(typeof d === "string" ? d : JSON.stringify(d))}</li>`).join("")
        : "<li>لا قرارات بعد</li>";

      const alerts = (snap.alerts || []).slice(0, 30);
      $("alertsBody").innerHTML = alerts.length
        ? alerts.map(a => `<tr><td class="mono">${escapeHtml(a.ts||"")}</td><td>${escapeHtml(a.level||"")}</td><td>${escapeHtml(a.message||"")}</td></tr>`).join("")
        : `<tr><td colspan="3">لا تنبيهات</td></tr>`;

      const prop = (s.prop?.summary || []);
      const costRows = Object.entries(s.prop?.execution_costs || {}).map(([id, c]) => {
        const row = c || {};
        return `<tr><td>Costs · ${escapeHtml(id)}</td><td class="mono">slip ${row.slippage_bps}bps · maker ${row.maker_fee_bps}bps · taker ${row.taker_fee_bps}bps</td></tr>`;
      }).join("");
      $("propBody").innerHTML = prop.map(r =>
        `<tr><td>${escapeHtml(r.label)}</td><td class="mono">${escapeHtml(String(r.value))}</td></tr>`
      ).join("") + costRows;
    }
    function escapeHtml(s) {
      return String(s).replace(/[&<>"']/g, c => ({
        "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
      })[c]);
    }
    async function refreshAll() {
      try {
        const s = await api("/api/control/status");
        renderStatus(s);
        const logs = await api("/api/control/logs?lines=100");
        $("logs").textContent = (logs.lines || []).join("\n");
        if (!s.control?.writes_enabled) {
          flash("القراءة تعمل. لتشغيل/إيقاف: ضع CONTROL_TOKEN في .env على السيرفر وفي الحقل أعلاه.", "");
        } else {
          flash("تم التحديث · " + new Date().toLocaleTimeString(), "ok");
        }
      } catch (e) {
        flash(String(e.message || e), "err");
      }
    }
    async function act(action) {
      flash("جاري التنفيذ…", "");
      try {
        const r = await api("/api/control/paper/" + action, { method: "POST" });
        flash((r.ok ? "تم: " : "فشل: ") + action + (r.output ? " — " + r.output : ""), r.ok ? "ok" : "err");
        await refreshAll();
      } catch (e) {
        flash(String(e.message || e), "err");
      }
    }
    async function discAct(action) {
      const uni = $("universeSelect").value;
      $("discFlash").textContent = "جاري…";
      $("discFlash").className = "flash";
      try {
        const q = action === "start" && uni ? ("?universe=" + encodeURIComponent(uni)) : "";
        const r = await api("/api/control/discovery/" + action + q, { method: "POST" });
        $("discFlash").textContent = r.ok ? ("تم: " + action + (r.pid ? " pid=" + r.pid : "")) : (r.error || "فشل");
        $("discFlash").className = "flash " + (r.ok ? "ok" : "err");
        await refreshAll();
      } catch (e) {
        $("discFlash").textContent = String(e.message || e);
        $("discFlash").className = "flash err";
      }
    }
    async function selectUniverse() {
      const uni = $("universeSelect").value;
      $("discFlash").textContent = "تعيين المحفظة…";
      try {
        const r = await api("/api/control/discovery/universe?universe=" + encodeURIComponent(uni), { method: "POST" });
        $("discFlash").textContent = "المحفظة النشطة: " + (r.universe?.id || uni);
        $("discFlash").className = "flash ok";
        await refreshAll();
      } catch (e) {
        $("discFlash").textContent = String(e.message || e);
        $("discFlash").className = "flash err";
      }
    }
    (function init() {
      const saved = localStorage.getItem(TOKEN_KEY);
      if (saved) $("token").value = saved;
      refreshAll();
      setInterval(refreshAll, 15000);
    })();
  </script>
</body>
</html>
"""
