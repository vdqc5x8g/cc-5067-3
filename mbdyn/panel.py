"""Mastering audition panel for the multiband dynamics kernel.

Stdlib-only web app: the kernel (MultibandEngine) is the single source of
truth; every control POSTs to it, every kernel failure comes back as a
status message, and the results area shows the band responses plus the
recombined audition (input/processed WAV, recombination error, per-band
gain reduction, switch-policy log).

Launch:  python3 -m mbdyn.panel [--host 127.0.0.1] [--port 8137]
then open the printed URL in a browser.
"""

import argparse
import io
import json
import math
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .engine import MultibandEngine

BAND_COLORS = ["#4da3ff", "#57d98d", "#ffb84d", "#ff6d6d"]


# ------------------------------------------------------------------- material
def gen_synthetic(fs, freqs, dur=1.2):
    """LF and HF with independent level steps (control verification)."""
    n = int(fs * dur)
    out = []
    for i in range(n):
        t = i / fs
        lf = 0.15 if t < 0.4 else 0.85
        hf = 0.70 if t < 0.8 else 0.10
        out.append(lf * math.sin(math.tau * 55.0 * t)
                   + hf * math.sin(math.tau * 5000.0 * t))
    return out


def gen_drum(fs, freqs, dur=1.0):
    n = int(fs * dur)
    out = [0.0] * n
    for t0 in (0.10, 0.45, 0.80):
        i0 = int(t0 * fs)
        for i in range(i0, n):
            t = (i - i0) / fs
            out[i] += 0.9 * math.exp(-t / 0.09) * math.sin(math.tau * 55.0 * t)
            if t < 0.008:
                out[i] += 0.5 * math.exp(-t / 0.0012) * math.sin(math.tau * 2500.0 * t)
    return out


def gen_sweep(fs, freqs, dur=1.2):
    n = int(fs * dur)
    f0, f1 = 30.0, min(16000.0, 0.45 * fs)
    k = math.log(f1 / f0) / dur
    phase = 0.0
    out = []
    for i in range(n):
        t = i / fs
        phase += math.tau * f0 * math.exp(k * t) / fs
        out.append(0.8 * math.sin(phase))
    return out


def gen_dualtone(fs, freqs, dur=0.8):
    n = int(fs * dur)
    fa, fb = freqs[0], freqs[-1]
    return [0.45 * math.sin(math.tau * fa * i / fs)
            + 0.45 * math.sin(math.tau * fb * i / fs) for i in range(n)]


MATERIALS = {
    "synthetic": gen_synthetic,
    "drum": gen_drum,
    "sweep": gen_sweep,
    "dualtone": gen_dualtone,
}


def wav16_bytes(samples, fs):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(round(fs)))
        frames = bytearray()
        for x in samples:
            v = int(max(-1.0, min(1.0, x)) * 32767.0)
            frames += v.to_bytes(2, "little", signed=True)
        w.writeframes(bytes(frames))
    return buf.getvalue()


# ---------------------------------------------------------------------- model
class PanelModel:
    """Headless controller: the view (HTTP/JS) only talks to this."""

    def __init__(self, fs=44100.0, crossovers=(200.0, 1000.0, 5000.0)):
        self.engine = MultibandEngine(sample_rate=fs, crossovers=crossovers)
        self.lock = threading.Lock()
        self.last_input = []
        self.last_output = []
        self.last_results = None
        # sensible mastering defaults so audition shows real gain riding
        for k in range(self.engine.num_bands):
            self.engine.set_band(k, threshold_db=-24.0, ratio=3.0,
                                 attack_ms=5.0, release_ms=150.0)

    # ---- controls (kernel-driven; kernel errors propagate to the view) ----
    def set_crossover(self, index, freq):
        new = list(self.engine.freqs)
        new[index] = float(freq)
        with self.lock:
            return self.engine.move_crossovers(new, policy="auto")

    def set_band_param(self, index, key, value):
        with self.lock:
            if key in ("threshold_db", "ratio", "knee_db", "attack_ms",
                       "release_ms", "makeup_db"):
                self.engine.set_band(index, **{key: float(value)})
            elif key == "enabled":
                self.engine.set_band(index, enabled=bool(value))
            else:
                raise ValueError(f"unknown band parameter {key!r}")

    def set_flag(self, what, index, on):
        with self.lock:
            if what == "solo":
                self.engine.set_solo(index, on)
            elif what == "mute":
                self.engine.set_mute(index, on)
            elif what == "bypass":
                self.engine.set_bypass(on)
            elif what == "processing":
                self.engine.set_processing(on)
            else:
                raise ValueError(f"unknown flag {what!r}")

    # ---- analysis ----
    def response_curves(self, points=120):
        eng = self.engine
        fmin, fmax = 20.0, eng.fs * 0.5
        grid = [fmin * (fmax / fmin) ** (i / (points - 1)) for i in range(points)]
        xo = eng._core.crossover
        band_db = []
        for k in range(eng.num_bands):
            band_db.append([20.0 * math.log10(abs(xo.H_band(k, math.tau * f / eng.fs)) + 1e-15)
                            for f in grid])
        sum_db = [20.0 * math.log10(abs(xo.H_sum(math.tau * f / eng.fs)) + 1e-15)
                  for f in grid]
        return grid, band_db, sum_db

    def audition(self, material):
        if material not in MATERIALS:
            raise ValueError(f"unknown material {material!r}")
        eng = self.engine
        with self.lock:
            x = MATERIALS[material](eng.fs, eng.freqs)
            # the summed audition is measured with solo/mute neutralized
            saved = [(c.solo, c.mute) for c in eng.bands]
            for c in eng.bands:
                c.solo = c.mute = False
            try:
                eng.reset_states()
                eng.probe_gains = True
                y = []
                for i in range(0, len(x), 256):
                    y.extend(eng.process(x[i:i + 256]))
                eng.probe_gains = False
                max_gr = [min(0.0, min(tr)) if tr else 0.0 for tr in eng.gain_trace]
                # recombination check: same material, processing off, vs the
                # declared allpass reference
                eng.reset_states(clear_log=False)
                was_on = eng.processing_on
                eng.processing_on = False
                u = []
                for i in range(0, len(x), 256):
                    u.extend(eng.process(x[i:i + 256]))
                eng.processing_on = was_on
            finally:
                for c, (s, m) in zip(eng.bands, saved):
                    c.solo, c.mute = s, m
            ref = eng.reference_allpass().process(x)
            recomb = max(abs(a - b) for a, b in zip(u, ref)) if x else 0.0
            self.last_input = x
            self.last_output = y
            self.last_results = {
                "material": material,
                "samples": len(x),
                "peak_in": max(abs(v) for v in x),
                "peak_out": max(abs(v) for v in y),
                "recomb_error": recomb,
                "max_gr_db": max_gr,
                "switch_log": list(eng.switch_log[-8:]),
                "latency_samples": eng.latency_samples,
            }
            return self.last_results

    def state_dict(self):
        eng = self.engine
        grid, band_db, sum_db = self.response_curves()
        return {
            "fs": eng.fs,
            "freqs": list(eng.freqs),
            "pending_freqs": eng._pending_freqs,
            "bypass": eng.bypass,
            "processing_on": eng.processing_on,
            "latency_samples": eng.latency_samples,
            "warmup_frames": eng.warmup.maxlen,
            "limits": {
                "min_hz": eng.MIN_CROSSOVER_HZ,
                "max_hz": eng.MAX_CROSSOVER_RATIO * eng.fs,
                "min_spacing_oct": eng.MIN_SPACING_OCT,
                "min_bands": eng.MIN_BANDS,
                "max_bands": eng.MAX_BANDS,
            },
            "bands": [dict(c.__dict__) for c in eng.bands],
            "ranges": eng.band_ranges(),
            "colors": BAND_COLORS[:eng.num_bands],
            "grid": grid,
            "band_db": band_db,
            "sum_db": sum_db,
            "switch_log": list(eng.switch_log[-8:]),
            "has_audio": bool(self.last_output),
            "results": self.last_results,
            "materials": sorted(MATERIALS),
        }


# ---------------------------------------------------------------------- pages
PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>mbdyn 母带试听面板</title>
<style>
 body{font:13px/1.4 system-ui,sans-serif;background:#16181d;color:#dde;margin:0;padding:14px}
 h1{font-size:16px;margin:0 0 10px}
 #wrap{display:flex;gap:14px;flex-wrap:wrap}
 #cv{background:#0c0e12;border:1px solid #333;border-radius:6px;cursor:crosshair}
 #side{min-width:280px;flex:1}
 .row{display:flex;align-items:center;gap:6px;margin:4px 0}
 button{background:#262b36;color:#dde;border:1px solid #3a4152;border-radius:4px;padding:3px 10px;cursor:pointer}
 button.on{background:#2f6dbf;border-color:#4da3ff}
 button:disabled{opacity:.4}
 .sw{width:12px;height:12px;border-radius:3px;display:inline-block}
 #status{color:#ff8080;min-height:18px;margin:6px 0}
 #results{background:#0c0e12;border:1px solid #333;border-radius:6px;padding:8px;white-space:pre-wrap;font-family:ui-monospace,monospace;font-size:12px;max-height:220px;overflow:auto}
 audio{width:260px;height:32px}
 label{color:#9ab}
</style></head><body>
<h1>mbdyn 多频带动态 · 母带试听面板</h1>
<div id="wrap">
 <canvas id="cv" width="920" height="380"></canvas>
 <div id="side">
  <div class="row"><label>全局:</label>
   <button id="bypass">旁路</button><button id="proc">处理开</button></div>
  <div id="bands"></div>
  <div class="row"><label>素材:</label><select id="mat"></select>
   <button id="run">运行试听</button></div>
  <div class="row"><label>输入:</label><audio id="ain" controls></audio></div>
  <div class="row"><label>合成:</label><audio id="aout" controls></audio></div>
 </div>
</div>
<div id="status"></div>
<div id="results">结果区:尚未运行试听</div>
<script>
let S=null, drag=null;
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
const W=cv.width, H=cv.height, DB_TOP=12, DB_BOT=-60;
function fx(f){const a=Math.log10(20),b=Math.log10(S.fs/2);return (Math.log10(f)-a)/(b-a)*W;}
function ff(x){const a=Math.log10(20),b=Math.log10(S.fs/2);return Math.pow(10,a+x/W*(b-a));}
function fy(db){return (DB_TOP-db)/(DB_TOP-DB_BOT)*H;}
async function api(url,obj){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(obj||{})});
 const j=await r.json(); if(!r.ok){throw new Error(j.error||r.status);} return j;}
async function refresh(){S=await (await fetch('/api/state')).json();draw();controls();}
function draw(){ctx.clearRect(0,0,W,H);
 ctx.strokeStyle='#232833';ctx.fillStyle='#567';ctx.font='10px sans-serif';
 [50,100,200,500,1000,2000,5000,10000,20000].forEach(f=>{if(f<S.fs/2){const x=fx(f);
  ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();ctx.fillText(f>=1000?(f/1000)+'k':f,x+2,H-4);}});
 for(let db=DB_BOT;db<=DB_TOP;db+=12){const y=fy(db);ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();
  ctx.fillText(db+'dB',4,y-2);}
 S.band_db.forEach((c,k)=>{ctx.strokeStyle=S.colors[k];ctx.lineWidth=1.6;ctx.beginPath();
  c.forEach((db,i)=>{const x=fx(S.grid[i]),y=fy(Math.max(DB_BOT,Math.min(DB_TOP,db)));
   i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();});
 ctx.strokeStyle='#fff';ctx.lineWidth=2.4;ctx.beginPath();
 S.sum_db.forEach((db,i)=>{const x=fx(S.grid[i]),y=fy(Math.max(DB_BOT,Math.min(DB_TOP,db)));
  i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();ctx.lineWidth=1;
 S.freqs.forEach((f,i)=>{const x=fx(drag&&drag.i===i?drag.f:f);
  ctx.strokeStyle='#888';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='#ccc';ctx.beginPath();ctx.moveTo(x-6,0);ctx.lineTo(x+6,0);ctx.lineTo(x,10);ctx.fill();});
 if(S.pending_freqs){ctx.fillStyle='#fa0';ctx.fillText('换带进行中 → '+S.pending_freqs.map(f=>f.toFixed(0)).join(' / '),10,14);}}
function controls(){
 const bp=document.getElementById('bypass');bp.className=S.bypass?'on':'';
 bp.textContent=S.bypass?'旁路中(位精确直通)':'旁路';
 const pr=document.getElementById('proc');pr.className=S.processing_on?'on':'';
 pr.textContent=S.processing_on?'处理开':'处理关(单位增益)';
 const bd=document.getElementById('bands');bd.innerHTML='';
 S.bands.forEach((b,k)=>{const r=document.createElement('div');r.className='row';
  const lo=S.ranges[k][0],hi=S.ranges[k][1];
  r.innerHTML='<span class="sw" style="background:'+S.colors[k]+'"></span><label>'+
   (lo<1000?lo.toFixed(0):lo/1000+'k')+'–'+(hi>=1000?hi/1000+'k':hi.toFixed(0))+' Hz</label>';
  [['solo','独奏'],['mute','静音'],['enabled','动态']].forEach(([key,lab])=>{
   const btn=document.createElement('button');const on=key==='enabled'?b.enabled:b[key];
   btn.textContent=lab+(key==='enabled'?(on?'开':'关'):(on?'中':''));btn.className=on?'on':'';
   btn.onclick=async()=>{try{
     if(key==='enabled')await api('/api/band',{index:k,key:'enabled',value:!on});
     else await api('/api/'+key,{index:k,on:!on});
     document.getElementById('status').textContent='';await refresh();}
    catch(e){document.getElementById('status').textContent='内核拒绝: '+e.message;}};
   r.appendChild(btn);});
  bd.appendChild(r);});
 const ms=document.getElementById('mat');if(!ms.options.length)
  S.materials.forEach(m=>{const o=document.createElement('option');o.value=o.textContent=m;ms.add(o);});
 if(S.results)showResults(S.results);}
function showResults(R){document.getElementById('results').textContent=
 '素材 '+R.material+'  样本 '+R.samples+'  延迟 '+R.latency_samples+' 样本\\n'+
 '峰值 in '+R.peak_in.toFixed(3)+' / out '+R.peak_out.toFixed(3)+'\\n'+
 '重组误差(处理关 vs 声明全通) '+R.recomb_error.toExponential(3)+'\\n'+
 '各带最大增益衰减 dB: '+R.max_gr_db.map(v=>v.toFixed(1)).join(', ')+'\\n'+
 '换带策略日志:\\n'+(R.switch_log.map(e=>'  n='+e.n+' '+e.policy+' '+(e.freqs?e.freqs.map(f=>f.toFixed(0)).join('/'):'')).join('\\n')||'  (无)');}
cv.addEventListener('mousedown',e=>{const x=e.offsetX;let best=null;
 S.freqs.forEach((f,i)=>{const d=Math.abs(fx(f)-x);if(d<8&&(!best||d<best.d))best={i,d,f};});
 if(best)drag=best;});
cv.addEventListener('mousemove',e=>{if(drag){drag.f=Math.min(Math.max(ff(e.offsetX),S.limits.min_hz),S.limits.max_hz);draw();}});
cv.addEventListener('mouseup',async e=>{if(!drag)return;const d=drag;drag=null;
 try{const r=await api('/api/crossover',{index:d.i,freq:Math.round(d.f*10)/10});
  document.getElementById('status').textContent='';await refresh();}
 catch(err){document.getElementById('status').textContent='内核拒绝: '+err.message;await refresh();}});
document.getElementById('bypass').onclick=async()=>{await api('/api/bypass',{on:!S.bypass});await refresh();};
document.getElementById('proc').onclick=async()=>{await api('/api/processing',{on:!S.processing_on});await refresh();};
document.getElementById('run').onclick=async btn=>{const b=btn.target;b.disabled=true;
 document.getElementById('status').textContent='处理中…';
 try{const R=await api('/api/audition',{material:document.getElementById('mat').value});
  showResults(R);document.getElementById('status').textContent='';
  const t=Date.now();
  document.getElementById('ain').src='/api/audio/input.wav?t='+t;
  document.getElementById('aout').src='/api/audio/processed.wav?t='+t;
  await refresh();}
 catch(e){document.getElementById('status').textContent='试听失败: '+e.message;}
 b.disabled=false;};
refresh();
</script></body></html>"""


# ---------------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    @property
    def model(self):
        return self.server.model

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/state"):
            self._json(self.model.state_dict())
        elif self.path.startswith("/api/audio/input.wav"):
            self._wav(self.model.last_input)
        elif self.path.startswith("/api/audio/processed.wav"):
            self._wav(self.model.last_output)
        else:
            self._json({"error": "not found"}, 404)

    def _wav(self, samples):
        if not samples:
            self._json({"error": "no audio yet; run an audition first"}, 404)
            return
        body = wav16_bytes(samples, self.model.engine.fs)
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            d = self._body()
            if self.path == "/api/crossover":
                policy = self.model.set_crossover(int(d["index"]), float(d["freq"]))
                self._json({"ok": True, "policy": policy})
            elif self.path == "/api/band":
                self.model.set_band_param(int(d["index"]), d["key"], d["value"])
                self._json({"ok": True})
            elif self.path in ("/api/solo", "/api/mute"):
                self.model.set_flag(self.path[5:], int(d["index"]), bool(d["on"]))
                self._json({"ok": True})
            elif self.path in ("/api/bypass", "/api/processing"):
                self.model.set_flag(self.path[5:], None, bool(d["on"]))
                self._json({"ok": True})
            elif self.path == "/api/audition":
                self._json(self.model.audition(d.get("material", "synthetic")))
            elif self.path == "/api/reset":
                with self.model.lock:
                    self.model.engine.reset_states()
                self._json({"ok": True})
            else:
                self._json({"error": "not found"}, 404)
        except (ValueError, KeyError, TypeError) as e:
            self._json({"error": str(e)}, 400)


def make_server(model=None, host="127.0.0.1", port=8137):
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.model = model or PanelModel()
    return srv


def main(argv=None):
    ap = argparse.ArgumentParser(description="mbdyn mastering audition panel")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8137)
    args = ap.parse_args(argv)
    srv = make_server(host=args.host, port=args.port)
    url = f"http://{args.host}:{srv.server_address[1]}/"
    print(f"mbdyn 母带试听面板已启动: {url}")
    print("在浏览器中打开上面的地址;Ctrl+C 停止。")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
