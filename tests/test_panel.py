"""Panel: headless model drives the kernel, kernel failures feed back,
and the HTTP app serves the page, state, control and audition routes.
"""

import json
import threading
import urllib.request
import urllib.error

from util import run, tone

from mbdyn.panel import PanelModel, make_server, wav16_bytes

FS = 24000.0


def _model():
    return PanelModel(fs=FS, crossovers=(250.0, 2500.0))


def test_model_controls_kernel_and_failures_return():
    m = _model()
    policy = m.set_crossover(0, 300.0)
    assert policy in ("xfade", "map", "deferred")
    assert m.engine.freqs[0] == 300.0 or m.engine._pending_freqs[0] == 300.0
    for bad in (10.0, 2000.0):  # below floor / violates 0.5-oct spacing
        try:
            m.set_crossover(0, bad)
        except ValueError as e:
            assert "Hz" in str(e) or "spacing" in str(e)
        else:
            raise AssertionError(f"model accepted {bad} Hz")
    m.set_flag("solo", 1, True)
    assert m.engine.bands[1].solo
    m.set_flag("bypass", None, True)
    assert m.engine.bypass
    m.set_band_param(0, "threshold_db", -30.0)
    assert m.engine.bands[0].threshold_db == -30.0
    try:
        m.set_band_param(0, "warp_factor", 9)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown band parameter accepted")


def test_model_audition_results_and_wav():
    m = _model()
    res = m.audition("synthetic")
    assert res["recomb_error"] < 1e-9, res["recomb_error"]
    assert res["samples"] == len(m.last_input) == len(m.last_output)
    assert min(res["max_gr_db"]) < -1.0, "audition never compressed"
    assert len(res["max_gr_db"]) == m.engine.num_bands
    wav = wav16_bytes(m.last_output, FS)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    # solo state must be verifiable through the model as well
    m.set_flag("solo", 0, True)
    st = m.state_dict()
    assert st["bands"][0]["solo"] and st["limits"]["min_hz"] == 40.0
    assert len(st["band_db"]) == 3 and len(st["sum_db"]) == len(st["grid"])
    assert max(abs(v) for v in st["sum_db"]) < 1e-6, "sum curve not flat"


def _post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_panel_end_to_end():
    srv = make_server(model=_model(), port=0)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"
    try:
        with urllib.request.urlopen(base + "/") as r:
            html = r.read().decode()
        assert r.status == 200 and "<canvas" in html and "api/state" in html

        with urllib.request.urlopen(base + "/api/state") as r:
            st = json.loads(r.read())
        assert st["freqs"] == [250.0, 2500.0] and len(st["bands"]) == 3

        code, j = _post(base + "/api/crossover", {"index": 0, "freq": 320.0})
        assert code == 200 and j["ok"]
        code, j = _post(base + "/api/crossover", {"index": 0, "freq": 2400.0})
        assert code == 400 and "error" in j  # spacing violation feeds back

        code, j = _post(base + "/api/solo", {"index": 2, "on": True})
        assert code == 200
        with urllib.request.urlopen(base + "/api/state") as r:
            st = json.loads(r.read())
        assert st["bands"][2]["solo"]

        code, res = _post(base + "/api/audition", {"material": "drum"})
        assert code == 200 and res["recomb_error"] < 1e-9
        with urllib.request.urlopen(base + "/api/audio/processed.wav") as r:
            assert r.status == 200 and r.read(4) == b"RIFF"
        with urllib.request.urlopen(base + "/api/audio/input.wav") as r:
            assert r.status == 200
    finally:
        srv.shutdown()
        srv.server_close()
