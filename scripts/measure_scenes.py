"""Measure the WebGL scenes: draw calls, triangles, GPU objects, frame time, leaks.

    python scripts/measure_scenes.py --label baseline
    python scripts/measure_scenes.py --label after --base http://localhost:8912

Drives headless Chrome over the DevTools protocol with the same launcher as
``capture_screenshots.py`` (no new dependency), loads each 3-D page against a
running server, and reads three.js's own counters (``renderer.info``) after the
page settles. It then:

* times N forced frames while orbiting the camera, finishing the GL queue each
  frame so the number is the real cost of a frame, not of queuing one;
* churns the view state (lead and layer changes, repeated) and reports whether
  the GPU object counts grew -- a leak shows up as geometries or textures that
  never come back down.

**Read the frame times as relative, not absolute.** Headless Chrome here renders
through SwiftShader, a CPU rasteriser; a laptop GPU is one to two orders of
magnitude faster. The draw-call, triangle and leak figures do not depend on the
rasteriser and are comparable across machines.

Writes ``artifacts/perf/<label>.json``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_screenshots import ROOT, Tab, _chrome, _free_port  # noqa: E402

# Each page names its own globals. The snippets use only names the page already
# declares at top level, so nothing in the product exists for the benefit of
# this script. A page that lacks a name reports null instead of failing.
PAGES = {
    "command": {
        "route": "/command.html",
        "frame": "orbit.theta += 0.02; applyCamera(); renderer.render(scene, camera);",
        "churn": "for (const v of ['baseline','truth','model']) setView(v); for (let L=1; L<=10; L++) setLead(L);",
    },
    "landing": {
        "route": "/landing.html",
        "frame": "hero.view.theta += 0.01; heroRender();",
        "churn": "for (let i = 0; i < 5; i++) { heroResize(); heroRender(); }",
    },
    "volume": {
        "route": "/volume.html",
        "frame": "view.theta += 0.02; renderer.render(scene, camera);",
        "churn": "for (let n = 0; n <= 10; n++) setSlice(n); setSlice(0);",
    },
}

INFO = """(() => {
  const r = (typeof renderer !== 'undefined' && renderer);
  if (!r) return null;
  const s = (typeof scene !== 'undefined' && scene);
  let objects = 0; if (s) s.traverse(() => objects++);
  return { calls: r.info.render.calls, triangles: r.info.render.triangles,
           geometries: r.info.memory.geometries, textures: r.info.memory.textures,
           programs: (r.info.programs || []).length, objects,
           pixelRatio: r.getPixelRatio() };
})()"""


def timed(tab: Tab, frame: str, n: int) -> float | None:
    js = f"""(() => {{
      const r = (typeof renderer !== 'undefined' && renderer);
      if (!r) return null;
      const gl = r.getContext();
      {frame} gl.finish();
      const t0 = performance.now();
      for (let i = 0; i < {n}; i++) {{ {frame} gl.finish(); }}
      return (performance.now() - t0) / {n};
    }})()"""
    v = tab.evaluate(js)
    return round(v, 2) if isinstance(v, (int, float)) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument("--base", default="http://localhost:8912")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--settle", type=float, default=6.0)
    ap.add_argument("--pages", nargs="*", default=list(PAGES))
    ap.add_argument("--size", default="1440x900")
    args = ap.parse_args()
    w, h = (int(x) for x in args.size.split("x"))
    try:
        urllib.request.urlopen(args.base + "/api/health", timeout=5)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"server not reachable at {args.base}: {exc}")

    port = _free_port()
    proc = subprocess.Popen([
        _chrome(), "--headless=new", f"--remote-debugging-port={port}",
        f"--user-data-dir={tempfile.mkdtemp(prefix='fbd-perf-')}", "--no-first-run",
        "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist",
        "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out: dict = {"label": args.label, "viewport": args.size, "rasteriser": "SwiftShader (CPU)",
                 "frames": args.frames, "pages": {}}
    try:
        for _ in range(50):
            try:
                targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                page = next(t for t in targets if t.get("type") == "page")
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.2)
        else:
            sys.exit("Chrome did not expose a DevTools target")
        tab = Tab(page["webSocketDebuggerUrl"])
        tab.call("Page.enable")
        tab.call("Emulation.setDeviceMetricsOverride", width=w, height=h,
                 deviceScaleFactor=1, mobile=False)
        for name in args.pages:
            spec = PAGES[name]
            t0 = time.time()
            tab.call("Page.navigate", url=args.base + spec["route"])
            while time.time() - t0 < 30 and tab.evaluate("document.readyState") != "complete":
                time.sleep(0.1)
            tab.pump(args.settle)
            settled = tab.evaluate(INFO)
            ms = timed(tab, spec["frame"], args.frames)
            after_frames = tab.evaluate(INFO)
            tab.evaluate(spec["churn"])
            tab.pump(0.5)
            tab.evaluate(spec["churn"])
            tab.pump(0.5)
            churned = tab.evaluate(INFO)
            leak = None
            if after_frames and churned:
                leak = {k: churned[k] - after_frames[k] for k in ("geometries", "textures", "programs")}
            out["pages"][name] = {"settled": settled, "frame_ms": ms,
                                  "after_frames": after_frames, "after_two_churns": churned,
                                  "growth_after_churn": leak}
            print(f"{name:8s} calls={settled and settled['calls']} tris={settled and settled['triangles']} "
                  f"objects={settled and settled['objects']} frame={ms} ms growth={leak}")
    finally:
        proc.terminate()
    dest = ROOT / "artifacts" / "perf" / f"{args.label}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
