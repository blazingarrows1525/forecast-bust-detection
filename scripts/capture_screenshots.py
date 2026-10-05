"""Capture every frontend route at fixed viewports, with evidence, via headless Chrome.

    python scripts/capture_screenshots.py --label baseline
    python scripts/capture_screenshots.py --label after --base http://localhost:8912

Drives the locally installed Chrome over the DevTools protocol, so it needs no
new dependency (websocket-client is already in the environment) and runs fully
offline against the local server. For each (route, viewport) it saves a
viewport PNG, and for the widest and narrowest sizes a full-page PNG, under
``artifacts/frontend/<label>/``. It also writes ``report.json`` with, per
capture: page title, load time, document height, horizontal overflow,
WebGL2 availability, console errors, uncaught exceptions and failed requests.

The same script captures "before" and "after", with identical viewports, waits
and data, so the comparison is fair. The server must already be running
(``python -m uvicorn fbd.api.app:app --app-dir src --port 8912``).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket  # websocket-client

ROOT = Path(__file__).resolve().parents[1]
CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]
ROUTES = {"landing": "/landing.html", "index": "/", "command": "/command.html",
          "volume": "/volume.html"}
VIEWPORTS = [(1440, 900), (1280, 800), (1024, 768), (768, 1024), (390, 844)]
FULL_PAGE = {(1440, 900), (390, 844)}


def _chrome() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    sys.exit("Chrome not found; set one of CHROME_CANDIDATES")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Tab:
    """A minimal DevTools-protocol client for one page target."""

    def __init__(self, ws_url: str):
        self.ws = websocket.create_connection(ws_url, timeout=60, suppress_origin=True)
        self.next_id = 0
        self.events: list[dict] = []

    def call(self, method: str, **params):
        self.next_id += 1
        mid = self.next_id
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
            self.events.append(msg)

    def pump(self, seconds: float) -> None:
        """Collect events for a while (console, network) without blocking forever."""
        end = time.time() + seconds
        self.ws.settimeout(0.25)
        try:
            while time.time() < end:
                try:
                    self.events.append(json.loads(self.ws.recv()))
                except websocket.WebSocketTimeoutException:
                    pass
        finally:
            self.ws.settimeout(60)

    def evaluate(self, expr: str):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")


def capture(tab: Tab, base: str, route: str, w: int, h: int, settle: float,
            out: Path, name: str) -> dict:
    tab.events.clear()
    mobile = w < 768
    tab.call("Emulation.setDeviceMetricsOverride", width=w, height=h,
             deviceScaleFactor=1, mobile=mobile)
    tab.call("Emulation.setTouchEmulationEnabled", enabled=mobile)
    t0 = time.time()
    tab.call("Page.navigate", url=base + route)
    # wait for load, then let data requests and first renders settle
    deadline = time.time() + 30
    while time.time() < deadline:
        if tab.evaluate("document.readyState") == "complete":
            break
        time.sleep(0.1)
    load_s = time.time() - t0
    tab.pump(settle)

    info = tab.evaluate("""(() => {
      const c = document.createElement('canvas');
      return {
        title: document.title,
        docHeight: document.documentElement.scrollHeight,
        overflowX: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
        scrollWidth: document.documentElement.scrollWidth,
        webgl2: !!c.getContext('webgl2'),
        reducedMotion: matchMedia('(prefers-reduced-motion: reduce)').matches,
      };
    })()""") or {}

    shot = tab.call("Page.captureScreenshot", format="png")
    vp = out / f"{name}@{w}x{h}.png"
    vp.write_bytes(base64.b64decode(shot["data"]))
    files = [vp.name]
    if (w, h) in FULL_PAGE and info.get("docHeight", h) > h + 4:
        full = tab.call("Page.captureScreenshot", format="png", captureBeyondViewport=True,
                        clip={"x": 0, "y": 0, "width": w,
                              "height": min(info["docHeight"], 12000), "scale": 1})
        fp = out / f"{name}@{w}x{h}-full.png"
        fp.write_bytes(base64.b64decode(full["data"]))
        files.append(fp.name)

    console, exceptions, failed = [], [], []
    statuses: dict[str, int] = {}
    for ev in tab.events:
        m, p = ev.get("method"), ev.get("params", {})
        if m == "Runtime.consoleAPICalled" and p.get("type") in ("error", "warning", "assert"):
            console.append({"type": p["type"], "text": " ".join(
                str(a.get("value", a.get("description", ""))) for a in p.get("args", []))[:300]})
        elif m == "Runtime.exceptionThrown":
            d = p.get("exceptionDetails", {})
            exceptions.append((d.get("exception", {}).get("description") or d.get("text", ""))[:300])
        elif m == "Log.entryAdded" and p.get("entry", {}).get("level") == "error":
            console.append({"type": "log-error", "text": p["entry"].get("text", "")[:300]})
        elif m == "Network.responseReceived":
            statuses[p["requestId"]] = p["response"]["status"]
            if p["response"]["status"] >= 400:
                failed.append({"url": p["response"]["url"], "status": p["response"]["status"]})
        elif m == "Network.loadingFailed" and not p.get("canceled"):
            failed.append({"requestId": p.get("requestId"), "error": p.get("errorText")})
    return {"route": route, "viewport": f"{w}x{h}", "load_s": round(load_s, 2),
            "requests": len(statuses), "files": files, "console": console,
            "exceptions": exceptions, "failed": failed, **info}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True, help="output folder name, e.g. baseline / after")
    ap.add_argument("--base", default="http://localhost:8912")
    ap.add_argument("--settle", type=float, default=4.0, help="seconds to wait after load")
    ap.add_argument("--routes", nargs="*", default=list(ROUTES))
    ap.add_argument("--viewports", nargs="*", default=[f"{w}x{h}" for w, h in VIEWPORTS])
    args = ap.parse_args()

    try:
        urllib.request.urlopen(args.base + "/api/health", timeout=5)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"server not reachable at {args.base}: {exc}")

    out = ROOT / "artifacts" / "frontend" / args.label
    out.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    profile = tempfile.mkdtemp(prefix="fbd-chrome-")
    proc = subprocess.Popen([
        _chrome(), "--headless=new", f"--remote-debugging-port={port}",
        f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
        "--hide-scrollbars", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
        "--ignore-gpu-blocklist", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    report = []
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
        for domain in ("Page", "Runtime", "Network", "Log"):
            tab.call(f"{domain}.enable")
        viewports = [tuple(int(v) for v in s.split("x")) for s in args.viewports]
        for name in args.routes:
            for w, h in viewports:
                r = capture(tab, args.base, ROUTES[name], w, h, args.settle, out, name)
                flag = "" if not (r["console"] or r["exceptions"] or r["failed"]) else "  <-- issues"
                print(f"{name:8s} {w}x{h}: load {r['load_s']:.2f}s, {r['requests']} req, "
                      f"h={r.get('docHeight')}, overflowX={r.get('overflowX')}, "
                      f"webgl2={r.get('webgl2')}{flag}", flush=True)
                report.append(r)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {len(report)} captures + report.json to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
