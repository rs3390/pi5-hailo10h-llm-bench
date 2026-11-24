"""Is the Raspberry Pi AI HAT+ 2 there, is it doing the work, and how fast is it?

    python3 bench.py check
    python3 bench.py check --probe llama3.2:3b
    python3 bench.py speed
    python3 bench.py speed --cpu-model llama3.2:3b     # the same model on both

`check` looks for the HAT step by step: on the PCIe bus, the driver's device
file, the firmware, then Hailo's model server (hailo-ollama, port 8000). With
--probe it sends one prompt to the HAT and watches, while the model writes,
which processes hold the HAT's device file open and how busy the Pi's own
processor is. A reply with the device held and the processor mostly idle is
the evidence the HAT did the work.

`speed` sends the same prompts to a model on the Pi's CPU (Ollama, port
11434) and to a model on the HAT (hailo-ollama, port 8000), and reports the
time to the first token, writing speed in tokens and in words a second, and
processor load. A table goes to results/.

Python 3.10 or later, standard library only. Processor load needs Linux.
MIT licence; see LICENSE.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

HAILO_VENDOR = "0x1e60"  # Hailo's PCI vendor ID
DEVICE = "/dev/hailo0"
CPU_HOST = "http://127.0.0.1:11434"
HAT_HOST = "http://127.0.0.1:8000"
HERE = Path(__file__).resolve().parent

# The largest model Hailo's Model Zoo GenAI 5.1.1 serves on the Hailo-10H.
# Its /hailo/v1/list on a Pi 5, 2026-11-22: deepseek_r1_distill_qwen:1.5b,
# llama3.2:3b, qwen2.5-coder:1.5b, qwen2.5-instruct:1.5b, qwen2:1.5b.
DEFAULT_HAT_MODEL = "llama3.2:3b"
DEFAULT_CPU_MODEL = "qwen3:4b-instruct"


# --- talking to an Ollama-style server ------------------------------------------


class ModelError(RuntimeError):
    """The server could not be reached or refused the request."""


@dataclass
class Reply:
    text: str
    stats: dict = field(default_factory=dict)


def chat(host: str, model: str, messages: list, *, options: Optional[dict] = None,
         keep_alive=None, timeout: float = 600, on_token: Optional[Callable[[str], None]] = None) -> Reply:
    """POST /api/chat with streaming on; call on_token for each piece as it arrives.

    keep_alive is sent only when given: hailo-ollama answers HTTP 500 to
    keep_alive "10m", which Ollama itself accepts.
    """
    body: dict = {"model": model, "messages": messages, "stream": True}
    if options:
        body["options"] = options
    if keep_alive is not None:
        body["keep_alive"] = keep_alive
    request = urllib.request.Request(
        host.rstrip("/") + "/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    text, stats = [], {}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise ModelError(str(chunk["error"]))
                piece = (chunk.get("message") or {}).get("content") or ""
                if piece:
                    text.append(piece)
                    if on_token:
                        on_token(piece)
                if chunk.get("done"):
                    stats = {
                        "wrote_tokens": chunk.get("eval_count", 0) or 0,
                        "wrote_s": (chunk.get("eval_duration", 0) or 0) / 1e9,
                        "read_tokens": chunk.get("prompt_eval_count", 0) or 0,
                    }
    except urllib.error.HTTPError as exc:
        raise ModelError(f"the model server answered with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise ModelError(f"cannot reach the model at {host}: {exc.reason}") from exc
    except (TimeoutError, json.JSONDecodeError) as exc:
        raise ModelError(str(exc)) from exc
    return Reply(text="".join(text), stats=stats)


# --- measuring the Pi itself ----------------------------------------------------


def read_cpu_times(stat_path: str = "/proc/stat") -> Optional[tuple]:
    """(busy, total) jiffies across all cores, or None where there is no /proc (macOS)."""
    try:
        with open(stat_path) as handle:
            parts = handle.readline().split()[1:]
    except OSError:
        return None
    values = [int(v) for v in parts]
    idle = values[3] + (values[4] if len(values) > 4 else 0)  # idle + iowait
    total = sum(values[:8])
    return total - idle, total


def busy_percent(before: tuple, after: tuple) -> float:
    """Share of all cores busy between two readings: 100 means every core flat out."""
    busy = after[0] - before[0]
    total = after[1] - before[1]
    return round(100.0 * busy / total, 1) if total > 0 else 0.0


def holders(device: str = DEVICE, proc: str = "/proc") -> list:
    """Names of the processes we can see holding the device file open."""
    names = set()
    try:
        pids = [p for p in os.listdir(proc) if p.isdigit()]
    except OSError:
        return []
    for pid in pids:
        fd_dir = os.path.join(proc, pid, "fd")
        try:
            for fd in os.listdir(fd_dir):
                try:
                    if os.readlink(os.path.join(fd_dir, fd)) == device:
                        with open(os.path.join(proc, pid, "comm")) as handle:
                            names.add(handle.read().strip())
                        break
                except OSError:
                    continue
        except OSError:
            continue  # another user's process, or gone
    return sorted(names)


class Watcher:
    """Samples processor load and HAT holders in the background during a request."""

    def __init__(self, interval: float = 0.5, device: str = DEVICE, stat_path: str = "/proc/stat"):
        self.interval, self.device, self.stat_path = interval, device, stat_path
        self.loads: list = []
        self.seen: set = set()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        previous = read_cpu_times(self.stat_path)
        if previous is None:
            return  # no /proc: load is reported as n/a
        while not self._stop.wait(self.interval):
            current = read_cpu_times(self.stat_path)
            if current is None:
                return
            self.loads.append(busy_percent(previous, current))
            previous = current
            self.seen.update(holders(self.device))

    def __enter__(self):
        self.seen.update(holders(self.device))
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    @property
    def load(self) -> Optional[float]:
        return round(statistics.mean(self.loads), 1) if self.loads else None


# --- one timed request -------------------------------------------------------------


@dataclass
class Timing:
    target: str
    model: str
    prompt: str
    run: int = 1
    ttft: float = 0.0          # seconds to the first token
    rate: float = 0.0          # tokens a second while writing
    words_rate: float = 0.0    # words a second while writing: fairer across models
    total: float = 0.0
    tokens: int = 0
    words: int = 0
    load: Optional[float] = None  # % of all cores, averaged; None without /proc
    hat_held_by: list = field(default_factory=list)
    error: str = ""
    text: str = ""


def writing_rate(tokens: int, first: Optional[float], last: Optional[float], stats: dict) -> float:
    """Tokens a second while writing. Use the server's own count when it gives one."""
    if stats.get("wrote_tokens") and stats.get("wrote_s"):
        return round(stats["wrote_tokens"] / stats["wrote_s"], 2)
    if tokens > 1 and first is not None and last is not None and last > first:
        return round((tokens - 1) / (last - first), 2)
    return 0.0


def timed_chat(send: Callable, target: str, host: str, model: str, name: str, messages: list, *,
               options: dict, keep_alive=None, run: int = 1, timeout: float = 600,
               device: str = DEVICE, stat_path: str = "/proc/stat") -> Timing:
    stamps = []

    def on_token(piece):
        stamps.append(time.monotonic())

    result = Timing(target=target, model=model, prompt=name, run=run)
    started = time.monotonic()
    with Watcher(device=device, stat_path=stat_path) as watch:
        try:
            reply = send(host, model, messages, options=options, keep_alive=keep_alive,
                         timeout=timeout, on_token=on_token)
        except Exception as exc:  # recorded, never fatal
            result.error = str(exc)[:120]
            reply = None
    result.total = round(time.monotonic() - started, 1)
    result.load = watch.load
    result.hat_held_by = sorted(watch.seen)
    if reply is None:
        return result
    stats = reply.stats or {}
    result.text = reply.text
    result.tokens = stats.get("wrote_tokens") or len(stamps)
    result.words = len(reply.text.split())
    result.ttft = round(stamps[0] - started, 2) if stamps else 0.0
    result.rate = writing_rate(len(stamps), stamps[0] if stamps else None, stamps[-1] if stamps else None, stats)
    writing = (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
    result.words_rate = round(result.words / writing, 2) if writing > 0 and result.words > 1 else 0.0
    return result


# --- check ---------------------------------------------------------------------------


def _run(cmd: list, timeout: int = 20) -> tuple:
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return done.returncode, (done.stdout + done.stderr).strip()
    except FileNotFoundError:
        return 127, f"{cmd[0]} not installed"
    except subprocess.TimeoutExpired:
        return 124, f"{cmd[0]} timed out"


def find_hailo_pci(sysfs: str = "/sys/bus/pci/devices") -> list:
    found = []
    try:
        for dev in os.listdir(sysfs):
            try:
                with open(os.path.join(sysfs, dev, "vendor")) as handle:
                    if handle.read().strip().lower() == HAILO_VENDOR:
                        found.append(dev)
            except OSError:
                continue
    except OSError:
        pass
    return found


def list_hat_models(host: str = HAT_HOST) -> list:
    with urllib.request.urlopen(host.rstrip("/") + "/hailo/v1/list", timeout=5) as response:
        data = json.loads(response.read().decode("utf-8"))
    models = data.get("models", data) if isinstance(data, dict) else data
    return [m.get("name", m.get("model", str(m))) if isinstance(m, dict) else str(m) for m in models or []]


def check(probe: str = "", host: str = HAT_HOST, send: Callable = chat,
          sysfs: str = "/sys/bus/pci/devices", device: str = DEVICE) -> int:
    ok = True

    def line(mark, what, detail=""):
        print(f"[{mark}] {what}" + (f": {detail}" if detail else ""))

    pci = find_hailo_pci(sysfs)
    if pci:
        line("PASS", "HAT on the PCIe bus", ", ".join(pci))
    else:
        ok = False
        line("FAIL", "HAT on the PCIe bus", "no Hailo device. Reseat the ribbon cable; check `dtparam=pciex1` is not disabled")

    if os.path.exists(device):
        line("PASS", "driver loaded", device)
    else:
        ok = False
        line("FAIL", "driver loaded", f"{device} missing. Run `sudo apt install dkms hailo-h10-all`, then reboot")

    code, out = _run(["hailortcli", "fw-control", "identify"])
    if code == 0:
        keep = [l.strip() for l in out.splitlines()
                if any(k in l for k in ("Firmware Version", "Device Architecture", "Board Name"))]
        line("PASS", "firmware answers", "; ".join(keep) or "identify ran")
    else:
        ok = False
        line("FAIL", "firmware answers", out.splitlines()[-1] if out else f"exit {code}")

    try:
        models = list_hat_models(host)
        line("PASS", "hailo-ollama running", f"{len(models)} models it can serve: {', '.join(models[:12])}")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        ok = False
        line("FAIL", "hailo-ollama running",
             f"nothing on {host} ({exc}). Install Hailo's Model Zoo GenAI package and start `hailo-ollama`")

    if probe:
        print(f"\nProbing {probe} on the HAT...")
        t = timed_chat(send, "hat", host, probe, "probe",
                       [{"role": "user", "content": "Reply with one short sentence: what is the capital of France?"}],
                       options={"temperature": 0}, device=device)
        if t.error:
            ok = False
            line("FAIL", "HAT answered", t.error)
        else:
            line("PASS", "HAT answered", repr(t.text[:80]))
            held = [n for n in t.hat_held_by if n]
            if held:
                line("PASS", "HAT in use while writing", f"{device} held by {', '.join(held)}")
            else:
                line("INFO", "HAT in use while writing",
                     f"no visible process held {device} (a server run by another user is hidden; run this as that user)")
            load = "n/a (needs Linux)" if t.load is None else f"{t.load}% of all cores"
            line("INFO", "Pi processor load while writing",
                 f"{load} (a CPU model runs near 100%; the HAT should leave the cores idle)")
            line("INFO", "speed", f"first token {t.ttft} s, {t.rate} tokens/s, {t.tokens} tokens")
    print("\nAll checks passed." if ok else "\nSome checks failed; fix them top to bottom.")
    return 0 if ok else 1


# --- speed -----------------------------------------------------------------------------

# A long prompt: about 400 words to read before the first token. Invented text.
MEETING_NOTES = """Notes from the March meeting of the Elm Street rooftop solar cooperative.

Present: eleven members, the treasurer, and two visitors from the Birch Avenue group, who are thinking of
starting a cooperative of their own and came to see how ours runs.

1. Output. The array produced 18,400 kWh in the twelve months to February, about 6% below the installer's
estimate. The treasurer put most of the gap down to the six weeks in December when snow covered the north
row of panels, and to one inverter that failed in July and took nineteen days to replace under warranty.
Members asked whether the estimate had been too optimistic to begin with. The installer's sheet assumed no
shading from the new building on Oak Street, which went up after the estimate was made.

2. Money. Credits from the utility came to 2,310 dollars for the year. After insurance, the maintenance
contract and the reserve, 1,640 dollars was paid out to members in proportion to their shares. The reserve
now holds 3,900 dollars against a target of 5,000. The treasurer proposed keeping payouts flat next year so
the reserve reaches its target by December. Two members objected: they joined for the payout and would
rather the reserve grew more slowly. The vote was eight in favour of the treasurer's proposal, three
against.

3. The Oak Street building. Its owners have offered to let the cooperative put panels on their roof, which
is larger than ours and unshaded. They would take a share of the output instead of rent. The committee
will ask the installer for an estimate and bring a proposal to the June meeting. Several members asked that
the new estimate account for snow and shading this time, and that it be checked against our own figures
from this year rather than the installer's model alone.

4. Visitors. The Birch Avenue visitors asked what they should do first. Members suggested three things: get
an independent estimate, agree how payouts and the reserve will work before anyone buys a share, and
choose a maintenance contract with a fixed response time for failed inverters. The visitors will come back
in May.

5. Next meeting: the second Tuesday of June, in the library's back room. The treasurer will circulate the
accounts two weeks before.
"""


def prompts() -> list:
    """Three jobs: a one-liner, a paragraph, and a long prompt that takes a while to read."""
    return [
        ("one line", [{"role": "user", "content": "Reply with one short sentence: what is the capital of France?"}]),
        ("paragraph", [{"role": "user", "content": "In about 80 words, explain what a Raspberry Pi is."}]),
        ("long prompt", [{"role": "user", "content":
                          "Summarise these meeting notes in three bullet points of under 20 words each.\n\n"
                          + MEETING_NOTES}]),
    ]


def _cell(value, unit: str = "") -> str:
    return "n/a" if value is None else f"{value}{unit}"


def speed(cpu_model: str, hat_model: str, runs: int, only: str, out_dir: Path, send: Callable = chat,
          cpu_host: str = CPU_HOST, hat_host: str = HAT_HOST, device: str = DEVICE,
          stat_path: str = "/proc/stat") -> Path:
    if read_cpu_times(stat_path) is None:
        print("note: no /proc/stat here, so processor load will read n/a. Run this on the Pi.", flush=True)
    targets = []
    if only in ("both", "cpu"):
        targets.append(("cpu", cpu_host, cpu_model))
    if only in ("both", "hat"):
        targets.append(("hat", hat_host, hat_model))
    options = {"temperature": 0, "num_predict": 200}
    results = []
    for target, host, model in targets:
        alive = "10m" if target == "cpu" else None  # hailo-ollama answers 500 to keep_alive "10m"
        print(f"{target}: {model} — warming up", flush=True)
        timed_chat(send, target, host, model, "warm-up", prompts()[0][1], options=options,
                   keep_alive=alive, device=device, stat_path=stat_path)
        for name, messages in prompts():
            for run in range(1, runs + 1):
                t = timed_chat(send, target, host, model, name, messages, options=options, keep_alive=alive,
                               run=run, device=device, stat_path=stat_path)
                results.append(t)
                print(f"  {name} #{run}: first token {t.ttft} s, {t.rate} tok/s, {t.words_rate} words/s, "
                      f"{t.tokens} tokens, load {_cell(t.load, '%')}{' ERROR ' + t.error if t.error else ''}",
                      flush=True)

    lines = [
        "# CPU against HAT",
        "",
        f"Run {datetime.now():%Y-%m-%d %H:%M}. Medians of {runs} runs after one warm-up, except "
        "\"first token, run 1\": Ollama on the CPU reuses a prompt it has just read, so later runs "
        "flatter it. Load is the Pi's own processor, as a share of all cores.",
        "",
        "| Where | Model | Prompt | First token, run 1 (s) | First token (s) | Tokens/s | Words/s | Tokens | "
        "Total (s) | Pi load | HAT held by |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for target, _, model in targets:
        for name, _ in prompts():
            group = [r for r in results if r.target == target and r.prompt == name and not r.error]
            errors = [r for r in results if r.target == target and r.prompt == name and r.error]
            if not group:
                why = errors[0].error if errors else "no runs"
                lines.append(f"| {target} | {model} | {name} | error: {why} | | | | | | | |")
                continue
            med = lambda attr: round(statistics.median(getattr(r, attr) for r in group), 2)
            first = next((r.ttft for r in group if r.run == 1), group[0].ttft)
            loads = [r.load for r in group if r.load is not None]
            load = f"{round(statistics.median(loads), 1)}%" if loads else "n/a"
            held = sorted({n for r in group for n in r.hat_held_by}) or ["none seen"]
            lines.append(f"| {target} | {model} | {name} | {first} | {med('ttft')} | {med('rate')} | "
                         f"{med('words_rate')} | {int(med('tokens'))} | {med('total')} | {load} | {', '.join(held)} |")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"speed-{datetime.now():%Y%m%d-%H%M}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines[4:]))
    print(f"\nwritten: {path}")
    return path


def main(argv: Optional[list] = None, send: Callable = chat) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="is the HAT there and working?")
    c.add_argument("--probe", default="", help="a model to send one prompt to on the HAT")
    s = sub.add_parser("speed", help="a CPU model against a HAT model, same prompts")
    s.add_argument("--cpu-model", default=DEFAULT_CPU_MODEL)
    s.add_argument("--hat-model", default=DEFAULT_HAT_MODEL)
    s.add_argument("--runs", type=int, default=3)
    s.add_argument("--only", choices=["both", "cpu", "hat"], default="both")
    s.add_argument("--out", default=str(HERE / "results"))
    args = parser.parse_args(argv)
    if args.command == "check":
        return check(args.probe, send=send)
    speed(args.cpu_model, args.hat_model, args.runs, args.only, Path(args.out), send=send)
    return 0


if __name__ == "__main__":
    sys.exit(main())
