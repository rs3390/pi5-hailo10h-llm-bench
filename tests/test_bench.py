"""Tests for bench.py. No Pi, HAT or model server needed: python3 -m unittest"""

import io
import os
import re
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bench  # noqa: E402


class LoadTest(unittest.TestCase):
    def test_busy_share_of_all_cores(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp, "a"), Path(tmp, "b")
            # user nice system idle iowait irq softirq steal
            a.write_text("cpu 100 0 100 800 0 0 0 0\n")
            b.write_text("cpu 400 0 100 900 0 0 0 0\n")  # +300 busy, +100 idle
            self.assertEqual(bench.busy_percent(bench.read_cpu_times(str(a)), bench.read_cpu_times(str(b))), 75.0)

    def test_no_proc_means_no_load_not_a_crash(self):
        # On a Mac there is no /proc/stat: the watcher thread must not die with a traceback.
        self.assertIsNone(bench.read_cpu_times("/no/such/stat"))
        with bench.Watcher(interval=0.01, stat_path="/no/such/stat") as watch:
            time.sleep(0.05)
        self.assertIsNone(watch.load)

    def test_rate_prefers_the_servers_own_count(self):
        self.assertEqual(bench.writing_rate(10, 0.0, 1.0, {"wrote_tokens": 45, "wrote_s": 10.0}), 4.5)

    def test_rate_counts_pieces_when_the_server_does_not_say(self):
        self.assertEqual(bench.writing_rate(11, 2.0, 4.0, {}), 5.0)
        self.assertEqual(bench.writing_rate(1, 2.0, 2.0, {}), 0.0)


class FindTheHatTest(unittest.TestCase):
    def test_pci_vendor_and_device_holders(self):
        with tempfile.TemporaryDirectory() as tmp:
            sysfs = Path(tmp, "pci")
            for dev, vendor in (("0000:01:00.0", "0x1e60"), ("0000:00:00.0", "0x14e4")):
                (sysfs / dev).mkdir(parents=True)
                (sysfs / dev / "vendor").write_text(vendor + "\n")
            self.assertEqual(bench.find_hailo_pci(str(sysfs)), ["0000:01:00.0"])

            device = Path(tmp, "hailo0")
            device.write_text("")
            proc = Path(tmp, "proc")
            (proc / "42" / "fd").mkdir(parents=True)
            (proc / "42" / "comm").write_text("hailo-ollama\n")
            os.symlink(device, proc / "42" / "fd" / "3")
            (proc / "43" / "fd").mkdir(parents=True)
            (proc / "43" / "comm").write_text("bash\n")
            self.assertEqual(bench.holders(str(device), str(proc)), ["hailo-ollama"])

    def test_check_fails_cleanly_with_no_hat(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with redirect_stdout(out):
                code = bench.check(host="http://127.0.0.1:9", sysfs=tmp, device=os.path.join(tmp, "none"))
            self.assertEqual(code, 1)
            text = out.getvalue()
            self.assertIn("[FAIL] HAT on the PCIe bus", text)
            self.assertIn("hailo-h10-all", text)
            self.assertIn("[FAIL] hailo-ollama running", text)


def fake_send(rate: float, server_counts: bool):
    def send(host, model, messages, *, options=None, keep_alive=None, timeout=0, on_token=None):
        send.calls.append((host, model, keep_alive))
        time.sleep(0.02)
        for _ in range(5):
            on_token("word ")
            time.sleep(1 / rate)
        stats = {"wrote_tokens": 5, "wrote_s": 5 / rate, "read_tokens": 12} if server_counts else {}
        return bench.Reply(text="one two three four five", stats=stats)
    send.calls = []
    return send


class SpeedTest(unittest.TestCase):
    def run_speed(self, send, only="both"):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            path = bench.speed("cpu-m", "hat-m", 1, only, Path(tmp), send=send,
                               device=os.path.join(tmp, "none"), stat_path=os.path.join(tmp, "nostat"))
            return path.read_text()

    def test_report_has_both_targets_every_prompt_and_both_rates(self):
        send = fake_send(rate=200, server_counts=False)
        text = self.run_speed(send)
        self.assertIn("| cpu | cpu-m | one line |", text)
        self.assertIn("| hat | hat-m | long prompt |", text)
        self.assertIn("Words/s", text)
        self.assertIn("First token, run 1", text)
        self.assertEqual({h for h, _, _ in send.calls}, {bench.CPU_HOST, bench.HAT_HOST})
        self.assertEqual(len(send.calls), 2 * (1 + 3))  # warm-up plus three prompts, each target

    def test_the_hat_is_never_sent_keep_alive(self):
        # hailo-ollama answered HTTP 500 to keep_alive "10m" on a Pi 5, 2026-11-22.
        send = fake_send(rate=200, server_counts=True)
        self.run_speed(send)
        self.assertEqual({k for h, _, k in send.calls if h == bench.HAT_HOST}, {None})
        self.assertEqual({k for h, _, k in send.calls if h == bench.CPU_HOST}, {"10m"})

    def test_a_failing_server_is_reported_not_fatal(self):
        def broken(*a, **k):
            raise bench.ModelError("cannot reach the model")
        self.assertIn("error: cannot reach the model", self.run_speed(broken, only="hat"))

    def test_the_long_prompt_is_long(self):
        name, messages = bench.prompts()[2]
        self.assertGreater(len(messages[0]["content"].split()), 380)

    def test_the_default_hat_model_is_one_model_zoo_5_1_1_serves(self):
        served = {"deepseek_r1_distill_qwen:1.5b", "llama3.2:3b", "qwen2.5-coder:1.5b",
                  "qwen2.5-instruct:1.5b", "qwen2:1.5b"}
        self.assertIn(bench.DEFAULT_HAT_MODEL, served)


class ChatTest(unittest.TestCase):
    def test_keep_alive_is_left_out_unless_given(self):
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        bodies = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"message":{"content":"hi"},"done":false}\n')
                self.wfile.write(b'{"done":true,"eval_count":1,"eval_duration":1000000}\n')

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        host = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            self.assertEqual(bench.chat(host, "m", [{"role": "user", "content": "x"}]).text, "hi")
            bench.chat(host, "m", [], keep_alive="10m")
        finally:
            server.shutdown()
            server.server_close()
        self.assertNotIn("keep_alive", bodies[0])
        self.assertEqual(bodies[1]["keep_alive"], "10m")


if __name__ == "__main__":
    unittest.main()
