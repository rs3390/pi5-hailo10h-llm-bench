# Setting up the HAT for `bench.py`

Every step I took on a Raspberry Pi 5 (8 GB) with the AI HAT+ 2, in order, with the problems I hit. 22 November 2026.

## 1. The Hailo runtime

```
sudo apt install dkms hailo-h10-all
sudo reboot
```

This gives you the driver, the firmware and the runtime, version 5.1.1. Check the chip answers:

```
hailortcli fw-control identify
```

It should report `HAILO10H`. If it does not, see "The HAT is not found" below.

## 2. Model Zoo GenAI 5.1.1 (hailo-ollama and its models)

The runtime cannot run a language model by itself. Hailo's Model Zoo GenAI package adds `hailo-ollama`, a server that speaks a subset of Ollama's API on port 8000. Install the version that matches the runtime:

```
wget https://dev-public.hailo.ai/2025_12/Hailo10/hailo_gen_ai_model_zoo_5.1.1_arm64.deb
sudo dpkg -i hailo_gen_ai_model_zoo_5.1.1_arm64.deb
```

**If it fails with `chmod: cannot access '/usr/share/hailo-ollama/models'`**, the folder is missing. Create it and finish the install:

```
sudo mkdir -p /usr/share/hailo-ollama/models
sudo dpkg --configure -a
```

**Stay on 5.1.1.** Newer Model Zoo packages (5.2, 5.3) have been reported on Hailo's forum to break against the 5.1.1 runtime, and in one report installing them removed the runtime. For the same reason, do not run `sudo apt full-upgrade` while this works, unless you are ready to repair the Hailo packages afterwards.

## 3. Start hailo-ollama

A server keeps running and waits for requests, so it holds its terminal. Open a terminal (or an SSH session) just for it:

```
hailo-ollama
```

It prints a few lines and then appears to hang. That means it is waiting. Stop it later with Ctrl+C. To run it in the background instead:

```
nohup hailo-ollama > ~/hailo-ollama.log 2>&1 &
```

## 4. Download a model onto the HAT

In a second terminal:

```
curl -sS http://localhost:8000/hailo/v1/list; echo
```

This lists the models you **can** download. On 5.1.1 it was:
`deepseek_r1_distill_qwen:1.5b`, `llama3.2:3b`, `qwen2.5-coder:1.5b`, `qwen2.5-instruct:1.5b`, `qwen2:1.5b`.

Download one:

```
curl -sS http://localhost:8000/api/pull -H 'Content-Type: application/json' -d '{"model":"llama3.2:3b","stream":true}'
```

Then check what you **have**:

```
curl -sS http://localhost:8000/api/tags; echo
```

`llama3.2:3b` is about 3.4 GB in Hailo's `hef` format.

## 5. Ollama for the CPU side

Install Ollama from ollama.com, then pull the same model:

```
ollama pull llama3.2:3b
```

## 6. Check, then run

```
python3 bench.py check --probe llama3.2:3b
python3 bench.py speed --cpu-model llama3.2:3b
```

`check` walks the chain one step at a time: the card on the PCIe bus, the driver's device file, the firmware, hailo-ollama, and with `--probe` one prompt answered on the chip while it watches the Pi's processor.

## Problems I hit

**Nothing comes back from `curl -s localhost:8000`.** `-s` hides errors. Use `curl -sS` to see them. "Connection refused" means hailo-ollama is not running, or not installed: `which hailo-ollama`, and `dpkg -l | grep -i hailo` to see what is.

**HTTP 500 from hailo-ollama, with `Value is NOT a Primitive type`.** The request carried `"keep_alive": "10m"`. Ollama accepts that; hailo-ollama does not. Leave `keep_alive` out. A plain request works:

```
curl -sS http://localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"model":"llama3.2:3b","stream":false,"messages":[{"role":"user","content":"Say hello."}]}'; echo
```

`"options": {"num_predict": 200}` is accepted.

**Is something else holding the HAT?** Only one program can open the chip at a time, and that should be hailo-ollama. `pgrep -a hailo-ollama` should print exactly one line. If it prints two, `pkill hailo-ollama` and start one again.

**The HAT is not found.** Power off, reseat the ribbon cable at both ends, and check that `dtparam=pciex1` is not disabled in `/boot/firmware/config.txt`.

**`bench.py` shows processor load as n/a.** It reads `/proc/stat`, which only Linux has. Run it on the Pi, not on a Mac.
