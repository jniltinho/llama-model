# llama-model

[![CI](https://github.com/jniltinho/llama-model/actions/workflows/ci.yml/badge.svg)](https://github.com/jniltinho/llama-model/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/jniltinho/llama-model)](https://github.com/jniltinho/llama-model/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Manage the local **GGUF model store** and the **llama-server** systemd service — one static Go binary, no Ollama daemon required.

```
ollama.com/library → pull (registry, resumable) → local blob store
local blob store   → set  (edit config, restart llama-server, health-check, rollback)
Claude Code        → proxy (Anthropic protocol fix-up) → llama-server
```

| Command | Purpose |
|---------|---------|
| `list` | List the models Ollama has pulled (name, size, quantization) |
| `pull <model>` | Download from the Ollama registry — no daemon, sha256-verified, resumable |
| `set <model>` | Switch the llama-server model, restart, wait for `/health`, rollback on failure |
| `rm <model>` | Delete a model, keeping blobs shared with other models |
| `gpu list` / `gpu watch` | List and live-monitor the NVIDIA GPUs (temp, power, clocks, throttle reasons) |
| `proxy` | Anthropic-protocol front end for llama-server (for Claude Code) |

## Install

```bash
# from a release
curl -sL https://github.com/jniltinho/llama-model/releases/latest/download/llama-model_$(curl -s https://api.github.com/repos/jniltinho/llama-model/releases/latest | grep -oP '"tag_name": "v\K[^"]+')_linux_amd64.tar.gz | tar xz
sudo install -m0755 llama-model /usr/local/bin/

# or from source (Go 1.26+)
make build && sudo make install
```

## Quick start

```bash
llama-model list                    # what is on disk
sudo llama-model pull qwen3.8:27b   # download from ollama.com/library
sudo llama-model set qwen3.8:27b    # switch, restart and validate
sudo llama-model rm gemma4:e4b      # delete, keeping shared blobs
llama-model gpu watch --gpu v100    # live GPU telemetry
```

### `set` — what it actually does

`set` edits `/etc/default/llama-server` (`LLAMA_MODEL`, `LLAMA_ALIAS` and the
`--mmproj` inside `LLAMA_ARGS`), restarts the service and waits for `/health`.
If the model does not come up it **restores the previous config**. On success it
also renames the model in the **OpenCode** and **Hermes** configs, if they exist
— each edited file gets a `.bak` alongside it. Vision models get
`attachment: true` + `modalities` wired into OpenCode automatically.

### `pull` — no Ollama daemon needed

Talks straight to the Ollama registry over HTTPS. Blobs are verified against
their sha256 and an interrupted download resumes where it stopped, so running
`pull` again after a broken connection is cheap.

### `gpu watch` — telemetry that survives a crash

Tracks peak temperature and power, decodes throttle reasons, and if a card
falls off the bus it stops and prints the last good reading together with the
kernel's Xid messages — the state you need to tell a thermal problem from a
power one. Select a card by index, UUID or name fragment; `--csv` appends every
reading to a file.

### `proxy` — Claude Code against llama-server

Claude Code sends some instructions as a mid-conversation `system` message,
which Qwen's chat template rejects with a 500. The proxy lifts those into the
top-level `system` field and forwards everything else untouched, streaming
included:

```bash
llama-model proxy --listen 127.0.0.1:11436 --upstream http://127.0.0.1:11435
export ANTHROPIC_BASE_URL=http://127.0.0.1:11436
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the package layout and data flow.

```
main.go            → cmd/ (cobra commands, one file per command)
internal/llama/    → store, registry client, config editing, proxy, client patching
internal/nvidia/   → nvidia-smi parsing and GPU telemetry
```

## Development

```bash
make build   # static binary → dist/llama-model (UPX-compressed if available)
make test    # go test ./...
make lint    # gofmt + go vet
```

## License

[MIT](LICENSE) © Nilton Oliveira
