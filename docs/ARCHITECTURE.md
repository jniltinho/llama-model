# Architecture

One static binary, three layers. `cmd/` is cobra plumbing only; all logic lives
in `internal/`.

```
                 ┌─────────────────────────────────────────────┐
                 │                  cmd/ (cobra)               │
                 │  list · pull · set · rm · gpu · proxy       │
                 └──────────────┬───────────────┬──────────────┘
                                │               │
                 ┌──────────────▼──────┐  ┌─────▼──────────────┐
                 │   internal/llama    │  │  internal/nvidia   │
                 │                     │  │  nvidia-smi wrapper│
                 │ ollama.go  store    │  │  (no cgo/NVML)     │
                 │ registry.go pull    │  └────────────────────┘
                 │ gguf.go    metadata │
                 │ config.go  set/svc  │
                 │ clients.go patching │
                 │ remove.go  rm       │
                 │ proxy.go   proxy    │
                 └──────┬───────┬──────┘
                        │       │
        ┌───────────────▼──┐  ┌─▼────────────────────────────┐
        │ Ollama store     │  │ /etc/default/llama-server    │
        │ manifests+blobs  │  │ systemd · OpenCode · Hermes  │
        └──────────────────┘  └──────────────────────────────┘
```

## internal/llama

| File | Responsibility |
|------|----------------|
| `ollama.go` | Reads the Ollama store (manifests + sha256-named blobs) — the same layout ollama itself writes |
| `registry.go` | Pulls models straight from the Ollama registry over HTTPS: manifest resolution, resumable blob downloads, sha256 verification. No daemon needed |
| `gguf.go` | Parses GGUF headers for model metadata (quantization, architecture, vision projector) |
| `config.go` | The `set` flow: edits `/etc/default/llama-server`, restarts the systemd unit, polls `/health`, rolls back the config if the model fails to come up |
| `clients.go` | Surgical text edits to OpenCode (`opencode.jsonc`) and Hermes configs — text-level, not parse+dump, so comments and hand-written structure survive. Writes a `.bak` next to every edited file. Wires vision (`attachment` + `modalities`) when the model has a projector |
| `remove.go` | Deletes a model's manifest and only the blobs no other manifest references |
| `proxy.go` | Anthropic-protocol proxy: hoists mid-conversation `system` messages into the top-level `system` field (Qwen templates reject them inline), streams everything else untouched |

## internal/nvidia

Wraps `nvidia-smi` by shelling out — deliberately. NVML would need cgo, and the
binary is built static (`CGO_ENABLED=0`) so it keeps working across driver
upgrades. Parses the query CSV output into typed readings: temperature
(core + memory), power, clocks, utilization, throttle-reason bitmask, Xid
kernel messages.

## Design decisions

- **No Ollama daemon dependency.** Both `pull` and `rm` operate on the store
  directly; the registry client speaks the same protocol ollama does.
- **Rollback over hope.** `set` keeps the previous config and restores it if
  `/health` never comes up — a bad model never leaves the service dead.
- **Text-surgical config patching.** Client configs are edited with targeted
  string operations so user comments and formatting survive; every touched
  file gets a `.bak`.
- **Static binary.** `CGO_ENABLED=0`, `-trimpath`, stripped, UPX-compressed
  when available. One file to copy to the GPU host.
