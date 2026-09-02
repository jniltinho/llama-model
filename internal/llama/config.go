// Package llama reads the Ollama model store and drives the llama-server
// systemd service and the client configs that point at it.
package llama

import (
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const (
	// ConfPath holds every llama-server setting: model, alias, GPU, port, flags.
	ConfPath = "/etc/default/llama-server"
	// Service is the systemd unit name.
	Service = "llama-server"
	// HermesMin is the smallest context Hermes accepts for a fallback provider.
	HermesMin = 64000
)

// OllamaDir is where Ollama keeps its manifests and blobs.
var OllamaDir = envOr("OLLAMA_MODELS", "/usr/share/ollama/.ollama/models")

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

var (
	reModel  = regexp.MustCompile(`(?m)^LLAMA_MODEL=.*$`)
	reAlias  = regexp.MustCompile(`(?m)^LLAMA_ALIAS=.*$`)
	reArgs   = regexp.MustCompile(`(?m)^LLAMA_ARGS=.*$`)
	reMmproj = regexp.MustCompile(`\s*--mmproj \S+`)
	reCtx    = regexp.MustCompile(`--ctx-size (\d+)`)
)

func ConfRead() (string, error) {
	b, err := os.ReadFile(ConfPath)
	if err != nil {
		return "", fmt.Errorf("%s not found — is the llama-cpp-cuda package installed?", ConfPath)
	}
	return string(b), nil
}

// ConfGet returns the value of a KEY=value line.
func ConfGet(txt, key string) string {
	m := regexp.MustCompile(`(?m)^` + regexp.QuoteMeta(key) + `=(.*)$`).FindStringSubmatch(txt)
	if m == nil {
		return ""
	}
	return strings.TrimSpace(m[1])
}

// ConfCtx reads the --ctx-size currently set in LLAMA_ARGS.
func ConfCtx(txt string) int {
	m := reCtx.FindStringSubmatch(txt)
	if m == nil {
		return 0
	}
	n, _ := strconv.Atoi(m[1])
	return n
}

// ConfApply rewrites the config for a given model and context size.
func ConfApply(txt string, m Model, ctx int) string {
	out := reModel.ReplaceAllLiteralString(txt, "LLAMA_MODEL="+m.Blob)
	out = reAlias.ReplaceAllLiteralString(out, "LLAMA_ALIAS="+m.Name)

	// LLAMA_ARGS may be written quoted; keep whatever style is already there
	line := ConfGet(out, "LLAMA_ARGS")
	quote := ""
	if len(line) >= 2 && (line[0] == '"' || line[0] == '\'') && line[len(line)-1] == line[0] {
		quote = string(line[0])
		line = line[1 : len(line)-1]
	}

	line = reMmproj.ReplaceAllLiteralString(line, "")
	if m.Proj != "" {
		if _, err := os.Stat(m.Proj); err == nil {
			line = "--mmproj " + m.Proj + " " + line
		}
	}
	size := "--ctx-size " + strconv.Itoa(ctx)
	if reCtx.MatchString(line) {
		line = reCtx.ReplaceAllLiteralString(line, size)
	} else {
		line += " " + size // no --ctx-size yet: adding it is the whole point
	}

	return reArgs.ReplaceAllLiteralString(out, "LLAMA_ARGS="+quote+strings.TrimSpace(line)+quote)
}

// Systemctl runs a systemctl subcommand with its output attached to ours.
func Systemctl(args ...string) error {
	cmd := exec.Command("systemctl", args...)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	return cmd.Run()
}

// Journal prints the last few log lines of the service.
func Journal(lines int) {
	cmd := exec.Command("journalctl", "-u", Service, "-n", strconv.Itoa(lines), "--no-pager")
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	cmd.Run()
}

// serviceStarting reports whether the unit is still worth waiting on:
// "active" or "activating". Anything else (failed, inactive) means give up.
func serviceStarting() bool {
	out, err := exec.Command("systemctl", "show", "-p", "ActiveState", "--value", Service).Output()
	if err != nil {
		return false
	}
	switch strings.TrimSpace(string(out)) {
	case "active", "activating", "reloading":
		return true
	}
	return false
}

// WaitHealth polls /health until it reports ready. Returns false as soon as the
// service dies, instead of sitting out the whole timeout.
//
// llama-server answers 503 while the weights are still loading, so only 200
// counts as up — anything else means keep waiting.
func WaitHealth(port string, timeout time.Duration) bool {
	client := &http.Client{Timeout: 2 * time.Second}
	url := "http://127.0.0.1:" + port + "/health"
	for deadline := time.Now().Add(timeout); time.Now().Before(deadline); {
		resp, err := client.Get(url)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return true
			}
		}
		if !serviceStarting() {
			return false
		}
		time.Sleep(2 * time.Second)
	}
	return false
}
