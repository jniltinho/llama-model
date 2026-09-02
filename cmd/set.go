package cmd

import (
	"fmt"
	"os"
	"strconv"
	"time"

	"llama-model/internal/llama"

	"github.com/spf13/cobra"
)

var ctxFlag string

var setCmd = &cobra.Command{
	Use:   "set <model>",
	Short: "Switch the llama-server model (needs sudo)",
	Long: `Switch the model llama-server runs.

Accepts a full name or a fragment of it (e.g. "qwen3.8"), as long as it is not
ambiguous. If the model does not fit in VRAM the context is halved and retried,
then quartered; if none works the previous config is restored.`,
	Example: `  sudo llama-model set qwen3.8:27b
  sudo llama-model set qwen3.8 --ctx max
  sudo llama-model set gemma4 --ctx 65536`,
	Args: cobra.ExactArgs(1),
	RunE: func(_ *cobra.Command, args []string) error { return runSet(args[0]) },
}

func init() {
	setCmd.Flags().StringVar(&ctxFlag, "ctx", "",
		"context size to use; 'max' = model ceiling (default: keep current)")
	rootCmd.AddCommand(setCmd)
}

func runSet(name string) error {
	if os.Geteuid() != 0 {
		return fmt.Errorf("run with sudo: sudo llama-model set %s", name)
	}

	m, err := llama.Find(llama.Manifests(), name)
	if err != nil {
		return err
	}
	if m.Blob == "" || !llama.Exists(m.Blob) {
		return fmt.Errorf("blob for %s not found", m.Name)
	}

	txt, err := llama.ConfRead()
	if err != nil {
		return err
	}
	oldAlias := llama.ConfGet(txt, "LLAMA_ALIAS")
	ctxMax := int(llama.ReadGGUF(m.Blob).Ctx)

	want := llama.ConfCtx(txt)
	switch {
	case ctxFlag == "max":
		if ctxMax > 0 {
			want = ctxMax
		}
	case ctxFlag != "":
		if want, err = strconv.Atoi(ctxFlag); err != nil {
			return fmt.Errorf("--ctx: %q is not a number or 'max'", ctxFlag)
		}
	}
	if ctxMax > 0 && want > ctxMax {
		fmt.Printf("  %d exceeds the model maximum (%d), using %d\n", want, ctxMax, ctxMax)
		want = ctxMax
	}
	if want <= 0 {
		return fmt.Errorf("could not determine a context size (no --ctx-size in %s)", llama.ConfPath)
	}

	port := llama.ConfGet(txt, "LLAMA_PORT")
	if port == "" {
		port = "11435"
	}
	if err := os.WriteFile(llama.ConfPath+".bak", []byte(txt), 0o644); err != nil {
		return err
	}

	for _, ctx := range ctxSteps(want) {
		vision := "no"
		if m.Proj != "" {
			vision = "yes"
		}
		fmt.Printf("==> %s  (%.1fG, ctx %d, vision %s)\n",
			m.Name, float64(m.Size)/(1<<30), ctx, vision)

		if err := os.WriteFile(llama.ConfPath, []byte(llama.ConfApply(txt, m, ctx)), 0o644); err != nil {
			return err
		}
		if err := llama.Systemctl("restart", llama.Service); err != nil {
			// the config on disk is already the new one: put the old one back
			restore(txt)
			return fmt.Errorf("systemctl restart: %w", err)
		}
		fmt.Println("  loading the model (up to ~1min from cold disk)...")

		if llama.WaitHealth(port, 4*time.Minute) {
			fmt.Printf("  OK: %s serving on http://127.0.0.1:%s\n", m.Name, port)
			if ctx < llama.HermesMin {
				fmt.Printf("  WARNING: ctx %d < %d — Hermes will refuse this fallback\n",
					ctx, llama.HermesMin)
			}
			if oldAlias == "" {
				oldAlias = m.Name
			}
			// also when the name did not change: --ctx may have moved the limit
			llama.PatchClients(oldAlias, m.Name, ctx, m.Proj != "")
			return nil
		}

		fmt.Printf("  did not start with ctx %d:\n", ctx)
		llama.Journal(5)
	}

	fmt.Println("==> no context size worked, restoring the previous config")
	restore(txt)
	return fmt.Errorf("%s did not come up", m.Name)
}

// restore puts the previous config back and restarts, so a failed switch never
// leaves the service pointing at a model it cannot load.
func restore(txt string) {
	if err := os.WriteFile(llama.ConfPath, []byte(txt), 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "could not restore %s: %v (backup at %s.bak)\n",
			llama.ConfPath, err, llama.ConfPath)
		return
	}
	llama.Systemctl("restart", llama.Service)
}

// ctxSteps returns the context sizes to try: the one asked for, then half and a
// quarter of it — the KV cache shrinks with the context, so a model that blows
// past the VRAM at full size often fits at half.
func ctxSteps(want int) []int {
	var steps []int
	for _, c := range []int{want, want / 2, want / 4} {
		if c >= 8192 && (len(steps) == 0 || steps[len(steps)-1] != c) {
			steps = append(steps, c)
		}
	}
	if len(steps) == 0 {
		steps = []int{want}
	}
	return steps
}
