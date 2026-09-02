package cmd

import (
	"fmt"
	"os"

	"llama-model/internal/llama"

	"github.com/spf13/cobra"
)

// version is stamped by the Makefile via -ldflags.
var version = "dev"

var rootCmd = &cobra.Command{
	Use:     "llama-model",
	Short:   "List Ollama models and switch the one llama-server runs",
	Version: version,
	Long: `Manage the local GGUF model store and the llama-server systemd service.

  llama-model list                    # what is on disk
  sudo llama-model pull qwen3.8:27b   # download from ollama.com/library
  sudo llama-model set qwen3.8:27b    # switch, restart and validate
  sudo llama-model rm gemma4:e4b      # delete, keeping shared blobs

pull talks to the Ollama registry directly, so the ollama daemon is not needed
to fetch or delete models.

set edits ` + llama.ConfPath + ` (LLAMA_MODEL, LLAMA_ALIAS and the --mmproj
inside LLAMA_ARGS), restarts the service and waits for /health. If the model
does not come up it restores the previous config. On success it also renames the
model in the OpenCode and Hermes configs, if they exist — each edited file gets
a .bak alongside it.`,
	SilenceUsage:  true,
	SilenceErrors: true,
}

// Execute runs the CLI and turns any error into a clean exit status.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func init() {
	rootCmd.PersistentFlags().StringVar(&llama.OllamaDir, "ollama-dir", llama.OllamaDir,
		"where Ollama keeps its manifests and blobs")
}
