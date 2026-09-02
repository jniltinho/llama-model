package cmd

import (
	"llama-model/internal/llama"

	"github.com/spf13/cobra"
)

var pullCmd = &cobra.Command{
	Use:   "pull <model>",
	Short: "Download a model from the Ollama registry (needs sudo)",
	Long: `Download a model straight from https://ollama.com/library into the local
store, without the ollama daemon.

Blobs are verified against their sha256 and an interrupted download resumes
where it stopped, so running pull again after a broken connection is cheap.`,
	Example: `  sudo llama-model pull qwen3.8:27b
  sudo llama-model pull gemma4:e4b`,
	Args: cobra.ExactArgs(1),
	RunE: func(_ *cobra.Command, args []string) error { return llama.Pull(args[0]) },
}

func init() {
	rootCmd.AddCommand(pullCmd)
}
