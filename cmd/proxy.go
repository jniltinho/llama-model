package cmd

import (
	"llama-model/internal/llama"

	"github.com/spf13/cobra"
)

var (
	proxyListen   string
	proxyUpstream string
	proxyVerbose  bool
)

var proxyCmd = &cobra.Command{
	Use:   "proxy",
	Short: "Anthropic-protocol front end for llama-server (for Claude Code)",
	Long: `Sit in front of llama-server and fix up Anthropic requests.

Claude Code sends some of its instructions as a message with role "system" in
the middle of the conversation. Qwen's chat template rejects that outright
("System message must be at the beginning") and the request fails with a 500.
This proxy lifts those into the top-level system field and forwards everything
else untouched, streaming included.

Point ANTHROPIC_BASE_URL at this proxy instead of at llama-server — which is
what claude-local.sh does for you.`,
	Example: `  llama-model proxy
  llama-model proxy --listen 127.0.0.1:11436 --upstream http://127.0.0.1:11435 -v`,
	Args: cobra.NoArgs,
	RunE: func(_ *cobra.Command, _ []string) error {
		return llama.Proxy(proxyListen, proxyUpstream, proxyVerbose)
	},
}

func init() {
	proxyCmd.Flags().StringVar(&proxyListen, "listen", "127.0.0.1:11436", "address to listen on")
	proxyCmd.Flags().StringVar(&proxyUpstream, "upstream", "http://127.0.0.1:11435", "llama-server address")
	proxyCmd.Flags().BoolVarP(&proxyVerbose, "verbose", "v", false, "log every rewritten request")
	rootCmd.AddCommand(proxyCmd)
}
