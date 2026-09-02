package cmd

import (
	"fmt"
	"os"
	"strconv"

	"llama-model/internal/llama"

	"github.com/spf13/cobra"
)

var listCmd = &cobra.Command{
	Use:   "list",
	Short: "List the models Ollama has pulled",
	Long: `List the models Ollama has pulled, with the architecture and context
ceiling read straight from each GGUF header. A "*" marks the one llama-server
is currently serving.`,
	Args: cobra.NoArgs,
	RunE: func(_ *cobra.Command, _ []string) error { return runList() },
}

func init() {
	rootCmd.AddCommand(listCmd)
}

func runList() error {
	txt, _ := llama.ConfRead() // absent config is fine: nothing gets marked as in use
	inUse := llama.ConfGet(txt, "LLAMA_MODEL")

	models := llama.Manifests()
	if len(models) == 0 {
		return fmt.Errorf("no manifests under %s/manifests", llama.OllamaDir)
	}

	type row struct {
		mark, name, size, arch, ctx, vision string
		belowHermes                         bool
	}

	rows := make([]row, 0, len(models))
	nameW, archW := len("MODEL"), len("ARCH")
	loaded, unreadable := "", false

	for _, m := range models {
		var meta llama.Meta
		if m.Blob != "" {
			meta = llama.ReadGGUF(m.Blob)
		}

		r := row{mark: " ", name: m.Name, size: llama.HumanBytes(m.Size), arch: meta.Arch, ctx: "?", vision: "no"}
		if r.arch == "" {
			r.arch, unreadable = "?", true
		}
		if meta.Ctx > 0 {
			r.ctx = humanCtx(meta.Ctx)
			r.belowHermes = meta.Ctx < llama.HermesMin
		}
		if m.Proj != "" {
			r.vision = "yes"
		}
		if inUse != "" && m.Blob == inUse {
			r.mark, loaded = "*", m.Name
		}

		nameW = max(nameW, len(r.name))
		archW = max(archW, len(r.arch))
		rows = append(rows, r)
	}

	line := fmt.Sprintf("%%-2s%%-%ds  %%9s  %%-%ds  %%6s  %%s\n", nameW, archW)
	fmt.Printf(line, "", "MODEL", "SIZE", "ARCH", "CTX", "VISION")
	for _, r := range rows {
		fmt.Printf(line, r.mark, r.name, r.size, r.arch, r.ctx, r.vision)
	}

	// Notes go under the table so they never break up the columns.
	var notes []string
	if loaded != "" {
		notes = append(notes, fmt.Sprintf("* %s is being served now, at %s context",
			loaded, humanCtx(uint64(llama.ConfCtx(txt)))))
	}
	for _, r := range rows {
		if r.belowHermes {
			notes = append(notes, fmt.Sprintf("  %s: %s context is below the 64K Hermes needs for a fallback",
				r.name, r.ctx))
		}
	}
	if unreadable && os.Geteuid() != 0 {
		notes = append(notes, "  '?' means the blob could not be read — run with sudo")
	}
	if len(notes) > 0 {
		fmt.Println()
		for _, n := range notes {
			fmt.Println(n)
		}
	}
	return nil
}

// humanCtx renders a context length the way models are talked about: 262144 as
// "256K", not as a number the reader has to divide in their head.
func humanCtx(n uint64) string {
	switch {
	case n == 0:
		return "?"
	case n >= 1<<20:
		return strconv.FormatFloat(float64(n)/(1<<20), 'f', -1, 64) + "M"
	case n >= 1024 && n%1024 == 0:
		return strconv.FormatUint(n/1024, 10) + "K"
	default:
		return strconv.FormatUint(n, 10)
	}
}
