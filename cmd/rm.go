package cmd

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"llama-model/internal/llama"

	"github.com/spf13/cobra"
)

var (
	rmYes   bool
	rmForce bool
)

var rmCmd = &cobra.Command{
	Use:     "rm <model>",
	Aliases: []string{"remove", "delete"},
	Short:   "Delete a model from the local store (needs sudo)",
	Long: `Delete a model's manifest and the blobs no other model shares.

Refuses to delete the model llama-server is currently serving, since that would
leave the service unable to restart. Use --force to do it anyway.`,
	Example: `  sudo llama-model rm gemma4:e4b
  sudo llama-model rm gemma4:e4b --yes`,
	Args: cobra.ExactArgs(1),
	RunE: func(_ *cobra.Command, args []string) error { return runRemove(args[0]) },
}

func init() {
	rmCmd.Flags().BoolVarP(&rmYes, "yes", "y", false, "do not ask for confirmation")
	rmCmd.Flags().BoolVar(&rmForce, "force", false, "delete even if llama-server is serving it")
	rootCmd.AddCommand(rmCmd)
}

func runRemove(name string) error {
	if err := llama.EnsureWritable(); err != nil {
		return err
	}

	plan, err := llama.PlanRemoval(name)
	if err != nil {
		return err
	}

	fmt.Printf("%s\n  manifest: %s\n  blobs:    %d (%s)\n",
		plan.Model.Name, plan.Manifest, len(plan.Blobs), llama.HumanBytes(plan.Bytes))
	if plan.Shared > 0 {
		fmt.Printf("  kept:     %d blob(s) shared with another model\n", plan.Shared)
	}
	if plan.InUse {
		if !rmForce {
			return fmt.Errorf("llama-server is serving %s — switch models first "+
				"(llama-model set <other>) or pass --force", plan.Model.Name)
		}
		fmt.Println("  WARNING: llama-server is serving this model and will fail to restart")
	}

	if !rmYes && !confirm() {
		fmt.Println("aborted")
		return nil
	}
	if err := plan.Apply(); err != nil {
		return err
	}
	fmt.Printf("deleted %s, freed %s\n", plan.Model.Name, llama.HumanBytes(plan.Bytes))
	return nil
}

func confirm() bool {
	fmt.Print("delete? [y/N] ")
	line, err := bufio.NewReader(os.Stdin).ReadString('\n')
	if err != nil {
		return false
	}
	return strings.EqualFold(strings.TrimSpace(line), "y")
}
