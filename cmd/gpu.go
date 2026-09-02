package cmd

import (
	"encoding/csv"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"llama-model/internal/nvidia"

	"github.com/spf13/cobra"
)

var gpuCmd = &cobra.Command{
	Use:   "gpu",
	Short: "List and monitor the NVIDIA GPUs",
}

var gpuListCmd = &cobra.Command{
	Use:   "list",
	Short: "List the GPUs with their current state",
	Args:  cobra.NoArgs,
	RunE:  func(_ *cobra.Command, _ []string) error { return runGPUList() },
}

var (
	watchSelect   string
	watchInterval time.Duration
	watchCSV      string
)

var gpuWatchCmd = &cobra.Command{
	Use:   "watch",
	Short: "Watch temperature, power and clocks until interrupted",
	Long: `Watch the GPUs live.

Tracks peak temperature and power, decodes why the card is throttling, and if a
card falls off the bus it stops and prints the last good reading together with
the kernel's Xid messages — the state you need to tell a thermal problem from a
power one.

Pick a card by index, by UUID, or by any part of its name; without --gpu every
card is watched.`,
	Example: `  llama-model gpu watch
  llama-model gpu watch --gpu v100
  llama-model gpu watch --gpu 1 --interval 5s --csv ~/v100.csv`,
	Args: cobra.NoArgs,
	RunE: func(_ *cobra.Command, _ []string) error { return runGPUWatch() },
}

func init() {
	gpuWatchCmd.Flags().StringVar(&watchSelect, "gpu", "", "index, UUID or name fragment (default: all)")
	gpuWatchCmd.Flags().DurationVar(&watchInterval, "interval", 2*time.Second, "time between readings")
	gpuWatchCmd.Flags().StringVar(&watchCSV, "csv", "", "also append every reading to this CSV file")
	gpuCmd.AddCommand(gpuListCmd, gpuWatchCmd)
	rootCmd.AddCommand(gpuCmd)
}

func runGPUList() error {
	gpus, err := nvidia.Select("")
	if err != nil {
		return err
	}
	samples := nvidia.Poll(gpus)

	nameW := len("NAME")
	for _, g := range gpus {
		nameW = max(nameW, len(g.Name))
	}
	line := fmt.Sprintf("%%-4s%%-%ds  %%14s  %%17s  %%6s  %%9s  %%s\n", nameW)
	fmt.Printf(line, "IDX", "NAME", "UUID", "MEMORY", "TEMP", "POWER", "UTIL")

	for i, g := range gpus {
		s := samples[i]
		if s.Down {
			fmt.Printf(line, strconv.Itoa(g.Index), g.Name, g.Short(),
				"-", "-", "-", "not responding")
			continue
		}
		fmt.Printf(line,
			strconv.Itoa(g.Index), g.Name, g.Short(),
			fmt.Sprintf("%d / %d MiB", s.MemUsed, g.MemTotal),
			fmt.Sprintf("%d C", s.Temp),
			fmt.Sprintf("%.0f/%.0f W", s.Power, s.PowerLimit),
			fmt.Sprintf("%d%%", s.Util))
	}

	fmt.Println("\nPin a service to a card by UUID, not by index — indexes reorder across boots:")
	for _, g := range gpus {
		fmt.Printf("  CUDA_VISIBLE_DEVICES=%s   # %s\n", g.UUID, g.Name)
	}
	return nil
}

// peak tracks the worst values seen, which is what matters after a crash.
type peak struct {
	temp  int
	power float64
}

func runGPUWatch() error {
	gpus, err := nvidia.Select(watchSelect)
	if err != nil {
		return err
	}
	if watchInterval < time.Second {
		watchInterval = time.Second // nvidia-smi itself takes ~100ms per call
	}

	var w *csv.Writer
	if watchCSV != "" {
		f, err := os.OpenFile(watchCSV, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return err
		}
		defer f.Close()
		w = csv.NewWriter(f)
		defer w.Flush()
		if st, _ := f.Stat(); st.Size() == 0 {
			w.Write([]string{"timestamp", "index", "name", "temp_c", "mem_temp_c",
				"power_w", "clock_sm_mhz", "util_pct", "mem_used_mib", "throttle"})
		}
	}

	names := make([]string, len(gpus))
	nameW := len("NAME")
	for i, g := range gpus {
		names[i] = shortName(g.Name)
		nameW = max(nameW, len(names[i]))
	}

	fmt.Printf("watching %s every %s", strings.Join(names, " and "), watchInterval)
	if watchCSV != "" {
		fmt.Printf(", logging to %s", watchCSV)
	}
	fmt.Println("  (Ctrl+C to stop)")

	// Header sits above the block that gets redrawn, so it stays put.
	rowFmt := fmt.Sprintf("%%-8s  %%3s  %%-%ds  %%-10s  %%7s  %%8s  %%4s  %%17s  %%9s  %%s", nameW)
	fmt.Println()
	fmt.Printf(rowFmt+"\n", "TIME", "GPU", "NAME", "TEMP (MEM)", "POWER", "CLOCK", "USE", "MEMORY", "PEAK", "STATUS")

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

	peaks := make([]peak, len(gpus))
	last := make([]string, len(gpus))
	tty := isTTY()
	printed := 0

	for {
		samples := nvidia.Poll(gpus)

		for i, s := range samples {
			if s.Down {
				reportCrash(gpus[i], last[i], peaks[i])
				return fmt.Errorf("GPU %d (%s) stopped responding", gpus[i].Index, gpus[i].Name)
			}
			peaks[i].temp = max(peaks[i].temp, s.Temp)
			if s.Power > peaks[i].power {
				peaks[i].power = s.Power
			}
			last[i] = row(rowFmt, gpus[i], s, peaks[i])
			if w != nil {
				w.Write([]string{
					s.Time.Format(time.RFC3339), strconv.Itoa(s.Index), gpus[i].Name,
					strconv.Itoa(s.Temp), strconv.Itoa(s.MemTemp),
					strconv.FormatFloat(s.Power, 'f', 1, 64), strconv.Itoa(s.ClockSM),
					strconv.Itoa(s.Util), strconv.Itoa(s.MemUsed),
					strings.Join(nvidia.Reasons(s.Throttle), "|"),
				})
				w.Flush()
			}
		}

		if tty && printed > 0 {
			fmt.Printf("\033[%dA", printed) // redraw in place
		}
		printed = 0
		clear := ""
		if tty {
			clear = "\033[K" // wipe leftovers from a longer previous line
		}
		for i := range samples {
			fmt.Println(last[i] + clear)
			printed++
		}

		select {
		case <-stop:
			fmt.Println("\nstopped. peaks:")
			for i, g := range gpus {
				fmt.Printf("  %d %-*s  %d C, %.0f W\n",
					g.Index, nameW, shortName(g.Name), peaks[i].temp, peaks[i].power)
			}
			return nil
		case <-time.After(watchInterval):
		}
	}
}

// shortName drops the vendor prefix every card repeats, to keep the column
// narrow: "NVIDIA GeForce RTX 4070" -> "GeForce RTX 4070".
func shortName(s string) string { return strings.TrimPrefix(s, "NVIDIA ") }

// row renders one reading into the same columns as the header, so cards with
// and without a memory sensor still line up under each other.
func row(format string, g nvidia.GPU, s nvidia.Sample, p peak) string {
	temp := fmt.Sprintf("%d C", s.Temp)
	if s.MemTemp >= 0 { // only some cards have a memory sensor
		temp += fmt.Sprintf(" (%d)", s.MemTemp)
	}

	status := "-"
	if reasons := nvidia.Reasons(s.Throttle); len(reasons) > 0 {
		status = strings.Join(reasons, ", ")
		if nvidia.Throttled(s.Throttle) {
			status = "THROTTLED: " + status
		}
	}

	return fmt.Sprintf(format,
		s.Time.Format("15:04:05"),
		strconv.Itoa(g.Index),
		shortName(g.Name),
		temp,
		fmt.Sprintf("%.0f W", s.Power),
		fmt.Sprintf("%d MHz", s.ClockSM),
		fmt.Sprintf("%d%%", s.Util),
		fmt.Sprintf("%d / %d MiB", s.MemUsed, g.MemTotal),
		fmt.Sprintf("%dC/%.0fW", p.temp, p.power),
		status)
}

// reportCrash prints what the script this replaces printed: the last good
// reading, the peaks, the kernel's Xid lines, and how to read them.
func reportCrash(g nvidia.GPU, last string, p peak) {
	fmt.Printf("\n\n==============================================\n")
	fmt.Printf("GPU %d (%s) DROPPED AT %s\n", g.Index, g.Name, time.Now().Format("15:04:05"))
	fmt.Printf("last good reading: %s\n", last)
	fmt.Printf("peaks: %dC, %.0fW\n\n", p.temp, p.power)

	fmt.Println("--- kernel Xid messages (last 5 min) ---")
	out, err := exec.Command("journalctl", "-k", "--since", "-5 min", "--no-pager").Output()
	if err != nil {
		fmt.Println("  (could not read the kernel log — try with sudo)")
	} else {
		hits := 0
		for _, l := range strings.Split(string(out), "\n") {
			if low := strings.ToLower(l); strings.Contains(low, "xid") || strings.Contains(low, "fallen off") {
				fmt.Println("  " + l)
				hits++
			}
		}
		if hits == 0 {
			fmt.Println("  (none)")
		}
	}

	fmt.Println()
	if p.temp >= 83 {
		fmt.Printf("Peak hit %dC: thermal. The V100 is passively cooled and needs forced airflow.\n", p.temp)
	} else {
		fmt.Printf("Peak was only %dC at %.0fW: look at power delivery (PSU, PCIe cables) before cooling.\n",
			p.temp, p.power)
	}
	if watchCSV != "" {
		fmt.Printf("full log: %s\n", watchCSV)
	}
}

func isTTY() bool {
	st, err := os.Stdout.Stat()
	return err == nil && st.Mode()&os.ModeCharDevice != 0
}
