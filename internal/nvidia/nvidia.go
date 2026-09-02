// Package nvidia wraps nvidia-smi. It shells out on purpose: NVML would need
// cgo, and this binary is built static so it keeps working across driver
// upgrades.
package nvidia

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// GPU is a card as the driver enumerates it.
type GPU struct {
	Index    int
	Name     string
	UUID     string
	MemTotal int // MiB
}

// Short returns the UUID tail, enough to tell two cards apart.
func (g GPU) Short() string {
	if i := strings.LastIndex(g.UUID, "-"); i >= 0 {
		return g.UUID[i+1:]
	}
	return g.UUID
}

// Sample is one reading of a card.
type Sample struct {
	Time       time.Time
	Index      int
	Temp       int     // C
	MemTemp    int     // C, -1 when the card has no memory sensor
	Power      float64 // W
	PowerLimit float64 // W
	ClockSM    int     // MHz
	Util       int     // %
	MemUsed    int     // MiB
	Throttle   uint64  // clocks_event_reasons.active bitmask
	Down       bool    // the card stopped answering
}

const (
	listQuery   = "index,name,uuid,memory.total"
	sampleQuery = "index,temperature.gpu,temperature.memory,power.draw,power.limit," +
		"clocks.sm,utilization.gpu,memory.used,clocks_event_reasons.active"
)

func run(args ...string) (string, error) {
	out, err := exec.Command("nvidia-smi", args...).Output()
	if err != nil {
		if _, lookErr := exec.LookPath("nvidia-smi"); lookErr != nil {
			return "", fmt.Errorf("nvidia-smi not found — no NVIDIA driver installed?")
		}
		return "", fmt.Errorf("nvidia-smi: %w", err)
	}
	return string(out), nil
}

// List enumerates the cards on the machine.
func List() ([]GPU, error) {
	out, err := run("--query-gpu="+listQuery, "--format=csv,noheader,nounits")
	if err != nil {
		return nil, err
	}
	gpus := parseGPUs(out)
	if len(gpus) == 0 {
		return nil, fmt.Errorf("no GPUs reported by nvidia-smi")
	}
	return gpus, nil
}

func parseGPUs(out string) []GPU {
	var gpus []GPU
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		f := splitCSV(line)
		if len(f) < 4 {
			continue
		}
		gpus = append(gpus, GPU{
			Index:    atoi(f[0]),
			Name:     f[1],
			UUID:     f[2],
			MemTotal: atoi(f[3]),
		})
	}
	return gpus
}

// Select resolves a user's choice: empty or "all" means every card, otherwise
// an index, a UUID, or any substring of the name ("v100", "4070").
func Select(sel string) ([]GPU, error) {
	gpus, err := List()
	if err != nil {
		return nil, err
	}
	if sel == "" || strings.EqualFold(sel, "all") {
		return gpus, nil
	}

	var hits []GPU
	for _, g := range gpus {
		switch {
		case strconv.Itoa(g.Index) == sel,
			strings.EqualFold(g.UUID, sel),
			strings.Contains(strings.ToLower(g.Name), strings.ToLower(sel)):
			hits = append(hits, g)
		}
	}
	switch len(hits) {
	case 0:
		names := make([]string, len(gpus))
		for i, g := range gpus {
			names[i] = fmt.Sprintf("%d=%s", g.Index, g.Name)
		}
		return nil, fmt.Errorf("no GPU matches %q (have: %s)", sel, strings.Join(names, ", "))
	case 1:
		return hits, nil
	default:
		return nil, fmt.Errorf("%q matches %d GPUs — use the index or the UUID", sel, len(hits))
	}
}

// Poll reads the current state of the given cards. A card missing from the
// output is reported as Down rather than dropped, which is the whole point when
// watching a card that falls off the bus.
func Poll(gpus []GPU) []Sample {
	now := time.Now()
	byIndex := map[int]Sample{}

	if out, err := run("--query-gpu="+sampleQuery, "--format=csv,noheader,nounits"); err == nil {
		byIndex = parseSamples(out, now)
	}

	samples := make([]Sample, len(gpus))
	for i, g := range gpus {
		s, ok := byIndex[g.Index]
		if !ok {
			s = Sample{Time: now, Index: g.Index, Down: true, MemTemp: -1}
		}
		samples[i] = s
	}
	return samples
}

func parseSamples(out string, now time.Time) map[int]Sample {
	byIndex := map[int]Sample{}
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		f := splitCSV(line)
		if len(f) < 9 {
			continue
		}
		idx := atoi(f[0])
		byIndex[idx] = Sample{
			Time:       now,
			Index:      idx,
			Temp:       atoi(f[1]),
			MemTemp:    atoiNA(f[2]),
			Power:      atof(f[3]),
			PowerLimit: atof(f[4]),
			ClockSM:    atoi(f[5]),
			Util:       atoi(f[6]),
			MemUsed:    atoi(f[7]),
			Throttle:   atohex(f[8]),
		}
	}
	return byIndex
}

// Throttle reason bits, from NVML's nvmlClocksEventReason.
var throttleBits = []struct {
	bit  uint64
	name string
}{
	{1 << 0, "idle"},
	{1 << 1, "app clocks"},
	{1 << 2, "sw power cap"},
	{1 << 3, "hw slowdown"},
	{1 << 4, "sync boost"},
	{1 << 5, "sw thermal"},
	{1 << 6, "hw thermal"},
	{1 << 7, "power brake"},
	{1 << 8, "display clocks"},
}

// Reasons decodes the throttle mask into readable causes.
func Reasons(mask uint64) []string {
	var out []string
	for _, b := range throttleBits {
		if mask&b.bit != 0 {
			out = append(out, b.name)
		}
	}
	return out
}

// Throttled reports whether the card is being held back for a reason that
// actually costs performance — idle and app-clock limits do not count.
func Throttled(mask uint64) bool {
	const bad = 1<<2 | 1<<3 | 1<<5 | 1<<6 | 1<<7
	return mask&bad != 0
}

func splitCSV(line string) []string {
	parts := strings.Split(line, ",")
	for i := range parts {
		parts[i] = strings.TrimSpace(parts[i])
	}
	return parts
}

func atoi(s string) int {
	n, _ := strconv.Atoi(strings.TrimSpace(s))
	return n
}

// atoiNA maps the "[N/A]" the driver prints for absent sensors to -1.
func atoiNA(s string) int {
	if n, err := strconv.Atoi(strings.TrimSpace(s)); err == nil {
		return n
	}
	return -1
}

func atof(s string) float64 {
	f, _ := strconv.ParseFloat(strings.TrimSpace(s), 64)
	return f
}

func atohex(s string) uint64 {
	n, _ := strconv.ParseUint(strings.TrimPrefix(strings.TrimSpace(s), "0x"), 16, 64)
	return n
}
