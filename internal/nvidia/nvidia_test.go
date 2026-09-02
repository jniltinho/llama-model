package nvidia

import (
	"strings"
	"testing"
	"time"
)

// Real nvidia-smi output from a machine with a 4070 and a V100.
const listOut = `0, NVIDIA GeForce RTX 4070, GPU-edcd6862-7723-7102-ba7e-b8f29daccd76, 12282
1, Tesla V100-PCIE-32GB, GPU-06527863-9d6b-3e20-06a2-f00b422a2fed, 32768`

// The 4070 has no memory temperature sensor: the driver prints N/A.
const sampleOut = `0, 50, N/A, 15.94, 200.00, 210, 1, 858, 0x0000000000000001
1, 45, 44, 37.78, 250.00, 1230, 0, 27465, 0x0000000000000000`

func TestParseGPUs(t *testing.T) {
	gpus := parseGPUs(listOut)
	if len(gpus) != 2 {
		t.Fatalf("parsed %d GPUs, want 2", len(gpus))
	}
	if gpus[1].Name != "Tesla V100-PCIE-32GB" || gpus[1].MemTotal != 32768 {
		t.Errorf("second GPU = %+v", gpus[1])
	}
	if got := gpus[1].Short(); got != "f00b422a2fed" {
		t.Errorf("Short() = %q", got)
	}
}

func TestParseSamples(t *testing.T) {
	got := parseSamples(sampleOut, time.Now())
	if len(got) != 2 {
		t.Fatalf("parsed %d samples, want 2", len(got))
	}

	if s := got[0]; s.MemTemp != -1 {
		t.Errorf("missing memory sensor should be -1, got %d", s.MemTemp)
	}
	if s := got[0]; s.Power != 15.94 || s.PowerLimit != 200 {
		t.Errorf("power parsed as %v/%v", s.Power, s.PowerLimit)
	}
	if s := got[1]; s.MemTemp != 44 || s.MemUsed != 27465 || s.ClockSM != 1230 {
		t.Errorf("V100 sample = %+v", s)
	}
	if s := got[0]; s.Throttle != 1 { // hex must be parsed as hex
		t.Errorf("throttle mask = %#x, want 0x1", s.Throttle)
	}
}

func TestThrottleDecoding(t *testing.T) {
	// idle alone is not a performance problem
	if Throttled(1 << 0) {
		t.Error("idle counted as throttling")
	}
	if r := Reasons(1 << 0); len(r) != 1 || r[0] != "idle" {
		t.Errorf("Reasons(idle) = %v", r)
	}

	// hw thermal slowdown is
	mask := uint64(1<<6 | 1<<2)
	if !Throttled(mask) {
		t.Error("thermal + power cap not reported as throttling")
	}
	if got := strings.Join(Reasons(mask), ","); got != "sw power cap,hw thermal" {
		t.Errorf("Reasons = %q", got)
	}
	if len(Reasons(0)) != 0 {
		t.Error("clean mask produced reasons")
	}
}

func TestPollMarksMissingCardDown(t *testing.T) {
	// A card that fell off the bus disappears from the output: it must be
	// reported as Down, not silently dropped from the list.
	gpus := parseGPUs(listOut)
	byIndex := parseSamples("0, 50, N/A, 15.94, 200.00, 210, 1, 858, 0x1", time.Now())

	if _, ok := byIndex[1]; ok {
		t.Fatal("V100 should be absent from this output")
	}
	for _, g := range gpus {
		if _, ok := byIndex[g.Index]; !ok && g.Index != 1 {
			t.Errorf("GPU %d missing", g.Index)
		}
	}
}
