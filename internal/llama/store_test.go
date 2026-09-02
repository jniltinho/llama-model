package llama

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// fakeStore builds an Ollama-shaped store: two models sharing one blob.
func fakeStore(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()

	blobs := filepath.Join(dir, "blobs")
	if err := os.MkdirAll(blobs, 0o755); err != nil {
		t.Fatal(err)
	}
	write := func(digest string, size int) {
		if err := os.WriteFile(filepath.Join(blobs, "sha256-"+digest), make([]byte, size), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("aaa", 4096) // qwen weights
	write("bbb", 512)  // qwen projector
	write("ccc", 2048) // gemma weights
	write("ddd", 64)   // license, shared by both

	manifest := func(ns, name, tag, body string) {
		d := filepath.Join(dir, "manifests", "registry.ollama.ai", ns, name)
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(d, tag), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	layer := func(kind, digest string, size int) string {
		return fmt.Sprintf(`{"mediaType":"application/vnd.ollama.image.%s","digest":"sha256:%s","size":%d}`,
			kind, digest, size)
	}
	manifest("library", "qwen3.8", "27b", `{"layers":[`+
		layer("model", "aaa", 4096)+","+layer("projector", "bbb", 512)+","+
		layer("license", "ddd", 64)+`]}`)
	manifest("library", "gemma4", "e4b", `{"layers":[`+
		layer("model", "ccc", 2048)+","+layer("license", "ddd", 64)+`]}`)

	old := OllamaDir
	OllamaDir = dir
	t.Cleanup(func() { OllamaDir = old })
	return dir
}

// TestManifests would have caught the store path being mangled: it walks the
// real directory layout instead of trusting the constant.
func TestManifests(t *testing.T) {
	fakeStore(t)

	got := Manifests()
	if len(got) != 2 {
		t.Fatalf("found %d models, want 2: %+v", len(got), got)
	}
	if got[0].Name != "gemma4:e4b" || got[1].Name != "qwen3.8:27b" {
		t.Fatalf("names/order wrong: %s, %s", got[0].Name, got[1].Name)
	}

	qwen := got[1]
	if filepath.Base(qwen.Blob) != "sha256-aaa" {
		t.Errorf("model blob = %s", qwen.Blob)
	}
	if filepath.Base(qwen.Proj) != "sha256-bbb" {
		t.Errorf("projector blob = %s", qwen.Proj)
	}
	if qwen.Size != 4096 {
		t.Errorf("size = %d, want 4096 (weights only)", qwen.Size)
	}
	if got[0].Proj != "" {
		t.Error("gemma4 has no projector but one was reported")
	}
}

func TestPlanRemovalKeepsSharedBlobs(t *testing.T) {
	dir := fakeStore(t)

	plan, err := PlanRemoval("gemma4")
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Blobs) != 1 || filepath.Base(plan.Blobs[0]) != "sha256-ccc" {
		t.Fatalf("would delete %v, want only sha256-ccc", plan.Blobs)
	}
	if plan.Shared != 1 {
		t.Errorf("shared = %d, want 1 (the license blob)", plan.Shared)
	}
	if plan.Bytes != 2048 {
		t.Errorf("freed = %d, want 2048", plan.Bytes)
	}

	if err := plan.Apply(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "blobs", "sha256-ccc")); !os.IsNotExist(err) {
		t.Error("blob still on disk")
	}
	if _, err := os.Stat(filepath.Join(dir, "blobs", "sha256-ddd")); err != nil {
		t.Error("deleted a blob another model still needs")
	}
	if _, err := os.Stat(filepath.Join(dir, "manifests/registry.ollama.ai/library/gemma4")); !os.IsNotExist(err) {
		t.Error("empty manifest directory was not pruned")
	}
	if len(Manifests()) != 1 {
		t.Error("model still listed after removal")
	}
}

// Two models with the same name under different namespaces must not be
// confused: deleting one has to leave the other alone.
func TestPlanRemovalDisambiguatesNamespace(t *testing.T) {
	dir := fakeStore(t)

	other := filepath.Join(dir, "manifests", "registry.ollama.ai", "someone", "gemma4")
	if err := os.MkdirAll(other, 0o755); err != nil {
		t.Fatal(err)
	}
	body := `{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:aaa","size":4096}]}`
	if err := os.WriteFile(filepath.Join(other, "e4b"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}

	plan, err := PlanRemoval("gemma4:e4b") // the library one
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(filepath.Dir(filepath.Dir(plan.Manifest))) != "library" {
		t.Fatalf("picked the wrong namespace: %s", plan.Manifest)
	}
	if err := plan.Apply(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(other, "e4b")); err != nil {
		t.Error("deleted the other namespace's manifest")
	}
	if _, err := os.Stat(filepath.Join(dir, "blobs", "sha256-aaa")); err != nil {
		t.Error("deleted a blob the other namespace references")
	}
}

func TestParseRef(t *testing.T) {
	cases := map[string]string{
		"qwen3.8:27b":                         "library/qwen3.8/27b",
		"gemma4":                              "library/gemma4/latest",
		"hf.co/user/model:q4":                 "hf.co/user/model/q4",
		"registry.ollama.ai/library/qwen:32b": "library/qwen/32b",
	}
	for in, want := range cases {
		r := ParseRef(in)
		if got := r.Namespace + "/" + r.Name + "/" + r.Tag; got != want {
			t.Errorf("ParseRef(%q) = %s, want %s", in, got, want)
		}
	}
	if s := ParseRef("qwen3.8:27b").String(); s != "qwen3.8:27b" {
		t.Errorf("String() = %s", s)
	}
}

func TestShortLabel(t *testing.T) {
	cases := map[string]string{
		"application/vnd.ollama.image.model":             "model",
		"application/vnd.ollama.image.projector":         "projector",
		"application/vnd.docker.container.image.v1+json": "abcdef123456",
	}
	for mt, want := range cases {
		if got := shortLabel(mt, "abcdef123456789"); got != want {
			t.Errorf("shortLabel(%q) = %q, want %q", mt, got, want)
		}
	}
}

func TestHumanBytes(t *testing.T) {
	cases := map[int64]string{0: "0 B", 512: "512 B", 1536: "1.5 KB", 17 << 30: "17.0 GB"}
	for n, want := range cases {
		if got := HumanBytes(n); got != want {
			t.Errorf("HumanBytes(%d) = %s, want %s", n, got, want)
		}
	}
}
