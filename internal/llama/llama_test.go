package llama

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ggufString encodes a length-prefixed GGUF string.
func ggufString(s string) []byte {
	b := make([]byte, 8)
	binary.LittleEndian.PutUint64(b, uint64(len(s)))
	return append(b, s...)
}

func u32(v uint32) []byte {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, v)
	return b
}

func u64(v uint64) []byte {
	b := make([]byte, 8)
	binary.LittleEndian.PutUint64(b, v)
	return b
}

// TestReadGGUF builds a synthetic header, including the array types the parser
// has to seek past to reach the fields it wants.
func TestReadGGUF(t *testing.T) {
	var buf bytes.Buffer
	buf.WriteString("GGUF")
	buf.Write(u32(3)) // version
	buf.Write(u64(0)) // tensor count
	buf.Write(u64(4)) // kv count

	buf.Write(ggufString("general.architecture"))
	buf.Write(u32(8))
	buf.Write(ggufString("qwen3"))

	buf.Write(ggufString("tokenizer.ggml.tokens")) // array of strings
	buf.Write(u32(9))
	buf.Write(u32(8))
	buf.Write(u64(2))
	buf.Write(ggufString("a"))
	buf.Write(ggufString("bb"))

	buf.Write(ggufString("general.junk")) // array of uint32
	buf.Write(u32(9))
	buf.Write(u32(4))
	buf.Write(u64(3))
	buf.Write(make([]byte, 12))

	buf.Write(ggufString("qwen3.context_length"))
	buf.Write(u32(4))
	buf.Write(u32(262144))

	path := filepath.Join(t.TempDir(), "m.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}

	got := ReadGGUF(path)
	if got.Arch != "qwen3" || got.Ctx != 262144 {
		t.Fatalf("ReadGGUF = %+v, want {qwen3 262144}", got)
	}
	if m := ReadGGUF("/dev/null"); m.Arch != "" || m.Ctx != 0 {
		t.Fatalf("ReadGGUF(/dev/null) = %+v, want zero value", m)
	}
}

func TestJSONBlock(t *testing.T) {
	// the brace inside the string must not throw off the depth counter
	j := `{"a":{"x":"}{"},"llamacpp":{"models":{"m:1":{"k":1}}},"b":2}`
	s, e, ok := jsonBlock(j, "llamacpp")
	if !ok {
		t.Fatal("block not found")
	}
	if want := `{"models":{"m:1":{"k":1}}}`; j[s:e] != want {
		t.Fatalf("jsonBlock = %s, want %s", j[s:e], want)
	}
	if _, _, ok := jsonBlock(j, "nope"); ok {
		t.Fatal("found a block that does not exist")
	}
}

// The file is .jsonc, so a brace inside a comment must not move the depth.
func TestJSONBlockIgnoresComments(t *testing.T) {
	j := `{
  // a comment with a stray { brace
  "llamacpp":{"models":{"m:1":{"k":1}}},
  /* block } comment */
  "b":2}`
	s, e, ok := jsonBlock(j, "llamacpp")
	if !ok {
		t.Fatal("block not found")
	}
	if want := `{"models":{"m:1":{"k":1}}}`; j[s:e] != want {
		t.Fatalf("jsonBlock = %s, want %s", j[s:e], want)
	}
}

func TestConfApplyAddsMissingCtx(t *testing.T) {
	src := "LLAMA_MODEL=/old\nLLAMA_ALIAS=old\nLLAMA_ARGS=--n-gpu-layers 999 --jinja\n"
	out := ConfApply(src, Model{Name: "new:1b", Blob: "/new"}, 65536)
	if ConfCtx(out) != 65536 {
		t.Errorf("--ctx-size was not added:\n%s", out)
	}
}

func TestConfApplyKeepsQuoting(t *testing.T) {
	src := "LLAMA_MODEL=/old\nLLAMA_ALIAS=old\nLLAMA_ARGS=\"--jinja --ctx-size 4096\"\n"
	out := ConfApply(src, Model{Name: "new:1b", Blob: "/new"}, 8192)
	if !strings.Contains(out, `LLAMA_ARGS="--jinja --ctx-size 8192"`) {
		t.Errorf("quoting not preserved:\n%s", out)
	}
}

func TestPatchOpencode(t *testing.T) {
	const src = `{"provider":{
      "llamacpp":{"name":"llama.cpp (V100)","models":{
        "muse-glimmer:30b":{"name":"Muse Glimmer 30B (V100)","tools":true,"limit":{"context":131072}}}},
      "ollama":{"models":{"muse-glimmer:30b":{"name":"do not touch"}}}}}`

	path := filepath.Join(t.TempDir(), "opencode.jsonc")
	if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}

	patchOpencode(path, "muse-glimmer:30b", "qwen3.8:27b", 262144, true)
	out := read(t, path)

	if !strings.Contains(out, `"qwen3.8:27b":{`) || !strings.Contains(out, `"name":"Qwen3.8 27B (V100)"`) {
		t.Errorf("model key or display name not updated:\n%s", out)
	}
	if !strings.Contains(out, `"context":262144`) {
		t.Errorf("context limit not updated:\n%s", out)
	}
	if strings.Count(out, "muse-glimmer:30b") != 1 {
		t.Errorf("touched the ollama provider:\n%s", out)
	}
	// "tools" is not in OpenCode's schema; the key is "tool_call"
	if strings.Contains(out, `"tools"`) || !strings.Contains(out, `"tool_call"`) {
		t.Errorf("tools key not normalised:\n%s", out)
	}
	// vision needs both flags or OpenCode never sends the image
	if !strings.Contains(out, `"attachment": true`) || !strings.Contains(out, `"image"`) {
		t.Errorf("vision flags missing:\n%s", out)
	}

	// same name, new context: what `set --ctx max` does
	patchOpencode(path, "qwen3.8:27b", "qwen3.8:27b", 131072, true)
	if !strings.Contains(read(t, path), `"context":131072`) {
		t.Error("context not updated when the model name is unchanged")
	}

	// switching to a text-only model must not leave attachment claiming vision
	patchOpencode(path, "qwen3.8:27b", "qwen3.8:27b", 131072, false)
	out = read(t, path)
	if !strings.Contains(out, `"attachment": false`) || strings.Contains(out, `"image"`) {
		t.Errorf("vision flags not cleared for a text-only model:\n%s", out)
	}
}

func TestPatchHermes(t *testing.T) {
	const src = `custom_providers:
  - name: llamacpp
    models:
      - muse-glimmer:30b
fallback_providers:
  - provider: llamacpp
    model: muse-glimmer:30b
  - provider: openrouter
    model: qwen/qwen3.8-max
`
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}

	patchHermes(path, "muse-glimmer:30b", "qwen3.8:27b")
	out := read(t, path)

	if n := strings.Count(out, `"qwen3.8:27b"`); n != 2 {
		t.Errorf("replaced %d lines, want 2:\n%s", n, out)
	}
	if !strings.Contains(out, "model: qwen/qwen3.8-max") {
		t.Errorf("clobbered the openrouter line:\n%s", out)
	}
	if !strings.Contains(out, "  - provider: llamacpp") {
		t.Errorf("mangled a mapping item:\n%s", out)
	}
	if !strings.Contains(out, `      - "qwen3.8:27b"`) {
		t.Errorf("lost the list indentation:\n%s", out)
	}
}

func TestConfApply(t *testing.T) {
	const src = `LLAMA_MODEL=/blobs/sha256-old
LLAMA_ALIAS=muse-glimmer:30b
LLAMA_PORT=11435
LLAMA_ARGS=--mmproj /blobs/sha256-oldproj --n-gpu-layers 999 --ctx-size 32768 --jinja
`
	out := ConfApply(src, Model{Name: "qwen3.8:27b", Blob: "/blobs/sha256-new"}, 262144)

	for _, want := range []string{
		"LLAMA_MODEL=/blobs/sha256-new",
		"LLAMA_ALIAS=qwen3.8:27b",
		"--ctx-size 262144",
		"LLAMA_PORT=11435", // untouched
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in:\n%s", want, out)
		}
	}
	// text-only model: the projector must be dropped, not left dangling
	if strings.Contains(out, "mmproj") {
		t.Errorf("--mmproj survived for a text-only model:\n%s", out)
	}
	if ConfCtx(out) != 262144 {
		t.Errorf("ConfCtx = %d, want 262144", ConfCtx(out))
	}
}

func TestFindModel(t *testing.T) {
	models := []Model{{Name: "qwen3.8:27b"}, {Name: "qwen3-coder:30b"}, {Name: "gemma4:e4b"}}

	if m, err := Find(models, "gemma"); err != nil || m.Name != "gemma4:e4b" {
		t.Errorf("substring match failed: %+v %v", m, err)
	}
	if m, err := Find(models, "qwen3.8:27b"); err != nil || m.Name != "qwen3.8:27b" {
		t.Errorf("exact match failed: %+v %v", m, err)
	}
	if _, err := Find(models, "qwen3"); err == nil {
		t.Error("ambiguous needle should fail")
	}
	if _, err := Find(models, "llama"); err == nil {
		t.Error("missing model should fail")
	}
}

func read(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}
