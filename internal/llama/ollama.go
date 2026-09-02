package llama

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Model is one entry from the Ollama manifest store.
type Model struct {
	Name string // qwen3.8:27b
	Blob string // path to the GGUF weights
	Proj string // path to the multimodal projector, empty if text-only
	Size int64
}

type manifest struct {
	Layers []struct {
		MediaType string `json:"mediaType"`
		Digest    string `json:"digest"`
		Size      int64  `json:"size"`
	} `json:"layers"`
}

// modelName derives the model reference from a manifest path, e.g.
// <root>/registry.ollama.ai/library/qwen3.8/27b -> "qwen3.8:27b".
// Returns "" for paths that are not a manifest. Both the listing and the
// removal planner go through here, so they can never disagree on identity.
func modelName(root, p string) string {
	rel, err := filepath.Rel(root, p)
	if err != nil {
		return ""
	}
	parts := strings.Split(rel, string(os.PathSeparator))
	if len(parts) < 3 { // host/name/tag at the very least
		return ""
	}
	parts = parts[1:] // drop the registry host
	if parts[0] == "library" {
		parts = parts[1:]
	}
	if len(parts) < 2 {
		return ""
	}
	return strings.Join(parts[:len(parts)-1], "/") + ":" + parts[len(parts)-1]
}

// Manifests walks the Ollama manifest tree and resolves each model to its blobs.
func Manifests() []Model {
	root := filepath.Join(OllamaDir, "manifests")
	var out []Model

	filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		name := modelName(root, p)
		if name == "" {
			return nil
		}

		data, err := os.ReadFile(p)
		if err != nil {
			return nil
		}
		var m manifest
		if json.Unmarshal(data, &m) != nil {
			return nil
		}

		e := Model{Name: name}
		for _, l := range m.Layers {
			blob := filepath.Join(OllamaDir, "blobs", strings.ReplaceAll(l.Digest, ":", "-"))
			switch {
			case strings.HasSuffix(l.MediaType, ".model"):
				e.Blob, e.Size = blob, l.Size
			case strings.HasSuffix(l.MediaType, ".projector"):
				e.Proj = blob
			}
		}
		out = append(out, e)
		return nil
	})

	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// Find matches an exact name first, then falls back to a substring match.
func Find(models []Model, needle string) (Model, error) {
	var hits []Model
	for _, m := range models {
		if m.Name == needle {
			return m, nil
		}
		if strings.Contains(m.Name, needle) {
			hits = append(hits, m)
		}
	}
	switch len(hits) {
	case 0:
		return Model{}, fmt.Errorf("no model matches %q. See: llama-model list", needle)
	case 1:
		return hits[0], nil
	}
	names := make([]string, len(hits))
	for i, m := range hits {
		names[i] = m.Name
	}
	return Model{}, fmt.Errorf("ambiguous: %s", strings.Join(names, ", "))
}
