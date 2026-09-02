package llama

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// Removal describes what deleting a model would do.
type Removal struct {
	Model    Model
	Manifest string   // manifest file to delete
	Blobs    []string // blobs no other model references
	Shared   int      // blobs kept because another model uses them
	Bytes    int64    // disk freed by deleting Blobs
	InUse    bool     // llama-server is currently serving this model
}

// PlanRemoval works out which blobs are safe to delete: a model's blobs are
// only removed when no other manifest references them.
func PlanRemoval(name string) (*Removal, error) {
	m, err := Find(Manifests(), name)
	if err != nil {
		return nil, err
	}

	manifestPath, mine, err := manifestOf(m.Name)
	if err != nil {
		return nil, err
	}

	// every digest referenced by some other model
	others := map[string]bool{}
	root := filepath.Join(OllamaDir, "manifests")
	filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || p == manifestPath {
			return nil
		}
		data, err := os.ReadFile(p)
		if err != nil {
			return nil
		}
		var rm registryManifest
		if json.Unmarshal(data, &rm) != nil {
			return nil
		}
		for _, l := range rm.blobs() {
			others[l.Digest] = true
		}
		return nil
	})

	r := &Removal{Model: m, Manifest: manifestPath}
	txt, _ := ConfRead()
	inUseBlob := ConfGet(txt, "LLAMA_MODEL")

	for _, l := range mine.blobs() {
		blob := filepath.Join(OllamaDir, "blobs", strings.ReplaceAll(l.Digest, ":", "-"))
		if others[l.Digest] {
			r.Shared++
			continue
		}
		if blob == inUseBlob {
			r.InUse = true
		}
		if st, err := os.Stat(blob); err == nil {
			r.Bytes += st.Size()
			r.Blobs = append(r.Blobs, blob)
		}
	}
	return r, nil
}

// Apply deletes the planned files and prunes the empty directories left behind.
//
// The manifest goes first, mirroring Pull: interrupted halfway, the model is
// simply gone rather than listed with blobs that no longer exist.
func (r *Removal) Apply() error {
	if err := os.Remove(r.Manifest); err != nil && !os.IsNotExist(err) {
		return err
	}
	for _, b := range r.Blobs {
		if err := os.Remove(b); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	// drop now-empty <name>/ and <namespace>/ directories, never the store root
	for d := filepath.Dir(r.Manifest); strings.HasPrefix(d, OllamaDir) && d != OllamaDir; d = filepath.Dir(d) {
		if entries, err := os.ReadDir(d); err != nil || len(entries) > 0 {
			break
		}
		if os.Remove(d) != nil {
			break
		}
	}
	return nil
}

// manifestOf locates a model's manifest file and parses it. Matching goes
// through modelName, so a name that exists under two namespaces cannot resolve
// to the wrong file.
func manifestOf(name string) (string, registryManifest, error) {
	root := filepath.Join(OllamaDir, "manifests")

	var found string
	filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || found != "" {
			return nil
		}
		if modelName(root, p) == name {
			found = p
		}
		return nil
	})
	if found == "" {
		return "", registryManifest{}, fmt.Errorf("no manifest on disk for %s", name)
	}

	data, err := os.ReadFile(found)
	if err != nil {
		return "", registryManifest{}, err
	}
	var m registryManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return "", registryManifest{}, fmt.Errorf("parsing %s: %w", found, err)
	}
	return found, m, nil
}
