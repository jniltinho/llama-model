package llama

// Talks to the Ollama registry directly, so pulling a model does not require
// the ollama daemon to be installed or running. The store layout it writes is
// exactly the one ollama uses:
//
//	<OllamaDir>/manifests/registry.ollama.ai/<namespace>/<name>/<tag>
//	<OllamaDir>/blobs/sha256-<hex>

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/user"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	registry     = "https://registry.ollama.ai"
	registryHost = "registry.ollama.ai"
	manifestMIME = "application/vnd.docker.distribution.manifest.v2+json"
)

// Ref is a parsed model reference: library/gemma4:e4b.
type Ref struct {
	Namespace string
	Name      string
	Tag       string
}

func (r Ref) String() string {
	if r.Namespace == "library" {
		return r.Name + ":" + r.Tag
	}
	return r.Namespace + "/" + r.Name + ":" + r.Tag
}

func (r Ref) path() string { return r.Namespace + "/" + r.Name }

// ParseRef fills in the defaults: namespace "library" and tag "latest".
func ParseRef(s string) Ref {
	r := Ref{Namespace: "library", Tag: "latest"}
	s = strings.TrimPrefix(s, registryHost+"/")
	if i := strings.Index(s, "/"); i >= 0 {
		r.Namespace, s = s[:i], s[i+1:]
	}
	if i := strings.LastIndex(s, ":"); i >= 0 {
		s, r.Tag = s[:i], s[i+1:]
	}
	r.Name = s
	return r
}

type layer struct {
	MediaType string `json:"mediaType"`
	Digest    string `json:"digest"`
	Size      int64  `json:"size"`
}

type registryManifest struct {
	Config layer   `json:"config"`
	Layers []layer `json:"layers"`
}

// blobs returns every blob the manifest references, config included.
func (m registryManifest) blobs() []layer {
	out := make([]layer, 0, len(m.Layers)+1)
	if m.Config.Digest != "" {
		out = append(out, m.Config)
	}
	return append(out, m.Layers...)
}

var httpClient = &http.Client{Timeout: 0} // large blobs: no overall deadline

// EnsureWritable checks we can actually create files in the store before doing
// any work, so a permission problem shows up as advice instead of a stack of
// failed writes halfway through a 17GB download.
func EnsureWritable() error {
	if err := os.MkdirAll(OllamaDir, 0o755); err != nil {
		return fmt.Errorf("%s: %w (try sudo)", OllamaDir, err)
	}
	probe, err := os.CreateTemp(OllamaDir, ".write-check-*")
	if err != nil {
		return fmt.Errorf("cannot write to %s — run with sudo", OllamaDir)
	}
	probe.Close()
	return os.Remove(probe.Name())
}

// Pull downloads a model from the Ollama registry into the local store.
func Pull(name string) error {
	if err := EnsureWritable(); err != nil {
		return err
	}
	ref := ParseRef(name)

	url := fmt.Sprintf("%s/v2/%s/manifests/%s", registry, ref.path(), ref.Tag)
	req, _ := http.NewRequest("GET", url, nil)
	req.Header.Set("Accept", manifestMIME)
	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("fetching manifest: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return fmt.Errorf("%s not found in the registry — check the tag at https://ollama.com/library/%s",
			ref, ref.Name)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("fetching manifest: %s", resp.Status)
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return err
	}

	var man registryManifest
	if err := json.Unmarshal(raw, &man); err != nil {
		return fmt.Errorf("parsing manifest: %w", err)
	}

	blobDir := filepath.Join(OllamaDir, "blobs")
	if err := os.MkdirAll(blobDir, 0o755); err != nil {
		return err
	}

	var total int64
	for _, l := range man.blobs() {
		total += l.Size
	}
	fmt.Printf("pulling %s (%s in %d blobs)\n", ref, HumanBytes(total), len(man.blobs()))

	for _, l := range man.blobs() {
		if err := fetchBlob(ref, l, blobDir); err != nil {
			return err
		}
	}

	// The manifest goes in last: a half-pulled model must not look complete.
	dir := filepath.Join(OllamaDir, "manifests", registryHost, ref.Namespace, ref.Name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	dest := filepath.Join(dir, ref.Tag)
	if err := os.WriteFile(dest, raw, 0o644); err != nil {
		return err
	}
	fixOwner(dest)
	for d := dir; strings.HasPrefix(d, OllamaDir) && d != OllamaDir; d = filepath.Dir(d) {
		fixOwner(d)
	}

	fmt.Printf("done: %s\n", ref)
	return nil
}

// fetchBlob downloads one blob, resuming a partial file when possible and
// verifying the sha256 before it counts as complete.
func fetchBlob(ref Ref, l layer, blobDir string) error {
	digest := strings.TrimPrefix(l.Digest, "sha256:")
	dest := filepath.Join(blobDir, "sha256-"+digest)
	label := shortLabel(l.MediaType, digest)

	if st, err := os.Stat(dest); err == nil && st.Size() == l.Size {
		fmt.Printf("  %-12s %10s  already present\n", label, HumanBytes(l.Size))
		return nil
	}

	partial := dest + "-partial"
	hasher := sha256.New()
	var offset int64

	if st, err := os.Stat(partial); err == nil && st.Size() > 0 && st.Size() < l.Size {
		// replay what we already have through the hasher so the final sum is valid
		f, err := os.Open(partial)
		if err == nil {
			if n, err := io.Copy(hasher, f); err == nil {
				offset = n
			} else {
				hasher.Reset()
			}
			f.Close()
		}
		if offset > 0 {
			fmt.Printf("  %-12s %10s  resuming at %s\n", label, HumanBytes(l.Size), HumanBytes(offset))
		}
	}

	url := fmt.Sprintf("%s/v2/%s/blobs/%s", registry, ref.path(), l.Digest)
	req, _ := http.NewRequest("GET", url, nil)
	if offset > 0 {
		req.Header.Set("Range", "bytes="+strconv.FormatInt(offset, 10)+"-")
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("%s: %w", label, err)
	}
	defer resp.Body.Close()

	flags := os.O_CREATE | os.O_WRONLY
	switch resp.StatusCode {
	case http.StatusPartialContent:
		flags |= os.O_APPEND
	case http.StatusOK: // server ignored Range: start over
		hasher.Reset()
		offset = 0
		flags |= os.O_TRUNC
	default:
		return fmt.Errorf("%s: %s", label, resp.Status)
	}

	f, err := os.OpenFile(partial, flags, 0o644)
	if err != nil {
		return err
	}

	bar := &progress{label: label, total: l.Size, done: offset, base: offset, start: time.Now()}
	_, err = io.Copy(io.MultiWriter(f, hasher, bar), resp.Body)
	f.Close()
	bar.finish()
	if err != nil {
		return fmt.Errorf("%s: %w (run pull again to resume)", label, err)
	}

	if got := hex.EncodeToString(hasher.Sum(nil)); got != digest {
		os.Remove(partial)
		return fmt.Errorf("%s: checksum mismatch (got %s), download discarded", label, got[:12])
	}
	if err := os.Rename(partial, dest); err != nil {
		return err
	}
	fixOwner(dest)
	return nil
}

// shortLabel names a blob by its role, falling back to the digest prefix.
func shortLabel(mediaType, digest string) string {
	if i := strings.LastIndex(mediaType, "."); i >= 0 {
		if kind := mediaType[i+1:]; kind != "" && !strings.Contains(kind, "+") {
			return kind
		}
	}
	if len(digest) > 12 { // a malformed manifest must not panic the pull
		return digest[:12]
	}
	return digest
}

// fixOwner hands files to the ollama user when we are running as root, so the
// store stays readable by both ollama and the llama-server unit.
func fixOwner(path string) {
	if os.Geteuid() != 0 {
		return
	}
	u, err := user.Lookup("ollama")
	if err != nil {
		return
	}
	uid, _ := strconv.Atoi(u.Uid)
	gid, _ := strconv.Atoi(u.Gid)
	os.Chown(path, uid, gid)
}

// progress prints a single self-overwriting status line.
type progress struct {
	label     string
	total     int64
	done      int64
	base      int64 // bytes already on disk when we started: not our throughput
	start     time.Time
	lastPrint time.Time
}

// isTTY reports whether stdout is a terminal. Redirected to a file or a pipe,
// the carriage-return updates would pile up as one line per refresh.
func isTTY() bool {
	st, err := os.Stdout.Stat()
	return err == nil && st.Mode()&os.ModeCharDevice != 0
}

func (p *progress) Write(b []byte) (int, error) {
	p.done += int64(len(b))
	if isTTY() && time.Since(p.lastPrint) > 300*time.Millisecond {
		p.print()
		p.lastPrint = time.Now()
	}
	return len(b), nil
}

func (p *progress) print() {
	pct := 0.0
	if p.total > 0 {
		pct = float64(p.done) / float64(p.total) * 100
	}
	speed := float64(p.done-p.base) / time.Since(p.start).Seconds()
	fmt.Printf("\r  %-12s %10s  %5.1f%%  %s/s     ",
		p.label, HumanBytes(p.total), pct, HumanBytes(int64(speed)))
}

func (p *progress) finish() {
	p.print()
	fmt.Println()
}

func HumanBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for v := n / unit; v >= unit; v /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(n)/float64(div), "KMGT"[exp])
}
