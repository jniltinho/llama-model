package llama

// A thin proxy in front of llama-server's /v1/messages endpoint, for clients
// that speak the Anthropic protocol (Claude Code).
//
// Claude Code sends part of its instructions as a message with role "system"
// in the middle of the conversation. Qwen's chat template refuses that with
// "System message must be at the beginning", so the whole request 500s. Here we
// lift those messages into the top-level system field, where every template
// expects them, and pass everything else through untouched — streaming
// included, since ReverseProxy forwards the body as it arrives.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"
)

// Proxy serves on listen and forwards to upstream, fixing up Anthropic
// requests on the way.
func Proxy(listen, upstream string, verbose bool) error {
	target, err := url.Parse(upstream)
	if err != nil {
		return fmt.Errorf("upstream %q: %w", upstream, err)
	}

	rp := &httputil.ReverseProxy{
		Director: func(r *http.Request) {
			r.URL.Scheme, r.URL.Host, r.Host = target.Scheme, target.Host, target.Host
			if r.Body == nil {
				return
			}
			body, err := io.ReadAll(r.Body)
			r.Body.Close()
			if err != nil {
				return
			}
			fixed, moved := hoistSystemMessages(body)
			if verbose && moved > 0 {
				log.Printf("%s: moved %d system message(s) into the system field", r.URL.Path, moved)
			}
			r.Body = io.NopCloser(bytes.NewReader(fixed))
			r.ContentLength = int64(len(fixed))
			r.Header.Set("Content-Length", strconv.Itoa(len(fixed)))
		},
		ErrorHandler: func(w http.ResponseWriter, _ *http.Request, err error) {
			log.Printf("upstream error: %v", err)
			http.Error(w, `{"error":{"type":"api_error","message":"upstream unreachable"}}`,
				http.StatusBadGateway)
		},
	}

	log.Printf("listening on %s -> %s", listen, upstream)
	return http.ListenAndServe(listen, rp)
}

// hoistSystemMessages moves any role:"system" entry out of messages and appends
// its content to the top-level system field. Returns the body unchanged if
// there is nothing to move or it is not the JSON we expect.
func hoistSystemMessages(body []byte) ([]byte, int) {
	var doc map[string]any
	if json.Unmarshal(body, &doc) != nil {
		return body, 0
	}
	msgs, ok := doc["messages"].([]any)
	if !ok {
		return body, 0
	}

	kept := make([]any, 0, len(msgs))
	var lifted []any
	for _, m := range msgs {
		msg, ok := m.(map[string]any)
		if !ok || msg["role"] != "system" {
			kept = append(kept, m)
			continue
		}
		lifted = append(lifted, contentBlocks(msg["content"])...)
	}
	if len(lifted) == 0 {
		return body, 0
	}

	doc["system"] = append(contentBlocks(doc["system"]), lifted...)
	doc["messages"] = kept

	out, err := json.Marshal(doc)
	if err != nil {
		return body, 0 // never send a broken body: let the original through
	}
	return out, len(msgs) - len(kept)
}

// contentBlocks normalises Anthropic content into a block list: it is either a
// plain string or already a list of blocks.
func contentBlocks(v any) []any {
	switch c := v.(type) {
	case nil:
		return nil
	case string:
		if c == "" {
			return nil
		}
		return []any{map[string]any{"type": "text", "text": c}}
	case []any:
		return c
	default:
		return nil
	}
}
