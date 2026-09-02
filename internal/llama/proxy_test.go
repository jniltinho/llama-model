package llama

import (
	"encoding/json"
	"strings"
	"testing"
)

// The shape Claude Code actually sends: a system block list, and a second
// message whose role is "system" sitting after the user turn.
func TestHoistSystemMessages(t *testing.T) {
	src := `{
	  "model": "qwen3.8:27b",
	  "system": [{"type":"text","text":"You are Claude Code."}],
	  "messages": [
	    {"role":"user","content":[{"type":"text","text":"hi"}]},
	    {"role":"system","content":[{"type":"text","text":"Available agent types:"}]},
	    {"role":"assistant","content":"sure"}
	  ]}`

	out, moved := hoistSystemMessages([]byte(src))
	if moved != 1 {
		t.Fatalf("moved %d messages, want 1", moved)
	}

	var doc map[string]any
	if err := json.Unmarshal(out, &doc); err != nil {
		t.Fatal(err)
	}

	msgs := doc["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("messages left = %d, want 2", len(msgs))
	}
	for _, m := range msgs {
		if m.(map[string]any)["role"] == "system" {
			t.Error("a system message survived in the array")
		}
	}

	sys := doc["system"].([]any)
	if len(sys) != 2 {
		t.Fatalf("system blocks = %d, want 2", len(sys))
	}
	if got := sys[1].(map[string]any)["text"]; got != "Available agent types:" {
		t.Errorf("lifted block = %v", got)
	}
	// order matters: the original instructions must still come first
	if got := sys[0].(map[string]any)["text"]; got != "You are Claude Code." {
		t.Errorf("system order changed: %v", got)
	}
}

func TestHoistWithStringSystem(t *testing.T) {
	src := `{"system":"be brief","messages":[
	   {"role":"system","content":"extra rules"},
	   {"role":"user","content":"hi"}]}`

	out, moved := hoistSystemMessages([]byte(src))
	if moved != 1 {
		t.Fatalf("moved %d, want 1", moved)
	}
	var doc map[string]any
	json.Unmarshal(out, &doc)
	sys := doc["system"].([]any)
	if len(sys) != 2 || sys[0].(map[string]any)["text"] != "be brief" {
		t.Errorf("string system not normalised: %v", sys)
	}
}

// Requests without a stray system message must come through byte-identical:
// no reserialisation surprises for OpenCode or plain curl.
func TestHoistLeavesCleanRequestsAlone(t *testing.T) {
	src := `{"system":"x","messages":[{"role":"user","content":"hi"}],"stream":true}`
	out, moved := hoistSystemMessages([]byte(src))
	if moved != 0 || string(out) != src {
		t.Errorf("clean request was rewritten: %s", out)
	}

	// garbage in, same garbage out — never break a request we do not understand
	bad := []byte("not json")
	if out, _ := hoistSystemMessages(bad); string(out) != "not json" {
		t.Errorf("non-JSON body was mangled: %s", out)
	}
	noMsgs := []byte(`{"prompt":"hi"}`)
	if out, _ := hoistSystemMessages(noMsgs); string(out) != string(noMsgs) {
		t.Errorf("body without messages was rewritten: %s", out)
	}
}

func TestContentBlocks(t *testing.T) {
	if b := contentBlocks(nil); b != nil {
		t.Error("nil should produce no blocks")
	}
	if b := contentBlocks(""); b != nil {
		t.Error("empty string should produce no blocks")
	}
	b := contentBlocks("hello")
	if len(b) != 1 || b[0].(map[string]any)["text"] != "hello" {
		t.Errorf("string not wrapped: %v", b)
	}
	if b := contentBlocks([]any{1, 2}); len(b) != 2 {
		t.Error("list should pass through")
	}
	if b := contentBlocks(42); b != nil {
		t.Errorf("unexpected type should be dropped, got %v", b)
	}
}

func TestProxyRejectsBadUpstream(t *testing.T) {
	if err := Proxy("127.0.0.1:0", "://nope", false); err == nil ||
		!strings.Contains(err.Error(), "upstream") {
		t.Errorf("bad upstream not reported: %v", err)
	}
}
