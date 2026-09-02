package llama

// Renames the model inside the OpenCode and Hermes configs.
//
// These are surgical text edits rather than parse+dump: the files carry
// comments (opencode.jsonc) and hand-written structure (config.yaml) that a
// round-trip through a marshaller would destroy.

import (
	"fmt"
	"io"
	"os"
	"os/user"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"unicode"
)

// userHome resolves the invoking user's home even under sudo.
func userHome() string {
	if u := os.Getenv("SUDO_USER"); u != "" {
		if usr, err := user.Lookup(u); err == nil {
			return usr.HomeDir
		}
	}
	h, _ := os.UserHomeDir()
	return h
}

// saveFile writes txt to path after keeping a .bak owned by the original user.
func saveFile(path, txt string) error {
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	src, err := os.Open(path)
	if err != nil {
		return err
	}
	bak, err := os.OpenFile(path+".bak", os.O_WRONLY|os.O_CREATE|os.O_TRUNC, info.Mode())
	if err != nil {
		src.Close()
		return err
	}
	_, err = io.Copy(bak, src)
	src.Close()
	if cerr := bak.Close(); err == nil {
		err = cerr // a backup that failed to flush is not a backup
	}
	if err != nil {
		return fmt.Errorf("writing %s.bak: %w", path, err)
	}
	if st, ok := info.Sys().(*syscall.Stat_t); ok {
		os.Chown(path+".bak", int(st.Uid), int(st.Gid))
	}
	return os.WriteFile(path, []byte(txt), info.Mode())
}

// jsonBlock returns the bounds of the `"key": { ... }` object, counting braces
// only outside of strings and comments. The file is .jsonc: a brace inside a
// // or /* */ comment would otherwise throw off the depth and the edit would
// splice the file at the wrong offset.
func jsonBlock(txt, key string) (int, int, bool) {
	loc := regexp.MustCompile(`"` + regexp.QuoteMeta(key) + `"\s*:\s*\{`).FindStringIndex(txt)
	if loc == nil {
		return 0, 0, false
	}
	start := loc[1] - 1
	depth, inStr, esc := 0, false, false
	lineCmt, blockCmt := false, false

	for i := start; i < len(txt); i++ {
		c := txt[i]
		var next byte
		if i+1 < len(txt) {
			next = txt[i+1]
		}

		switch {
		case lineCmt:
			if c == '\n' {
				lineCmt = false
			}
		case blockCmt:
			if c == '*' && next == '/' {
				blockCmt = false
				i++
			}
		case inStr && esc:
			esc = false
		case inStr && c == '\\':
			esc = true
		case inStr:
			if c == '"' {
				inStr = false
			}
		case c == '/' && next == '/':
			lineCmt = true
			i++
		case c == '/' && next == '*':
			blockCmt = true
			i++
		case c == '"':
			inStr = true
		case c == '{':
			depth++
		case c == '}':
			if depth--; depth == 0 {
				return start, i + 1, true
			}
		}
	}
	return 0, 0, false
}

// titleCase mirrors Python's str.title(): "qwen3.8 27b" -> "Qwen3.8 27B".
func titleCase(s string) string {
	out := []rune(strings.ToLower(s))
	upNext := true
	for i, r := range out {
		if upNext && unicode.IsLetter(r) {
			out[i] = unicode.ToUpper(r)
		}
		upNext = !unicode.IsLetter(r)
	}
	return string(out)
}

// replaceValue swaps the text between the end of capture group 1 and the end of
// the match. Avoids ReplaceAll's $-expansion in the replacement string.
func replaceValue(re *regexp.Regexp, txt, val string) string {
	loc := re.FindStringSubmatchIndex(txt)
	if loc == nil {
		return txt
	}
	return txt[:loc[3]] + val + txt[loc[1]:]
}

var (
	reJSONName = regexp.MustCompile(`("name"\s*:\s*)"[^"]*"`)
	reJSONCtx  = regexp.MustCompile(`("context"\s*:\s*)\d+`)
	// "tools" is not in OpenCode's schema and is silently ignored: the key is
	// "tool_call". Fix it in passing so the mistake does not survive a switch.
	reJSONTools = regexp.MustCompile(`"tools"(\s*:)`)
)

// setJSONKey sets `"key": value` inside a model block, replacing the existing
// entry or inserting one after the opening brace, matching its indentation.
func setJSONKey(blk, key, value string) string {
	re := regexp.MustCompile(`"` + regexp.QuoteMeta(key) + `"\s*:\s*(?:true|false|null|\{[^{}]*\}|\[[^\][]*\]|"[^"]*"|[\d.]+)`)
	if re.MatchString(blk) {
		return re.ReplaceAllLiteralString(blk, `"`+key+`": `+value)
	}

	open := strings.Index(blk, "{")
	if open < 0 {
		return blk
	}
	indent := " " // fall back to something harmless on a one-line block
	if nl := strings.Index(blk[open:], "\n"); nl >= 0 {
		rest := blk[open+nl+1:]
		indent = "\n" + rest[:len(rest)-len(strings.TrimLeft(rest, " \t"))]
	}
	return blk[:open+1] + indent + `"` + key + `": ` + value + "," + blk[open+1:]
}

// patchOpencode renames the model key inside the llamacpp provider and updates
// its display name, context limit and vision capability. Other providers are
// left untouched.
func patchOpencode(path, old, new string, ctx int, vision bool) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return "  opencode: " + err.Error()
	}
	txt := string(data)

	s, e, ok := jsonBlock(txt, "llamacpp")
	if !ok {
		return "  opencode: no 'llamacpp' provider, skipped"
	}
	body := txt[s:e]
	if !strings.Contains(body, `"`+old+`"`) {
		return fmt.Sprintf("  opencode: %q not found in the llamacpp block — fix it by hand", old)
	}
	body = strings.Replace(body, `"`+old+`"`, `"`+new+`"`, 1) // the model key

	if bs, be, ok := jsonBlock(body, new); ok { // inside the model block
		blk := body[bs:be]
		blk = replaceValue(reJSONName, blk, strconv.Quote(titleCase(strings.ReplaceAll(new, ":", " "))+" (V100)"))
		blk = replaceValue(reJSONCtx, blk, strconv.Itoa(ctx))
		blk = reJSONTools.ReplaceAllString(blk, `"tool_call"$1`)

		// Without both of these OpenCode never sends the attachment, even
		// though the server would accept it.
		if vision {
			blk = setJSONKey(blk, "attachment", "true")
			blk = setJSONKey(blk, "modalities", `{ "input": ["text", "image"], "output": ["text"] }`)
		} else {
			blk = setJSONKey(blk, "attachment", "false")
			blk = setJSONKey(blk, "modalities", `{ "input": ["text"], "output": ["text"] }`)
		}
		body = body[:bs] + blk + body[be:]
	}

	if err := saveFile(path, txt[:s]+body+txt[e:]); err != nil {
		return "  opencode: " + err.Error()
	}
	return fmt.Sprintf("  opencode: %s -> %s", old, new)
}

// patchHermes rewrites every YAML line whose scalar value is exactly the old
// model name: `model: <old>` and the `- <old>` items under custom_providers.
func patchHermes(path, old, new string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return "  hermes: " + err.Error()
	}
	lines := strings.Split(string(data), "\n")
	n := 0

	for i, line := range lines {
		body := strings.TrimSpace(line)
		indent := line[:len(line)-len(strings.TrimLeft(line, " \t"))]

		var prefix, val string
		switch {
		case strings.HasPrefix(body, "model:"):
			prefix, val = "model: ", strings.TrimSpace(body[len("model:"):])
		case strings.HasPrefix(body, "- "):
			v := body[len("- "):]
			// "- provider: llamacpp" is a mapping, not a scalar we should touch.
			// A model name like "qwen3.8:27b" has no ": " and no trailing colon.
			if strings.Contains(v, ": ") || strings.HasSuffix(v, ":") {
				continue
			}
			prefix, val = "- ", strings.TrimSpace(v)
		default:
			continue
		}

		if strings.Trim(val, `"'`) != old {
			continue
		}
		lines[i] = indent + prefix + `"` + new + `"`
		n++
	}

	if n == 0 {
		return fmt.Sprintf("  hermes: %q not found — fix it by hand", old)
	}
	if err := saveFile(path, strings.Join(lines, "\n")); err != nil {
		return "  hermes: " + err.Error()
	}
	return fmt.Sprintf("  hermes: %s -> %s (%d line%s)", old, new, n, plural(n))
}

func plural(n int) string {
	if n > 1 {
		return "s"
	}
	return ""
}

// PatchClients updates whichever client configs exist. vision says whether the
// new model has a multimodal projector.
func PatchClients(old, new string, ctx int, vision bool) {
	home := userHome()
	if p := filepath.Join(home, ".config/opencode/opencode.jsonc"); Exists(p) {
		fmt.Println(patchOpencode(p, old, new, ctx, vision))
	}
	if p := filepath.Join(home, ".hermes/config.yaml"); Exists(p) {
		fmt.Println(patchHermes(p, old, new))
	}
}

func Exists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}
