package llama

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"
)

// Meta holds the only two header fields we care about.
type Meta struct {
	Arch string
	Ctx  uint64
}

// Byte width of the fixed-size GGUF value types.
var ggufFixed = map[uint32]int64{0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

// ggufReader carries a sticky error so the parser reads like straight-line code.
type ggufReader struct {
	f   *os.File
	err error
}

func (r *ggufReader) read(v any) {
	if r.err == nil {
		r.err = binary.Read(r.f, binary.LittleEndian, v)
	}
}

func (r *ggufReader) u32() uint32 {
	var v uint32
	r.read(&v)
	return v
}

func (r *ggufReader) u64() uint64 {
	var v uint64
	r.read(&v)
	return v
}

func (r *ggufReader) skip(n int64) {
	if r.err == nil {
		_, r.err = r.f.Seek(n, io.SeekCurrent)
	}
}

func (r *ggufReader) str() string {
	n := r.u64()
	if r.err != nil || n > 1<<20 { // no sane metadata string is a megabyte
		return ""
	}
	b := make([]byte, n)
	if _, err := io.ReadFull(r.f, b); err != nil {
		r.err = err
		return ""
	}
	return string(b)
}

// skipValue advances past one metadata value of the given type.
func (r *ggufReader) skipValue(t uint32) {
	switch {
	case ggufFixed[t] != 0:
		r.skip(ggufFixed[t])
	case t == 8: // string
		r.skip(int64(r.u64()))
	case t == 9: // array
		et, n := r.u32(), r.u64()
		switch {
		case ggufFixed[et] != 0:
			r.skip(ggufFixed[et] * int64(n))
		case et == 8:
			for i := uint64(0); i < n && r.err == nil; i++ {
				r.skip(int64(r.u64()))
			}
		default:
			r.err = fmt.Errorf("unsupported array element type %d", et)
		}
	default:
		r.err = fmt.Errorf("unknown GGUF type %d", t)
	}
}

// ReadGGUF pulls architecture and context length out of a GGUF header.
// It seeks past the values instead of reading the 17GB of tensor data.
// Returns a zero Meta when the file is unreadable or malformed.
func ReadGGUF(path string) Meta {
	f, err := os.Open(path)
	if err != nil {
		return Meta{}
	}
	defer f.Close()

	magic := make([]byte, 4)
	if _, err := io.ReadFull(f, magic); err != nil || string(magic) != "GGUF" {
		return Meta{}
	}

	r := &ggufReader{f: f}
	r.skip(4) // version
	r.u64()   // tensor count
	nKV := r.u64()

	var m Meta
	for i := uint64(0); i < nKV && r.err == nil; i++ {
		key := r.str()
		t := r.u32()
		switch {
		case key == "general.architecture" && t == 8:
			m.Arch = r.str()
		case strings.HasSuffix(key, ".context_length") && t == 4:
			m.Ctx = uint64(r.u32())
		case strings.HasSuffix(key, ".context_length") && t == 10:
			m.Ctx = r.u64()
		default:
			r.skipValue(t)
		}
		if m.Arch != "" && m.Ctx != 0 {
			return m
		}
	}
	if r.err != nil {
		return Meta{}
	}
	return m
}
