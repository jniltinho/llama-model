BIN     := dist/llama-model
SRC     := $(shell find . -name '*.go') go.mod go.sum
VERSION ?= 0.1.0
PREFIX  ?= /usr/local
LDFLAGS := -s -w -X llama-model/cmd.version=$(VERSION)

.PHONY: all build test fmt vet lint release-cross install uninstall clean

all: build

build: $(BIN)

$(BIN): $(SRC)
	@mkdir -p dist
	CGO_ENABLED=0 go build -trimpath -ldflags "$(LDFLAGS)" -o $@ .
	@echo "  built:      $$(du -h $@ | cut -f1)"
	@if command -v upx >/dev/null; then \
	  upx -q --best --lzma $@ >/dev/null && echo "  compressed: $$(du -h $@ | cut -f1)"; \
	else \
	  echo "  upx not installed, binary left uncompressed (sudo apt install upx-ucl)"; \
	fi

test:
	go test ./...

fmt:
	gofmt -l -w .

vet:
	go vet ./...

lint: vet
	@test -z "$$(gofmt -l .)" || { echo "gofmt needed:"; gofmt -l .; exit 1; }

# linux/amd64 only: the tool drives systemd and nvidia-smi
release-cross:
	@mkdir -p dist/pkg
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags "$(LDFLAGS)" \
	  -o dist/pkg/llama-model .
	tar -czf dist/llama-model_$(VERSION)_linux_amd64.tar.gz \
	  -C dist/pkg llama-model -C $(CURDIR) LICENSE README.md
	@echo "  dist/llama-model_$(VERSION)_linux_amd64.tar.gz"
	@rm -rf dist/pkg

# needs sudo: writes to $(PREFIX)/bin
install: build
	install -m 0755 $(BIN) $(PREFIX)/bin/llama-model

uninstall:
	rm -f $(PREFIX)/bin/llama-model

clean:
	rm -rf dist
