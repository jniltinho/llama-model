# Releasing llama-model

This skill guides the complete process for creating version tags and GitHub Releases for `llama-model`.

---

## Before You Start

```bash
LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "none")
echo "Last tag: $LAST_TAG"

gh auth status          # must have repo scope
git status              # must be clean
git fetch origin && git checkout main && git pull --ff-only
```

---

## Versioning Scheme

Semver with `v` prefix: `v0.1.0`, `v0.2.0`, `v0.2.1`, …
Patch for fixes, minor for new commands/flags.

```bash
LAST=$(git describe --tags --abbrev=0)
NEXT="v0.$(( $(echo ${LAST#v} | cut -d. -f2) + 1 )).0"   # minor bump; adjust as needed
echo "Next version: $NEXT"
```

---

## Recommended Process

### 1. Review changes since last release

```bash
git log $LAST_TAG..HEAD --oneline
```

### 2. Categorize commits

Sections in this order; omit empty ones:

| Section | Prefix(es) |
|---------|-----------|
| **✨ New Features** | `feat:` |
| **🔧 Improvements** | `fix:`, `perf:`, `refactor:`, `ci:` |
| **🧹 Cleanup** | `chore:`, `cleanup:` |
| **📚 Documentation** | `docs:` |

### 3. Tag and push (triggers CI)

```bash
git tag -a $NEXT -m "Release $NEXT"
git push origin $NEXT
```

The `release.yml` workflow will:

1. Run `make test`
2. Run `make release-cross` → linux/amd64 tarball
3. Create the GitHub Release with auto-generated notes and attach the `.tar.gz`

### 4. Monitor CI and polish notes

```bash
gh run list --limit 3
gh release view $NEXT          # wait until assets appear

cat > /tmp/release-notes-$NEXT.md << 'EOF'
# Release vX.Y.Z

## ✨ New Features
- ...

**Full Changelog**: https://github.com/jniltinho/llama-model/compare/vPREV...vNEXT
EOF

gh release edit $NEXT --title "$NEXT" --notes-file /tmp/release-notes-$NEXT.md
```

### 5. Verify

```bash
gh release view $NEXT
```

- Title matches `$NEXT`
- Artifact: `llama-model_X.Y.Z_linux_amd64.tar.gz`
- Full Changelog link correct

---

## Workflow Capabilities (`release.yml`)

| Artifact | Status |
|----------|--------|
| `linux/amd64` tarball | ✅ |
| other arches/OS | ❌ intentionally — the tool drives systemd and nvidia-smi on amd64 hosts |
| `.deb`/`.rpm` | ❌ not yet; add a `nfpm` step if needed |
