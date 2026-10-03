#!/usr/bin/env python3
"""Validate every ```mermaid block in the given Markdown file(s) actually renders.

Why: mermaid parse errors (reserved-keyword class names like `:::graph`, a bare `&` in a
label, an unquoted special char) render as an ugly red "Parse error" box in the viewer but
are invisible in the raw Markdown. Run this before committing any doc that adds/edits a
mermaid diagram (see docs/doc-standards.md §6 and the Documentation Conventions Rule).

How it validates each block, in order of preference:
  1. local mermaid CLI `mmdc` (or `npx @mermaid-js/mermaid-cli`), if installed — fully offline;
  2. else kroki.io (https://kroki.io/mermaid/svg/<deflate+base64url>) — needs network. The
     diagram text is non-sensitive doc content; if you must stay offline, install mmdc.

Usage:
  python scripts/check_mermaid.py docs/papers/**/some-paper.md [more.md ...]
  python scripts/check_mermaid.py $(git diff --cached --name-only -- '*.md')   # pre-commit

Exit code 0 = all blocks valid; 1 = at least one block failed (or could not be checked).
"""
import base64
import glob
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zlib

MERMAID_RE = re.compile(r"```mermaid\n(.*?)```", re.S)


def _find_mmdc():
    if shutil.which("mmdc"):
        return ["mmdc"]
    if shutil.which("npx"):
        return ["npx", "-y", "@mermaid-js/mermaid-cli"]
    return None


def _check_mmdc(cmd, diagram):
    """Return (ok, message) using the local mermaid CLI."""
    with tempfile.NamedTemporaryFile("w", suffix=".mmd", delete=False) as fin:
        fin.write(diagram)
        src = fin.name
    out = src + ".svg"
    try:
        r = subprocess.run(cmd + ["-i", src, "-o", out],
                           capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            return True, "mmdc OK"
        return False, (r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout) else "mmdc failed"
    except Exception as e:  # noqa: BLE001
        return False, f"mmdc error: {e}"


def _check_kroki(diagram):
    """Return (ok, message) using kroki.io. HTTP 200 = valid SVG; 400 = parse error."""
    enc = base64.urlsafe_b64encode(zlib.compress(diagram.encode(), 9)).decode()
    url = "https://kroki.io/mermaid/svg/" + enc
    # kroki sits behind Cloudflare, which 403s (error 1010) the default urllib UA — send a browser UA.
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "image/svg+xml,*/*",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read(200).decode("utf-8", "replace")
            return (resp.status == 200 and "<svg" in body), f"kroki HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace").strip().replace("\n", " ")[:200]
        return False, f"kroki HTTP {e.code}: {detail}"
    except Exception as e:  # noqa: BLE001
        return None, f"kroki unreachable: {e}"


def main(paths):
    files = [f for p in paths for f in glob.glob(p, recursive=True)]
    if not files:
        print("check_mermaid: no files to check")
        return 0
    mmdc = _find_mmdc()
    engine = "mmdc" if mmdc else "kroki.io"
    total = fails = uncheckable = 0
    for f in files:
        try:
            blocks = MERMAID_RE.findall(open(f, encoding="utf-8").read())
        except OSError as e:
            print(f"SKIP {f}: {e}")
            continue
        for i, b in enumerate(blocks, 1):
            total += 1
            ok, msg = _check_mmdc(mmdc, b) if mmdc else _check_kroki(b)
            if ok is True:
                print(f"PASS {f} [mermaid #{i}] ({engine})")
            elif ok is None:
                uncheckable += 1
                print(f"WARN {f} [mermaid #{i}] could not check — {msg}")
            else:
                fails += 1
                print(f"FAIL {f} [mermaid #{i}] — {msg}")
    print(f"\nchecked {total} mermaid block(s) via {engine}: "
          f"{total - fails - uncheckable} ok, {fails} failed, {uncheckable} unchecked")
    return 1 if (fails or uncheckable) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["docs/**/*.md"]))
