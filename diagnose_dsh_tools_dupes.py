"""Diagnose the `Cannot read properties of undefined (reading 'prepare')` crash
in a DeepSeek Harness (DSH) checkout.

Background
----------
DSH's agent loop reads a scheduler object off the tool runtime service:

    // packages/core/agent-loop/src/tool-calls.ts (around line 170)
    const prepared = await ctx.tools[TOOL_RUNTIME_SCHEDULER].prepare(call.exec)

`TOOL_RUNTIME_SCHEDULER` is a `unique symbol` created with `Symbol(...)` in
`packages/core/tools/src/index.ts`:

    export const TOOL_RUNTIME_SCHEDULER: unique symbol =
      Symbol('@deepseek-ai/dsh-tools.scheduler')

`ToolRuntime` assigns that same symbol as a class field on its instance. If the
process loads TWO different copies of `@deepseek-ai/dsh-tools`, the symbol that
`dsh-agent-loop` reads is a DIFFERENT object than the one `ToolRuntime` used to
assign the field. The lookup then returns `undefined`, and `undefined.prepare`
throws exactly the error above.

This is the classic "dual package hazard": the same package name resolved to
two physical directories inside one Node process.

Usage
-----
Run this from the DSH checkout root (the directory that contains the root
`package.json` and `pnpm-workspace.yaml`):

    python diagnose_dsh_tools_dupes.py

Optional: point it at a different root:

    python diagnose_dsh_tools_dupes.py --root C:\\Users\\mm\\deepseek-harness

What it does
------------
1. Finds every `@deepseek-ai/dsh-tools/package.json` under the checkout
   (skipping nested `node_modules` chains that are only dependencies of other
   packages when you ask it to, but by default it lists them all so you can see
   the nesting).
2. Reports how many DISTINCT physical copies exist and prints their versions.
3. Tells you exactly which directories to remove / dedupe if more than one copy
   is present.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PACKAGE_REL = Path("@deepseek-ai") / "dsh-tools" / "package.json"


def find_copies(root: Path) -> list[Path]:
    """Return every physical `@deepseek-ai/dsh-tools/package.json` under *root*."""
    copies: list[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        # `os.walk` is fine here; we intentionally do not follow symlinked dirs
        # beyond their real target because a symlink would be the *correct*
        # single copy (pnpm uses links), so we still want to record it.
        current = Path(dirpath)
        candidate = current / PACKAGE_REL
        if candidate.is_file():
            copies.append(candidate.resolve())
    # De-duplicate by real path so a symlink and its target collapse to one.
    return sorted(set(copies))


def read_version(manifest: Path) -> str:
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return str(data.get("version", "?"))
    except Exception:  # noqa: BLE001 - diagnostic tool, never crash
        return "?"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=".",
        help="DSH checkout root (default: current directory)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print(f"Scanning for {PACKAGE_REL} under: {root}")
    print()

    copies = find_copies(root)

    if not copies:
        print("No @deepseek-ai/dsh-tools/package.json found.")
        print("Are you sure this is the DSH checkout root?")
        return 2

    print(f"Found {len(copies)} physical copy/copies:\n")
    for i, manifest in enumerate(copies, 1):
        version = read_version(manifest)
        # Show the directory that CONTAINS node_modules, which is the thing to
        # fix, rather than the manifest path itself.
        pkg_dir = manifest.parent
        try:
            rel = pkg_dir.relative_to(root)
        except ValueError:
            rel = pkg_dir
        print(f"  [{i}] version={version}")
        print(f"      {rel}")
    print()

    if len(copies) == 1:
        print("OK: only one physical copy of @deepseek-ai/dsh-tools is present.")
        print("    The `prepare` crash is probably NOT caused by a duplicate here.")
        return 0

    print("PROBLEM: more than one physical copy of @deepseek-ai/dsh-tools.")
    print("This is the dual-package hazard that makes")
    print("`ctx.tools[TOOL_RUNTIME_SCHEDULER]` evaluate to `undefined` and crash")
    print("at `ctx.tools[...].prepare(...)`.")
    print()
    print("Suggested fix (run from the checkout root):")
    print("  git status")
    print("  git stash            REM if the working tree is dirty")
    print("  pnpm why @deepseek-ai/dsh-tools")
    print("  pnpm dedupe")
    print("  pnpm install")
    print("  pnpm build")
    print()
    print("Then restart DSH. If a nested copy survives dedupe, remove it and")
    print("reinstall, e.g.:")
    for manifest in copies[1:]:
        nested = manifest.parent.parent.parent  # .../@deepseek-ai/dsh-tools -> node_modules
        print(f"  rmdir /s /q \"{nested}\"")
    print("  pnpm install")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())