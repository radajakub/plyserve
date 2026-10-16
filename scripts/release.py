#!/usr/bin/env python3
# Cut a release: bump the version, tag it, and publish a GitHub Release.
# plyserve is not on PyPI: it runs from a clone, so the tag is the release (`git checkout <tag>` on the cluster).
#
# Usage: make release VERSION=0.2.0  (or: uv run python scripts/release.py 0.2.0 [--dry-run])

import argparse
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

BRANCH = "master"
ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"
UNRELEASED = "## [Unreleased]"
VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?$")
TRACKED_FILES = ["pyproject.toml", "CHANGELOG.md", "uv.lock"]


class ReleaseError(Exception):
    pass


def read(*args: str) -> str:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise ReleaseError(f"{' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def execute(*args: str, dry_run: bool) -> None:
    print(f"$ {' '.join(args)}")
    if dry_run:
        return
    if subprocess.run(args, cwd=ROOT).returncode != 0:
        raise ReleaseError(f"{' '.join(args)} failed")


def check_tools() -> None:
    for tool in ("git", "uv", "gh"):
        if shutil.which(tool) is None:
            raise ReleaseError(f"{tool} is not installed")


def check_repo(tag: str) -> None:
    if read("git", "rev-parse", "--abbrev-ref", "HEAD") != BRANCH:
        raise ReleaseError(f"not on {BRANCH}")
    if read("git", "status", "--porcelain"):
        raise ReleaseError("working tree is dirty; commit or stash first")
    read("git", "fetch", "--quiet", "origin", BRANCH, "--tags")
    if read("git", "rev-parse", "HEAD") != read("git", "rev-parse", f"origin/{BRANCH}"):
        raise ReleaseError(f"{BRANCH} is not in sync with origin/{BRANCH}")
    if tag in read("git", "tag", "--list").splitlines():
        raise ReleaseError(f"tag {tag} already exists")


# Everything between the Unreleased heading and the previous release's heading.
def unreleased_notes() -> str:
    text = CHANGELOG.read_text()
    if UNRELEASED not in text:
        raise ReleaseError(f"{CHANGELOG.name} has no '{UNRELEASED}' section")
    body = text.split(UNRELEASED, 1)[1]
    return body.split("\n## [", 1)[0].strip()


def open_changelog_section(version: str, dry_run: bool) -> None:
    heading = f"## [{version}] - {date.today().isoformat()}"
    print(f"# {CHANGELOG.name}: insert '{heading}'")
    if dry_run:
        return
    text = CHANGELOG.read_text()
    CHANGELOG.write_text(text.replace(UNRELEASED, f"{UNRELEASED}\n\n{heading}", 1))


def commit_and_tag(tag: str, dry_run: bool) -> None:
    files = [name for name in TRACKED_FILES if (ROOT / name).exists()]
    execute("git", "add", *files, dry_run=dry_run)
    execute("git", "commit", "-m", f"[RELEASE] {tag}", dry_run=dry_run)
    execute("git", "tag", "-a", tag, "-m", f"[RELEASE] {tag}", dry_run=dry_run)
    execute("git", "push", "origin", BRANCH, tag, dry_run=dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tag a release and publish it as a GitHub Release.")
    parser.add_argument("version", help="version to release, e.g. 0.2.0")
    parser.add_argument("--dry-run", action="store_true", help="print every step without changing anything")
    args = parser.parse_args()

    version: str = args.version
    dry_run: bool = args.dry_run
    # tags are the bare version; only the Release title carries the `v`
    tag = version

    try:
        if not VERSION_PATTERN.match(version):
            raise ReleaseError(f"'{version}' is not a valid version (expected X.Y.Z)")
        check_tools()
        check_repo(tag)

        notes = unreleased_notes()
        if not notes:
            print(f"warning: {CHANGELOG.name} has nothing under {UNRELEASED}", file=sys.stderr)
            notes = f"Release {version}"

        print(f"Releasing {read('uv', 'version', '--short')} -> {version}")
        execute("uv", "version", version, dry_run=dry_run)
        open_changelog_section(version, dry_run=dry_run)
        commit_and_tag(tag, dry_run=dry_run)
        execute("gh", "release", "create", tag, "--title", f"v{version}", "--notes", notes, dry_run=dry_run)
    except ReleaseError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Released {version}." if not dry_run else "Dry run complete; nothing changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
