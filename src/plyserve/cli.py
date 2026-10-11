from __future__ import annotations

import argparse
from pathlib import Path

from plyserve.manifest import DEFAULT_MANIFEST, load_manifest


def _models(args: argparse.Namespace) -> None:
    for spec in load_manifest(args.manifest).values():
        print(f"{spec.name}\t{spec.repo}@{spec.revision[:12]}\ttunnel {spec.tunnel_port}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="plyserve")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="models.toml to read (default: the one in this repository)")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("models", help="list the models in the manifest").set_defaults(run=_models)

    args = parser.parse_args(argv)
    args.run(args)
