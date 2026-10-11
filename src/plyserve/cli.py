from __future__ import annotations

import argparse
from pathlib import Path

from plyserve.download import download
from plyserve.manifest import DEFAULT_MANIFEST, ModelSpec, load_manifest


def _model(args: argparse.Namespace) -> ModelSpec:
    specs = load_manifest(args.manifest)
    if args.model not in specs:
        raise SystemExit(f"unknown model {args.model!r}; the manifest has {sorted(specs)}")
    return specs[args.model]


def _models(args: argparse.Namespace) -> None:
    for spec in load_manifest(args.manifest).values():
        print(f"{spec.name}\t{spec.repo}@{spec.revision[:12]}\ttunnel {spec.tunnel_port}")


def _download(args: argparse.Namespace) -> None:
    download(_model(args))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="plyserve")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="models.toml to read (default: the one in this repository)")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("models", help="list the models in the manifest").set_defaults(run=_models)

    download_parser = commands.add_parser("download", help="fetch a model's weights at its pinned revision into $PLYSERVE_HOME/hf")
    download_parser.add_argument("model", help="served name from the manifest")
    download_parser.set_defaults(run=_download)

    args = parser.parse_args(argv)
    try:
        args.run(args)
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        # setup mistakes (no PLYSERVE_HOME, no runtime, a bad manifest): the message says what to do
        raise SystemExit(f"plyserve: {error}") from error
