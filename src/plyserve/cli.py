from __future__ import annotations

import argparse
from pathlib import Path

from plyserve.download import download
from plyserve.endpoint import find_endpoint, shell_exports
from plyserve.manifest import DEFAULT_MANIFEST, ModelSpec, load_manifest
from plyserve.serve import serve


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


def _serve(args: argparse.Namespace) -> None:
    # argparse keeps the `--` that separates the client command on some Python versions
    command = args.client[1:] if args.client[:1] == ["--"] else args.client
    raise SystemExit(serve(_model(args), port=args.port, timeout=args.timeout, command=command, expose=args.expose))


def _endpoint(args: argparse.Namespace) -> None:
    print(shell_exports(find_endpoint(_model(args).name, wait=args.wait, timeout=args.timeout)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="plyserve")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="models.toml to read (default: the one in this repository)")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("models", help="list the models in the manifest").set_defaults(run=_models)

    download_parser = commands.add_parser("download", help="fetch a model's weights at its pinned revision into $PLYSERVE_HOME/hf")
    download_parser.add_argument("model", help="served name from the manifest")
    download_parser.set_defaults(run=_download)

    serve_parser = commands.add_parser("serve", help="serve a model, optionally running a client command against it")
    serve_parser.add_argument("model", help="served name from the manifest")
    serve_parser.add_argument(
        "--expose", action="store_true", help="listen for other nodes too and publish the endpoint in $PLYSERVE_HOME/endpoints (see PLYSERVE_EXPOSE_* in .env.example)"
    )
    serve_parser.add_argument("--port", type=int, help="port for vLLM (default: a free one)")
    serve_parser.add_argument("--timeout", type=float, default=1800.0, help="seconds to wait for vLLM to load (default: 1800)")
    serve_parser.add_argument(
        "client",
        nargs="*",
        metavar="command",
        help="after --: the client to run with VLLM_BASE_URL, VLLM_API_KEY and VLLM_MODEL set; vLLM stops when it exits. Without it, serve until the job ends",
    )
    serve_parser.set_defaults(run=_serve)

    endpoint_parser = commands.add_parser("endpoint", help='print exports for a server started with --expose: eval "$(plyserve endpoint <model>)"')
    endpoint_parser.add_argument("model", help="served name from the manifest")
    endpoint_parser.add_argument("--wait", action="store_true", help="wait until the server answers, e.g. while its job is queued or loading")
    endpoint_parser.add_argument("--timeout", type=float, help="with --wait: give up after this many seconds (default: never)")
    endpoint_parser.set_defaults(run=_endpoint)

    args = parser.parse_args(argv)
    try:
        args.run(args)
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        # setup mistakes (no PLYSERVE_HOME, no runtime, a bad manifest): the message says what to do
        raise SystemExit(f"plyserve: {error}") from error
