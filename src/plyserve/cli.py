from __future__ import annotations

import argparse
from pathlib import Path

from plyserve.download import download
from plyserve.endpoint import find_endpoint, shell_exports
from plyserve.manifest import ModelSpec, default_manifest, load_manifest
from plyserve.remote import remote
from plyserve.serve import serve


def _model(args: argparse.Namespace) -> ModelSpec:
    specs = load_manifest(args.manifest)
    if args.model not in specs:
        raise SystemExit(f"unknown model {args.model!r}; the manifest has {sorted(specs)}")
    return specs[args.model]


def _models(args: argparse.Namespace) -> None:
    for spec in load_manifest(args.manifest).values():
        print(f"{spec.name}\t{spec.backend}\t{spec.repo}@{spec.revision[:12]}\ttunnel {spec.tunnel_port}")


def _download(args: argparse.Namespace) -> None:
    download(_model(args))


def _command(args: argparse.Namespace) -> list[str]:
    # argparse keeps the `--` that separates the client command on some Python versions
    return args.client[1:] if args.client[:1] == ["--"] else args.client


def _serve(args: argparse.Namespace) -> None:
    command = _command(args)
    raise SystemExit(serve(_model(args), port=args.port, timeout=args.timeout, command=command, expose=args.expose, tunnel=args.tunnel, monitor=args.monitor))


def _remote(args: argparse.Namespace) -> None:
    command = _command(args)
    if not command:
        raise SystemExit("usage: plyserve remote [options] <model> -- <command...>")
    raise SystemExit(remote(_model(args), command, length=args.length, monitor=args.monitor, manifest=args.remote_manifest, timeout=args.timeout, notify=args.notify))


def _endpoint(args: argparse.Namespace) -> None:
    spec = _model(args)
    print(shell_exports(find_endpoint(spec.name, wait=args.wait, timeout=args.timeout), spec.backend))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="plyserve")
    parser.add_argument("--manifest", type=Path, help="models.toml to read (default: PLYSERVE_MANIFEST, else the one in this repository)")
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
    serve_parser.add_argument(
        "--tunnel", action="store_true", help="forward the model's tunnel_port on another machine to the server over SSH (see PLYSERVE_TUNNEL_* in .env.example)"
    )
    serve_parser.add_argument("--monitor", action="store_true", help="sample the GPU (nvidia-smi, and DCGM if installed) and the CPU (top) into the run folder while serving")
    serve_parser.add_argument("--port", type=int, help="port for the server (default: a free one)")
    serve_parser.add_argument("--timeout", type=float, default=1800.0, help="seconds to wait for the model to load (default: 1800)")
    serve_parser.add_argument(
        "client",
        nargs="*",
        metavar="command",
        help="after --: the client to run with VLLM_BASE_URL, VLLM_API_KEY and VLLM_MODEL set; the server stops when it exits. Without it, serve until the job ends",
    )
    serve_parser.set_defaults(run=_serve)

    remote_parser = commands.add_parser(
        "remote", help="on the client machine: run `serve --tunnel` as a job on the cluster, run a command against it, then cancel the job (see PLYSERVE_REMOTE in .env.example)"
    )
    remote_parser.add_argument("model", help="served name from the manifest; its tunnel_port must be the same in the cluster's manifest")
    remote_parser.add_argument("--length", choices=("short", "long"), default="short", help="the job's partition and time limit, as LENGTH of `make serve-job` (default: short)")
    remote_parser.add_argument("--monitor", action="store_true", help="pass --monitor to serve: GPU and CPU samples in the run folder on the cluster")
    remote_parser.add_argument("--remote-manifest", help="models.toml on the cluster for the job, as PLYSERVE_MANIFEST there (default: the cluster's own setting)")
    remote_parser.add_argument("--timeout", type=float, default=3600.0, help="seconds to wait for the server once the job runs (default: 3600)")
    remote_parser.add_argument(
        "--notify",
        action="store_true",
        help="push an ntfy notification when the job starts, when the server answers, and when the command ends or fails (needs NTFY_URL and NTFY_TOPIC)",
    )
    remote_parser.add_argument("client", nargs="*", metavar="command", help="after --: the command to run with VLLM_BASE_URL, VLLM_API_KEY and VLLM_MODEL set")
    remote_parser.set_defaults(run=_remote)

    endpoint_parser = commands.add_parser("endpoint", help='print exports for a server started with --expose: eval "$(plyserve endpoint <model>)"')
    endpoint_parser.add_argument("model", help="served name from the manifest")
    endpoint_parser.add_argument("--wait", action="store_true", help="wait until the server answers, e.g. while its job is queued or loading")
    endpoint_parser.add_argument("--timeout", type=float, help="with --wait: give up after this many seconds (default: never)")
    endpoint_parser.set_defaults(run=_endpoint)

    args = parser.parse_args(argv)
    try:
        args.manifest = args.manifest or default_manifest()
        args.run(args)
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        # setup mistakes (no PLYSERVE_HOME, no runtime, a bad manifest): the message says what to do
        raise SystemExit(f"plyserve: {error}") from error
