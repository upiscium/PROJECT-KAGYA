"""Command-line interface for the durable training worker."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

from kagya.config import load_settings
from kagya.training.worker import TrainingWorkerService, WorkerJobStatus


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    settings = load_settings()
    service = TrainingWorkerService(settings)
    try:
        if args.action == "run":
            job, created = service.submit(
                Path(args.bundle), Path(args.output), args.idempotency_key
            )
            if created or job.status == WorkerJobStatus.READY:
                try:
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "kagya.training.worker_cli",
                            "_execute",
                            "--job-id",
                            job.job_id,
                        ],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                        env=os.environ.copy(),
                    )
                except OSError as exc:
                    service.store.update(
                        job.job_id,
                        status=WorkerJobStatus.FAILED,
                        error=f"failed to start worker process: {exc}",
                    )
                    raise
            _print_job(job)
        elif args.action == "_execute":
            _print_job(service.execute(args.job_id))
        elif args.action in {"status", "inspect"}:
            _print_job(service.inspect(args.job_id))
        elif args.action == "cancel":
            _print_job(service.cancel(args.job_id))
        elif args.action == "health":
            print(json.dumps(service.health(), sort_keys=True))
        elif args.action == "cleanup":
            print(json.dumps(service.cleanup(args.retention_days), sort_keys=True))
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kagya-worker")
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("run")
    run.add_argument("--bundle", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--idempotency-key")
    for action in ("status", "cancel", "inspect", "_execute"):
        command = commands.add_parser(action)
        command.add_argument("--job-id", required=True)
    commands.add_parser("health")
    cleanup = commands.add_parser("cleanup")
    cleanup.add_argument("--retention-days", type=int, required=True)
    return parser


def _print_job(job) -> None:
    payload = asdict(job)
    payload["status"] = job.status.value
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
