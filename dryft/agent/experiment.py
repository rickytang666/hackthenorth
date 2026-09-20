"""Snapshot engine candidates and track runs created by GitHub submissions."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from client import Dryft
from loop import report
from package import package

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("snapshot", "track", "status", "logs"))
    parser.add_argument("name")
    parser.add_argument("run_id", nargs="?")
    args = parser.parse_args()
    folder = ROOT / "experiments" / args.name
    if folder.parent != ROOT / "experiments" or args.name in (".", ".."):
        parser.error("name must be a single directory name")
    if args.action == "snapshot":
        archive = package(ROOT / "engine")
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "engine.tar.gz").write_bytes(archive)
        record = {"name": args.name, "sha256": hashlib.sha256(archive).hexdigest()}
        (folder / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
        return
    if args.action == "track":
        if not args.run_id:
            parser.error("track requires a run_id")
        path = folder / "run.json"
        record = json.loads(path.read_text()) if path.exists() else {"name": args.name}
        record["run_id"] = args.run_id
        path.write_text(json.dumps(record, indent=2) + "\n")
        return
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
    client = Dryft(base_url=os.environ.get("DRYFT_API", "https://htn.dryft.ai"))
    record = json.loads((folder / "run.json").read_text())
    if args.action == "logs":
        logs = client.logs(record["run_id"])
        (folder / "logs.json").write_text(json.dumps(logs, indent=2) + "\n")
        print(json.dumps(logs, indent=2))
    else:
        detail = client.run(record["run_id"])
        (folder / "result.json").write_text(json.dumps(detail, indent=2) + "\n")
        report(detail)


if __name__ == "__main__":
    main()
