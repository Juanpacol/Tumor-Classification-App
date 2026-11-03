"""``python -m src.release package <bundle>`` or ``python -m src.release fetch <version>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.errors import ModelUnavailableError
from src.release.fetch import fetch
from src.release.package import package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("package", help="archive a trained bundle with its checksum")
    pack.add_argument("bundle", type=Path, help="artifacts/model-v<version>")
    pack.add_argument("--out", type=Path, default=Path("release"))
    get = sub.add_parser("fetch", help="download a published bundle and verify it")
    get.add_argument("version")
    get.add_argument("--repo", default="Juanpacol/Tumor-Classification-App")
    get.add_argument("--dest", type=Path, default=Path("artifacts"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "package":
            archive, checksum, notes = package(args.bundle, args.out)
            tag = archive.name.removesuffix(".tar.gz")
            print(
                f"gh release create {tag} {archive} {checksum} "
                f"--title {tag} --notes-file {notes}"
            )
        else:
            print(fetch(args.version, args.repo, args.dest))
    except ModelUnavailableError as error:
        print(f"error: {error.reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
