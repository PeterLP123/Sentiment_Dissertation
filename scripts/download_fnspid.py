#!/usr/bin/env python3
"""Download the frozen FNSPID release into lossless, hash-verified local archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DATASET_ID = "Zihan1004/FNSPID"
DATASET_REVISION = "bf9189c41527198897d1af3e17b1a0095279fc45"
GITHUB_REVISION = "4054842ec476953b30ee874d4b7e8eea786a21fa"
FILES = (
    {
        "path": "Stock_news/All_external.csv",
        "size": 5_731_397_037,
        "sha256": "5d4c018036bd82ca821da71b7a9c0c7db3289642e0fc6f897ea69f4a0c5135c3",
        "compress": True,
    },
    {
        "path": "Stock_news/nasdaq_exteral_data.csv",
        "size": 23_232_979_597,
        "sha256": "1a7a3eb8e6b97ec19f286f2cfca3371542bddb272ab1eb8f36e33ad98fa5c4da",
        "compress": True,
    },
    {
        "path": "Stock_price/full_history.zip",
        "size": 589_525_596,
        "sha256": "03da4fce7ebea90d5715ba3501773d410ae663b617027b338ef000a9955dab91",
        "compress": False,
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(entry: dict[str, Any], output_root: Path) -> dict[str, Any]:
    relative = str(entry["path"])
    encoded = urllib.parse.quote(relative, safe="/")
    url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{encoded}"
    destination = output_root / f"{relative}.zst" if entry["compress"] else output_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing download: {destination}")
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite incomplete download: {temporary}")

    request = urllib.request.Request(url, headers={"User-Agent": "Sentiment-Dissertation-FNSPID-Audit/1.0"})
    upstream_hash = hashlib.sha256()
    downloaded = 0
    next_progress = 1024**3
    compressor: subprocess.Popen[bytes] | None = None
    compressor_stdin: Any = None
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            if entry["compress"]:
                zstd = shutil.which("zstd")
                if zstd is None:
                    raise RuntimeError("zstd executable is required for lossless streaming compression")
                compressor = subprocess.Popen(
                    [zstd, "-q", "-T0", "-3", "-c"],
                    stdin=subprocess.PIPE,
                    stdout=output,
                )
                if compressor.stdin is None:
                    raise RuntimeError("failed to open zstd input stream")
                compressor_stdin = compressor.stdin
            while chunk := response.read(8 * 1024 * 1024):
                upstream_hash.update(chunk)
                downloaded += len(chunk)
                if compressor is None:
                    output.write(chunk)
                else:
                    compressor_stdin.write(chunk)
                if downloaded >= next_progress:
                    print(
                        f"{relative}: {downloaded / 1024**3:.1f} GiB / {int(entry['size']) / 1024**3:.1f} GiB",
                        file=sys.stderr,
                        flush=True,
                    )
                    next_progress += 1024**3
            if compressor is not None:
                compressor_stdin.close()
                return_code = compressor.wait()
                if return_code != 0:
                    raise RuntimeError(f"zstd failed for {relative} with status {return_code}")
        if downloaded != int(entry["size"]):
            raise RuntimeError(f"size mismatch for {relative}: expected {entry['size']}, got {downloaded}")
        observed_hash = upstream_hash.hexdigest()
        if observed_hash != entry["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {relative}: expected {entry['sha256']}, got {observed_hash}")
        temporary.replace(destination)
    except BaseException:
        if compressor is not None and compressor.poll() is None:
            compressor.kill()
        temporary.unlink(missing_ok=True)
        raise
    return {
        "upstream_path": relative,
        "upstream_url": url,
        "upstream_size_bytes": downloaded,
        "upstream_sha256": upstream_hash.hexdigest(),
        "local_path": str(destination),
        "local_size_bytes": destination.stat().st_size,
        "local_sha256": _sha256(destination),
        "lossless_compression": "zstd level 3" if entry["compress"] else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root
    manifest_path = output_root / "download_manifest.json"
    if manifest_path.exists():
        parser.error(f"refusing to overwrite completed manifest: {manifest_path}")
    output_root.mkdir(parents=True, exist_ok=True)

    records = [_download(dict(entry), output_root) for entry in FILES]
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "downloaded_at": datetime.now(UTC).isoformat(),
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "github_revision": GITHUB_REVISION,
        "files": records,
    }
    temporary = manifest_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
