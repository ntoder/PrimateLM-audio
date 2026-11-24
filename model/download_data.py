"""Download the PrimateLM-audio dataset from Zenodo.

What it does, step by step:
  1. Asks Zenodo for the record's metadata and PRINTS THE LICENSE, so we always
     know we're allowed to use the data.
  2. Downloads each zip file (skips files that are already downloaded).
  3. Checks each file's MD5 checksum, so a broken download is caught.
  4. Unzips into data/raw/<zip name>/.

Run from the repo folder:
    python model/download_data.py
"""
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.config import RAW_DIR, DATA_DIR, ZENODO_RECORD, ZENODO_FILES  # noqa: E402

API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
ALLOWED_LICENSES = {"cc-by-4.0", "cc0-1.0", "cc-by-sa-4.0"}  # licenses we accept


def md5_of(path: Path) -> str:
    """Compute a file's MD5 checksum, reading 1 MB at a time to save memory."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """Download url to dest, printing a simple progress percentage."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as out:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dest.name}: {100 * done / total:5.1f}%", end="", flush=True)
    print()
    tmp.rename(dest)  # only rename once the download fully finished


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Metadata + license check
    with urllib.request.urlopen(API_URL) as r:
        record = json.load(r)
    meta = record["metadata"]
    license_id = (meta.get("license") or {}).get("id", "unknown")
    print(f"Dataset : {meta['title']}")
    print(f"Authors : {', '.join(c['name'] for c in meta['creators'])}")
    print(f"DOI     : {record['doi']}")
    print(f"License : {license_id}")
    if license_id not in ALLOWED_LICENSES:
        sys.exit(f"Stopping: license '{license_id}' is not one we have approved.")

    # Save who to credit, so the docs and demo can show it.
    (DATA_DIR / "dataset_info.json").write_text(json.dumps({
        "title": meta["title"], "creators": [c["name"] for c in meta["creators"]],
        "doi": record["doi"], "license": license_id,
        "url": f"https://zenodo.org/records/{ZENODO_RECORD}",
    }, indent=2))

    files = {f["key"]: f for f in record["files"]}
    for name in ZENODO_FILES:
        info = files[name]
        dest = DATA_DIR / name
        expected_md5 = info["checksum"].split(":", 1)[1]

        # 2. Download (skip if we already have a good copy)
        if dest.exists() and md5_of(dest) == expected_md5:
            print(f"  {name}: already downloaded")
        else:
            download(info["links"]["self"], dest)
            # 3. Verify
            if md5_of(dest) != expected_md5:
                dest.unlink()
                sys.exit(f"Checksum mismatch for {name} - please run again.")

        # 4. Unzip
        out_dir = RAW_DIR / name.removesuffix(".zip")
        if not out_dir.exists():
            print(f"  unzipping {name} ...")
            with zipfile.ZipFile(dest) as z:
                z.extractall(out_dir)

    print(f"\nDone. Audio is in {RAW_DIR}")


if __name__ == "__main__":
    main()
