"""Download pinned public project resources, with checksums and safe extraction."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATASET_REVISION = "7628202a2180972f291ba1bc6723834921e72c19"
REPOS = {
    "gym-pusht": ("huggingface/gym-pusht", "e49cf07fd32ad6e69443af2921dd8343b0f9ad7d"),
    "diffusion_policy": ("real-stanford/diffusion_policy", "5ba07ac6661db573af695b419a7947ecb704690f"),
}


def read_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "flow-contact-course-project"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def download(job):
    url, relative, expected_size, expected_hash = job
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "flow-contact-course-project"})
        temporary = target.with_suffix(target.suffix + ".partial")
        with urllib.request.urlopen(req, timeout=90) as response, temporary.open("wb") as f:
            while chunk := response.read(1024 * 1024):
                f.write(chunk)
        temporary.replace(target)
    size = target.stat().st_size
    with target.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if expected_size is not None and size != expected_size:
        raise ValueError(f"Size mismatch: {relative}")
    if expected_hash and digest != expected_hash:
        raise ValueError(f"SHA256 mismatch: {relative}")
    print(f"Downloaded {relative}: {size:,} bytes", flush=True)
    return {"url": url, "path": str(relative), "bytes": size, "sha256": digest}


def main():
    dataset = "lerobot/pusht"
    revision = DATASET_REVISION
    tree_url = f"https://huggingface.co/api/datasets/{dataset}/tree/{revision}?recursive=true&limit=1000"
    tree = read_json(tree_url)
    jobs = []
    for entry in tree:
        if entry["type"] != "file":
            continue
        path = entry["path"]
        jobs.append((
            f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{path}",
            Path("data/raw/lerobot_pusht") / path,
            entry["size"], entry.get("lfs", {}).get("oid"),
        ))
    for name, (repo, sha) in REPOS.items():
        jobs.append((f"https://codeload.github.com/{repo}/zip/{sha}",
                     Path("downloads") / f"{name}-{sha}.zip", None, None))
    with ThreadPoolExecutor(max_workers=4) as executor:
        files = list(executor.map(download, jobs))
    for name, (_, sha) in REPOS.items():
        archive = ROOT / "downloads" / f"{name}-{sha}.zip"
        destination = (ROOT / "external" / name).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                parts = Path(member.filename).parts[1:]
                if not parts:
                    continue
                target = destination.joinpath(*parts).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError("Unsafe archive path")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(member) as source, target.open("wb") as output:
                        while chunk := source.read(1024 * 1024):
                            output.write(chunk)
    manifest = {"dataset": dataset, "dataset_revision": revision,
                "repositories": REPOS, "files": files,
                "note": "Public upstream data; contact labels must be reconstructed and validated."}
    (ROOT / "resource_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Complete. Dataset revision: {revision}", flush=True)


if __name__ == "__main__":
    main()
