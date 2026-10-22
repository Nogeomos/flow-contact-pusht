"""Build a separate, privacy-reviewed publication tree without changing experiments.

Run with a Python containing pypdf. Checkpoint metadata is handled by a separate
CPU-only process using --torch-python. Existing destination trees are refused.
The repository manifest records original and publication SHA256 values: historical
audit hashes are not silently rewritten to pretend redacted files are originals.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("runs/stages234_20261022")
MODELS = ["contact_main", "flow_action", "contact_seed1", "contact_seed2", "contact_32",
          "contact_128", "contact_hard", "contact_endpoint", "contact_gru", "contact_no_flow", "visual_dp"]
FROZEN = [f"research/{x}.py" for x in ["controller", "evaluate_closed_loop", "contact_learning", "visual_dp"]]
TEXT = {".py", ".ps1", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".csv", ".log", ".cff"}
SECRETS = {
    "github_token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    "huggingface_token": re.compile(r"hf_[A-Za-z0-9]{25,}"),
    "openai_token": re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{35,}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
USER_PATH = re.compile(r"(?i)[a-z]:[\\/]+Users[\\/]+")
EXCLUDED_PARTS = {"__pycache__", "runtime_repair_backup", "runtime_repair_wheels"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sanitize(text):
    # Exact project root first; retain root-relative paths useful after extraction.
    for original, replacement in [(str(ROOT), "."), (str(Path.home()), "<USER_HOME>")]:
        forms = {original, original.replace("\\", "/"), original.replace("/", "\\")}
        forms |= {s.replace("\\", "\\\\") for s in list(forms)}
        for form in sorted(forms, key=len, reverse=True):
            text = re.sub(re.escape(form), lambda _: replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"\b" + re.escape(Path.home().name) + r"\b", "<USER>", text)
    return text


def is_text(path):
    return path.suffix.lower() in TEXT or path.name in {".gitignore", ".gitattributes", "LICENSE", "NOTICE"}


def safe_relative(path):
    rel = path.relative_to(ROOT)
    if path.is_symlink() or any(p in EXCLUDED_PARTS for p in rel.parts):
        return None
    return rel


def selection():
    """Explicit allowlist; never scan credentials, environments, caches or tmp."""
    repo = {}
    def add(source, dest=None):
        source = Path(source)
        if not source.is_absolute():
            source = ROOT/source
        if source.is_file() and safe_relative(source) is not None:
            repo[Path(dest).as_posix() if dest else source.relative_to(ROOT).as_posix()] = source
    for p in ROOT.iterdir():
        if p.is_file() and (p.suffix in {".md", ".cff"} or p.name.startswith("requirements") or
                            p.name in {".gitignore", ".gitattributes", "resource_manifest.json", "LICENSE", "NOTICE", "pyproject.toml"}):
            add(p)
    for folder in ["research", "scripts", "docs", "artifacts"]:
        if not (ROOT/folder).exists():
            continue
        for p in (ROOT/folder).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                add(p)
    for name in ["splits.json", "windows.json"]:
        add(Path("data/derived")/name)
    for p in (ROOT/"reports").rglob("*"):
        if p.is_file() and "historical_palette_debug" not in p.parts:
            rel = p.relative_to(ROOT/"reports")
            if rel.parts[0] == "portability_20261022":
                add(p, Path("artifacts/portability")/Path(*rel.parts[1:]))
            else:
                add(p, Path("artifacts/stage1")/rel)
    for folder in ["stage1_20261022_01", "stage1_20261022_02"]:
        base = ROOT/"runs"/folder
        for p in base.glob("*.json"):
            add(p, Path("artifacts/stage1")/folder/p.name)
    example = ROOT/"runs/stage1_20261022_02"
    for source, dest in [("simple_push_a/simple_push.gif", "simple_push.gif"),
                         ("visualization/contact_sample.png", "contact_sample.png"),
                         ("visualization/no_contact_sample.png", "no_contact_sample.png")]:
        add(example/source, Path("artifacts/stage1")/dest)
    base = ROOT/RUN
    selected_names = {"config.json", "history.json", "training_manifest.json", "training_result.json", "test_metrics.json",
                      "summary.json", "failure_analysis.json", "figure_sources.json", "manifest.json", "result.json"}
    for p in base.rglob("*"):
        if not p.is_file() or safe_relative(p) is None:
            continue
        rel = p.relative_to(base)
        take = (len(rel.parts) == 1 and p.suffix == ".json") or p.name in selected_names
        take |= rel.parts[0] in {"figures", "stage2_samples"}
        take |= rel.parts[0].endswith("_test_analysis")
        take |= p.suffix in {".py", ".ps1"}
        if take:
            add(p, Path("artifacts/stages234")/rel)
    for p in (ROOT/"runs/wall_diagnostics_20261022").glob("*"):
        if p.suffix in {".json", ".png"}:
            add(p, Path("artifacts/wall_diagnostics")/p.name)
    add("output/pdf/PushT_阶段工作报告_2026-10-22.pdf", "artifacts/stage1/report.pdf")
    add("output/pdf/PushT_第二至四阶段实验报告_2026-10-22.pdf", "artifacts/stages234/report.pdf")
    evidence = {}
    for folder in ["runs", "reports", "data/derived/replayed_contacts"]:
        for p in (ROOT/folder).rglob("*"):
            if not p.is_file() or safe_relative(p) is None or p.suffix in {".pt", ".pth", ".pyc"}:
                continue
            evidence[p.relative_to(ROOT).as_posix()] = p
    # Exact frozen split/window membership and upstream acquisition provenance.
    for p in [ROOT/"data/derived/splits.json", ROOT/"data/derived/windows.json", ROOT/"resource_manifest.json"]:
        evidence[p.relative_to(ROOT).as_posix()] = p
    models = {str(RUN/name/"best.pt").replace("\\", "/"): ROOT/RUN/name/"best.pt" for name in MODELS}
    return {"repository": repo, "evidence": evidence, "models": models}


def transform_pdf(source, target):
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import ContentStream, TextStringObject, NameObject
    reader = PdfReader(source)
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    modified_pages = []
    changed = 0
    for i, page in enumerate(writer.pages):
        content = ContentStream(page.get_contents(), writer)
        page_changes = 0
        for operands, operator in content.operations:
            items = operands[0] if operator == b"TJ" else operands if operator in {b"Tj", b"'", b'"'} else []
            for k, item in enumerate(items):
                if isinstance(item, TextStringObject):
                    new = sanitize(str(item))
                    if new != str(item):
                        items[k] = TextStringObject(new)
                        page_changes += 1
        if page_changes:
            # PDF stream objects must be indirect; a direct stream is accepted
            # by some text extractors but rejected by strict renderers.
            page[NameObject("/Contents")] = writer._add_object(content)
            modified_pages.append(i+1)
            changed += page_changes
    if not changed:
        shutil.copy2(source, target)
    else:
        with target.open("wb") as f:
            writer.write(f)
    check = PdfReader(target)
    text = "\n".join(p.extract_text() or "" for p in check.pages)
    if USER_PATH.search(text) or re.search(r"\b"+re.escape(Path.home().name)+r"\b", text):
        raise RuntimeError("PDF still contains a local user path after content-stream redaction")
    if len(check.pages) != len(reader.pages):
        raise RuntimeError("PDF page count changed")
    return {"pages": len(check.pages), "modified_pages": modified_pages, "text_operands_redacted": changed}


CHECKPOINT_HELPER = r'''
import sys,json,re,torch,hashlib
from pathlib import Path
source,target,root,home=sys.argv[1:]
def clean(v):
 if isinstance(v,str):
  for old,new in [(root,'.'),(home,'<USER_HOME>')]:
   for form in sorted({old,old.replace('\\','/'),old.replace('/','\\')},key=len,reverse=True):
    v=re.sub(re.escape(form),lambda _:new,v,flags=re.I)
  return re.sub(r'\b'+re.escape(Path(home).name)+r'\b','<USER>',v)
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,list):return [clean(x) for x in v]
 return v
original=torch.load(source,map_location='cpu',weights_only=True)
published=dict(original);published['manifest']=clean(original['manifest'])
torch.save(published,target)
loaded=torch.load(target,map_location='cpu',weights_only=True)
assert set(loaded)==set(original)
for key in original:
 if key not in ['manifest','state_dict']:assert loaded[key]==original[key],key
assert set(loaded['state_dict'])==set(original['state_dict'])
h=hashlib.sha256();count=0;elements=0
for key,t in original['state_dict'].items():
 other=loaded['state_dict'][key]
 assert t.shape==other.shape and t.dtype==other.dtype and torch.equal(t,other),key
 h.update(key.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 count+=1;elements+=t.numel()
print(json.dumps({'tensors_exactly_equal':True,'tensor_count':count,'tensor_elements':elements,
 'tensor_payload_sha256':h.hexdigest(),'changed_only_checkpoint_field':'manifest'}))
'''


def copy_file(source, target, torch_python):
    target.parent.mkdir(parents=True, exist_ok=True)
    details = {}
    transformation = "unchanged"
    if source.suffix == ".pdf":
        details = transform_pdf(source, target)
        if details["text_operands_redacted"]:
            transformation = "pdf_local_path_text_operands_redacted"
    elif source == ROOT/RUN/"visual_dp/best.pt":
        result = subprocess.run([str(torch_python), "-c", CHECKPOINT_HELPER, str(source), str(target), str(ROOT), str(Path.home())],
                                text=True, capture_output=True, check=True)
        details = json.loads(result.stdout)
        transformation = "checkpoint_manifest_paths_redacted_tensors_unchanged"
    elif is_text(source):
        raw = source.read_bytes()
        decoded = raw.decode("utf-8-sig")
        cleaned = sanitize(decoded)
        if cleaned != decoded:
            if source.relative_to(ROOT).as_posix() in FROZEN:
                raise RuntimeError("Frozen scientific source requires redaction; manual review required")
            target.write_bytes(cleaned.encode("utf-8"))
            transformation = "local_paths_redacted_in_publication_copy"
        else:
            target.write_bytes(raw)
    else:
        shutil.copy2(source, target)
    return transformation, details


def inspect_file(path):
    """Report categories/counts only, never secret-like values."""
    if path.suffix == ".pdf":
        from pypdf import PdfReader
        text = "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    elif path.suffix == ".pt":
        with zipfile.ZipFile(path) as z:
            text = "\n".join(z.read(n).decode("utf-8", errors="ignore") for n in z.namelist() if n.endswith("data.pkl"))
    elif path.suffix == ".npz":
        # Text arrays only; numeric recordings remain byte-identical copies.
        import numpy as np
        text_parts = []
        with np.load(path, allow_pickle=False) as z:
            for name in z.files:
                a = z[name]
                if a.dtype.kind in "US":
                    text_parts.extend(a.astype(str).reshape(-1).tolist())
        text = "\n".join(text_parts)
    elif is_text(path):
        text = path.read_text(encoding="utf-8-sig")
    else:
        return {}
    findings = {name: len(pattern.findall(text)) for name, pattern in SECRETS.items() if pattern.search(text)}
    if USER_PATH.search(text):
        findings["absolute_user_path"] = len(USER_PATH.findall(text))
    return findings


def make_zip(directory, path):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file in sorted(directory.rglob("*")):
            if not file.is_file():
                continue
            info = zipfile.ZipInfo(file.relative_to(directory).as_posix(), (2026,10,22,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, file.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"publication")
    parser.add_argument("--torch-python", type=Path, default=ROOT/".venv-train/Scripts/python.exe")
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT)  # Refuse targets outside the authorized project.
    for name in ["repository", "release", "staging"]:
        if (output/name).exists():
            raise FileExistsError(output/name)
    selection_map = selection()
    sources = {str(p): sha(p) for items in selection_map.values() for p in items.values()}
    directories = {"repository": output/"repository", "models": output/"staging/models", "evidence": output/"staging/evidence"}
    for directory in directories.values():
        directory.mkdir(parents=True)
    entries = []
    problems = []
    for destination, items in selection_map.items():
        for relative, source in sorted(items.items()):
            target = directories[destination]/relative
            transformation, details = copy_file(source, target, args.torch_python)
            entry = {"destination": destination, "source_path": source.relative_to(ROOT).as_posix(),
                     "source_sha256": sources[str(source)], "published_path": relative,
                     "published_sha256": sha(target), "bytes": target.stat().st_size,
                     "transformation": transformation}
            if details:
                entry["details"] = details
            entries.append(entry)
            findings = inspect_file(target)
            if findings:
                problems.append({"destination": destination, "path": relative, "categories": findings})
    # Generated public metadata contains no hidden full-path values.
    model_manifest = {"schema_version": 1, "release_asset": "models.zip",
                      "note": "Only the visual checkpoint manifest paths changed; all learned tensor values are exactly equal. Historical audit hashes refer to originals.",
                      "files": [entry for entry in entries if entry["destination"] == "models"]}
    model_path = directories["repository"]/"artifacts/models/manifest.json"
    save(model_path, model_manifest)
    entries.append({"destination": "repository", "source_path": None, "source_sha256": None,
                    "published_path": "artifacts/models/manifest.json", "published_sha256": sha(model_path),
                    "bytes": model_path.stat().st_size, "transformation": "generated_public_model_provenance"})
    if problems:
        save(output/"publication_scan_failure.json", {"status": "failed", "findings": problems})
        raise RuntimeError(f"Publication privacy scan found {len(problems)} files; see category-only report")
    if any(sha(Path(path)) != expected for path, expected in sources.items()):
        raise RuntimeError("A source file changed during packaging; wait for writers and build a fresh output directory")
    frozen = {name: sha(ROOT/name) for name in FROZEN}
    for name, digest in frozen.items():
        if sha(directories["repository"]/name) != digest:
            raise RuntimeError("Frozen scientific source changed in publication")
    release = output/"release"
    release.mkdir()
    assets = []
    for name in ["models", "evidence"]:
        target = release/f"{name}.zip"
        make_zip(directories[name], target)
        assets.append({"filename": target.name, "sha256": sha(target), "bytes": target.stat().st_size,
                       "file_count": sum(entry["destination"] == name for entry in entries)})
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "files": entries, "frozen_scientific_sources": frozen,
                "self_excluded": ["artifacts/PUBLICATION_MANIFEST.json", "artifacts/RELEASE_ASSETS.json"],
                "historical_hash_policy": "source_sha256 records immutable original artifacts; published_sha256 authenticates publication copies. Original audit hash fields are preserved as historical evidence and may refer to redacted or excluded files. Run verify_publication.py for the publication tree, and fresh experiments for new execution validation.",
                "excluded": ["virtual environments", "raw upstream dataset copies", "visual image cache", "download archives", "external checkouts", "package wheels and backups", "temporary helpers", "probe checkpoints", "last checkpoints"]}
    save(directories["repository"]/"artifacts/PUBLICATION_MANIFEST.json", manifest)
    save(directories["repository"]/"artifacts/RELEASE_ASSETS.json",
         {"schema_version": 1, "release_tag": "v1.0.0-experiments", "assets": assets})
    summary = {"status": "passed", "sensitive_pattern_findings": [], "absolute_user_path_findings": [],
               "scan_scope": "Every selected text file, PDF extracted text, checkpoint pickle metadata and NPZ string array; no credential stores or files outside the project were read.",
               "original_files_unchanged": len(sources), "frozen_scientific_sources_unchanged": frozen,
               "files_by_destination": {name: sum(x["destination"] == name for x in entries) for name in directories},
               "bytes_by_destination": {name: sum(x["bytes"] for x in entries if x["destination"] == name) for name in directories},
               "transformations": {name: sum(x["transformation"] == name for x in entries) for name in sorted({x["transformation"] for x in entries})},
               "assets": assets, "pdf_visual_review": "Required separately: render and inspect the changed public PDF page before release."}
    save(output/"publication_build_audit.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
