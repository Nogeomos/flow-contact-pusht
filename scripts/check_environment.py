r"""Collect current machine/runtime evidence without installing packages.

Usage: .venv\Scripts\python.exe -B scripts/check_environment.py --output-dir PATH
The output directory may exist, but environment_check.json must not exist.
"""
from pathlib import Path
import argparse
import base64
import contextlib
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]


def command(arguments, timeout=25):
    result = {"arguments": arguments}
    try:
        completed = subprocess.run(
            arguments, capture_output=True,
            timeout=timeout, cwd=ROOT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        result["returncode"] = completed.returncode
        for label, raw in [("stdout", completed.stdout), ("stderr", completed.stderr)]:
            try:
                decoded, encoding = raw.decode("utf-8"), "utf-8"
            except UnicodeDecodeError:
                encoding = "mbcs" if os.name == "nt" else "latin-1"
                decoded = raw.decode(encoding, errors="replace")
                # Preserve the original diagnostic bytes even if a codepage is ambiguous.
                result[label + "_raw_base64"] = base64.b64encode(raw).decode("ascii")
            result[label], result[label + "_encoding"] = decoded.strip(), encoding
    except Exception as error:
        result.update(returncode=None, error=f"{type(error).__name__}: {error}")
    return result


def registry_snapshot():
    result = {}
    if os.name != "nt":
        return {"status": "not_windows"}
    import winreg
    for label, key_name, names in [
        ("windows", r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
         ["ProductName", "DisplayVersion", "CurrentBuildNumber", "UBR", "EditionID"]),
        ("cpu", r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
         ["ProcessorNameString", "VendorIdentifier", "~MHz"]),
    ]:
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_name) as key:
                values = {}
                for name in names:
                    try:
                        values[name] = winreg.QueryValueEx(key, name)[0]
                    except OSError as error:
                        values[name] = {"error": str(error)}
                result[label] = values
        except OSError as error:
            result[label] = {"error": str(error)}
    result["note"] = "Windows ProductName can retain a legacy marketing name; compare the OS build and platform.win32_ver."
    return result


def memory_snapshot():
    if os.name != "nt":
        return {"status": "unavailable", "reason": "GlobalMemoryStatusEx is Windows-only"}

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    try:
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        function = kernel32.GlobalMemoryStatusEx
        function.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
        function.restype = ctypes.c_int
        if not function(ctypes.byref(status)):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"status": "ok", "source": "kernel32.GlobalMemoryStatusEx",
                "physical_total_bytes": status.ullTotalPhys,
                "physical_available_bytes": status.ullAvailPhys,
                "physical_total_gib": status.ullTotalPhys / 2**30,
                "physical_available_gib": status.ullAvailPhys / 2**30,
                "memory_load_percent": status.dwMemoryLoad}
    except Exception as error:
        return {"status": "failed", "error": f"{type(error).__name__}: {error}"}


def package_imports():
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    sys.path.insert(0, str(ROOT / "external/gym-pusht"))
    packages = [("numpy", "numpy"), ("pymunk", "pymunk"), ("gymnasium", "gymnasium"),
                ("pygame", "pygame"), ("shapely", "shapely"), ("opencv-python", "cv2"),
                ("av", "av"), ("pyarrow", "pyarrow"), ("pillow", "PIL"),
                ("gym-pusht", "gym_pusht")]
    results = []
    for distribution, module_name in packages:
        item = {"distribution": distribution, "module": module_name}
        try:
            item["installed_distribution_version"] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            item["installed_distribution_version"] = None
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured), warnings.catch_warnings(record=True) as found:
                warnings.simplefilter("always")
                module = importlib.import_module(module_name)
            item.update(import_ok=True, module_file=getattr(module, "__file__", None),
                        warnings=[str(w.message) for w in found])
            runtime = getattr(module, "__version__", None)
            if runtime is None:
                runtime = getattr(module, "version", None)
            item["runtime_version_attribute"] = str(runtime) if runtime is not None else None
            if distribution == "gym-pusht":
                item["source"] = "vendored source from external/gym-pusht, not necessarily an installed distribution"
        except Exception as error:
            item.update(import_ok=False, error=f"{type(error).__name__}: {error}")
        if captured.getvalue():
            item["import_stdout"] = captured.getvalue().strip()
        results.append(item)
    return results


def gpu_snapshot():
    candidates = [shutil.which("nvidia-smi")]
    if os.name == "nt":
        candidates += [str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/nvidia-smi.exe"),
                       str(Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation/NVSMI/nvidia-smi.exe")]
    executable = next((p for p in candidates if p and Path(p).is_file()), None)
    if executable is None:
        return {"status": "not_found", "searched_paths": candidates}
    fields = ["name", "uuid", "driver_version", "memory.total", "memory.used", "memory.free"]
    query = command([executable, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"])
    report = {"source": "nvidia-smi", "query": query, "gpus": []}
    if query.get("returncode") == 0:
        for row in csv.reader(query["stdout"].splitlines()):
            if not row:
                continue
            values = dict(zip(fields, [value.strip() for value in row]))
            for field in fields[3:]:
                try:
                    values[field + "_mib"] = float(values.pop(field))
                except (KeyError, ValueError):
                    pass
            report["gpus"].append(values)
        report["status"] = "ok"
    else:
        report["status"] = "query_failed"
    report["driver_summary"] = command([executable])
    report["note"] = "CUDA Version printed by nvidia-smi is driver capability, not proof of an installed toolkit or a usable PyTorch CUDA build."
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    target = output / "environment_check.json"
    if target.exists():
        parser.error(f"Refusing to overwrite existing output: {target}")
    output.mkdir(parents=True, exist_ok=True)
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    disk = shutil.disk_usage(ROOT)
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    cim = command([powershell, "-NoProfile", "-NonInteractive", "-Command",
                   "[Console]::OutputEncoding=[Text.UTF8Encoding]::new(); $ErrorActionPreference='Stop'; Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory,NumberOfLogicalProcessors | ConvertTo-Json -Compress"]) if powershell else {"error": "PowerShell not found"}
    torch_spec = importlib.util.find_spec("torch")
    result = {
        "schema_version": 1, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "operating_system": {"platform": platform.platform(), "system": platform.system(),
                             "release": platform.release(), "version": platform.version(),
                             "machine": platform.machine(), "win32_ver": platform.win32_ver(),
                             "registry": registry_snapshot()},
        "python": {"version": sys.version, "executable": sys.executable,
                   "prefix": sys.prefix, "base_prefix": sys.base_prefix,
                   "is_virtual_environment": sys.prefix != sys.base_prefix,
                   "is_project_virtual_environment": Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(),
                   "pyvenv_cfg": cfg.read_text(encoding="utf-8") if cfg.exists() else None},
        "cpu_logical_count": os.cpu_count(), "cim_probe": cim,
        "memory_fallback": memory_snapshot(),
        "disk": {"queried_path": str(ROOT), "volume": ROOT.anchor,
                 "total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free,
                 "free_gib": disk.free / 2**30},
        "git": command([shutil.which("git") or "git", "--version"]),
        "nvidia": gpu_snapshot(),
        "cuda_toolkit": command([shutil.which("nvcc") or "nvcc", "--version"]),
        "dependencies": package_imports(),
        "torch": {"module_discoverable": torch_spec is not None,
                  "module_origin": torch_spec.origin if torch_spec else None,
                  "imported_or_installed_by_this_check": False},
        "scope": "Read-only inspection plus this new report; no training, package installation, or driver changes.",
    }
    result["data_dependencies_import_ok"] = all(item["import_ok"] for item in result["dependencies"])
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(target), "data_dependencies_import_ok": result["data_dependencies_import_ok"],
                      "is_project_venv": result["python"]["is_project_virtual_environment"],
                      "gpu": result["nvidia"]["gpus"] if "gpus" in result["nvidia"] else [],
                      "ram_gib": result["memory_fallback"].get("physical_total_gib")}))


if __name__ == "__main__":
    main()
