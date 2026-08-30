"""Optional PyMOL cartoon renders from raw mmCIF.

PyMOL is not a required dependency. If no executable is found, callers skip.
Cartoons are rendered from ``raw/*.cif.gz``, not C-alpha npz graphs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.io.discover import discover_mmcif_files
from src.paths import PLOTS_DIR, RAW_DIR, ensure_dirs

_PYMOL_NAMES = ("pymol", "pymol.exe", "PyMOL.exe", "PyMOLWin.exe")
_PYMOL_DIRS = (
    Path(r"C:\Program Files\PyMOL"),
    Path(r"C:\Program Files\Schrödinger\PyMOL"),
    Path(r"C:\Program Files (x86)\PyMOL"),
)


def _depth_limited_search(root: Path, names: tuple[str, ...], max_depth: int = 4) -> Path | None:
    if not root.is_dir():
        return None
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).resolve().relative_to(root)
        depth = 0 if rel == Path(".") else len(rel.parts)
        if depth >= max_depth:
            dirnames.clear()
            continue
        for name in names:
            if name in filenames:
                return Path(dirpath) / name
    return None


def find_pymol() -> Path | None:
    for name in _PYMOL_NAMES:
        found = shutil.which(name)
        if found:
            return Path(found)
    for folder in _PYMOL_DIRS:
        if not folder.is_dir():
            continue
        for name in _PYMOL_NAMES:
            candidate = folder / name
            if candidate.is_file():
                return candidate
        found = _depth_limited_search(folder, _PYMOL_NAMES, max_depth=3)
        if found:
            return found
    extra_roots = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" if os.environ.get("LOCALAPPDATA") else None,
    ]
    for root in extra_roots:
        if root is None or not root.is_dir():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if "pymol" not in child.name.lower() and "schrodinger" not in child.name.lower():
                continue
            found = _depth_limited_search(child, _PYMOL_NAMES, max_depth=4)
            if found:
                return found
    return None


def pdb_id_from_mmcif(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".cif.gz"):
        name = name[: -len(".cif.gz")]
    elif name.lower().endswith(".cif"):
        name = name[: -len(".cif")]
    return name.upper()


def _pml_script(cif_path: Path, png_path: Path, object_name: str) -> str:
    cif = cif_path.resolve().as_posix()
    png = png_path.resolve().as_posix()
    return "\n".join(
        [
            f'load "{cif}", {object_name}',
            "remove solvent",
            "hide everything",
            "show cartoon",
            "show sticks, organic",
            "spectrum chain",
            "bg_color white",
            "set ray_opaque_background, 1",
            "orient",
            "zoom",
            "ray 1200, 900",
            f'png "{png}", dpi=150',
            "quit",
            "",
        ]
    )


def render_cartoon(
    cif_path: Path,
    out_path: Path | None = None,
    *,
    pymol: Path | None = None,
    timeout_s: float = 180.0,
) -> Path:
    pymol = pymol or find_pymol()
    if pymol is None:
        raise FileNotFoundError("PyMOL executable not found")
    cif_path = Path(cif_path)
    if not cif_path.is_file():
        raise FileNotFoundError(cif_path)
    ensure_dirs()
    pdb_id = pdb_id_from_mmcif(cif_path)
    out_path = out_path or (PLOTS_DIR / f"{pdb_id}_cartoon.png")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    script = _pml_script(cif_path, out_path, pdb_id)
    with tempfile.NamedTemporaryFile("w", suffix=".pml", delete=False, encoding="utf-8") as fh:
        fh.write(script)
        pml = Path(fh.name)
    try:
        proc = subprocess.run(
            [str(pymol), "-cq", str(pml)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    finally:
        pml.unlink(missing_ok=True)
    if proc.returncode != 0 or not out_path.is_file() or out_path.stat().st_size < 100:
        err = (proc.stderr or proc.stdout or "").strip()[-1500:]
        raise RuntimeError(
            f"PyMOL cartoon failed for {cif_path.name} (exit {proc.returncode}): {err}"
        )
    return out_path


def render_raw_cartoons(
    raw_dir: Path | None = None,
    *,
    skip_if_missing: bool = True,
) -> tuple[list[Path], str | None]:
    """Render cartoons for every mmCIF in raw/.

    Returns (written_paths, skip_reason). skip_reason is set when PyMOL is absent
    and skip_if_missing is True.
    """
    pymol = find_pymol()
    if pymol is None:
        msg = "PyMOL not found; skipping cartoon renders"
        if skip_if_missing:
            return [], msg
        raise FileNotFoundError(msg)
    written: list[Path] = []
    for cif in discover_mmcif_files(raw_dir or RAW_DIR):
        written.append(render_cartoon(cif, pymol=pymol))
    return written, None
