"""Zero-dependency backend for the figure labeling tool.

A tiny stdlib-only HTTP server that powers ``index.html``. It lets you browse
the review folder tree, preview cropped figures, reassign them to the correct
class folder, and -- on *Commit* -- physically move the files and stamp a
``manual_root`` attribute onto the matching records in ``manifest.jsonl``.

Run it with either::

    uv run python -m DocumentFigureClassifier.extract.labeling_tool
    python src/DocumentFigureClassifier/extract/labeling_tool/server.py

then open http://localhost:8765/ (it opens automatically unless --no-browser).

Nothing on disk changes until you press *Commit* in the UI. Reassignments live
only in the browser until then, so you can relabel freely and still back out.
"""

from __future__ import annotations

import argparse
import gzip
import json
import mimetypes
import os
import shutil
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
INDEX_HTML = HERE / "index.html"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

# Only worth compressing the JSON listings; images are already compressed.
GZIP_MIN_BYTES = 4096

# Filled in by main() from CLI args; used to seed the UI's default paths.
DEFAULT_ROOT: Path | None = None
DEFAULT_MANIFEST: Path | None = None


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #
def _repo_root() -> Path:
    """Best-effort walk up to the project root (contains a ``data`` folder)."""
    for parent in [HERE, *HERE.parents]:
        if (parent / "data" / "parsed").is_dir():
            return parent
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def _is_within(path: Path, root: Path) -> bool:
    """True if ``path`` resolves to somewhere inside ``root`` (or equals it)."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _root_manifest_mismatch(root: Path, manifest: Path | None) -> bool:
    """True when an existing manifest lives outside ``root``'s dataset folder.

    Commits move files under ``root`` and stamp ``manual_root`` onto the matching
    records in ``manifest``. If the two belong to different dataset trees (e.g. a
    ``data/parsed`` manifest paired with a ``data/parsed_new/review`` root), none
    of the moved filenames exist in the manifest -- so nothing is written and the
    moves silently lose their labels. Callers surface this as a warning. A missing
    manifest returns False; that case is reported separately as "not found".
    """
    if manifest is None or not manifest.is_file():
        return False
    return not _is_within(root, manifest.parent)


def _resolve_manifest(raw) -> Path | None:
    """Normalise a user-supplied manifest path. ``None`` when nothing was given.

    Always resolved, so the root/manifest comparison below can't be fooled by a
    relative path or a ``..`` segment.
    """
    text = str(raw or "").strip()
    return Path(text).expanduser().resolve() if text else None


def _image_names(folder: Path) -> list[str]:
    """Image basenames directly inside ``folder``, sorted case-insensitively.

    ``os.scandir`` matters here: the review folders hold tens of thousands of
    crops, and it reuses the attributes the directory walk already returned
    instead of issuing a ``stat`` per file (~10x faster on Windows).
    """
    try:
        with os.scandir(folder) as it:
            names = [
                e.name
                for e in it
                if os.path.splitext(e.name)[1].lower() in IMAGE_EXTS and e.is_file()
            ]
    except OSError:
        return []
    names.sort(key=str.lower)
    return names


def _count_images(folder: Path) -> int:
    try:
        with os.scandir(folder) as it:
            return sum(
                1
                for e in it
                if os.path.splitext(e.name)[1].lower() in IMAGE_EXTS and e.is_file()
            )
    except OSError:
        return 0


# The manifest is ~12 MB / 23k records and every folder switch used to re-parse
# it. Cache the parsed index and re-read only when the file changes on disk (a
# commit rewrites it, which bumps mtime and invalidates this by itself).
_MANIFEST_LOCK = threading.Lock()
_MANIFEST_CACHE: dict[str, tuple[tuple[int, int], dict[str, dict]]] = {}


def _load_manifest_index(manifest: Path | None) -> dict[str, dict]:
    """Map basename -> manifest record. Empty dict if manifest is unavailable."""
    if not manifest:
        return {}
    try:
        st = manifest.stat()
    except OSError:
        return {}
    key = str(manifest)
    stamp = (st.st_mtime_ns, st.st_size)

    with _MANIFEST_LOCK:
        cached = _MANIFEST_CACHE.get(key)
        if cached and cached[0] == stamp:
            return cached[1]

        index: dict[str, dict] = {}
        try:
            with manifest.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    fn = obj.get("filename")
                    if fn:
                        index[fn] = obj
        except OSError:
            return {}

        _MANIFEST_CACHE.clear()  # one manifest at a time; they are large
        _MANIFEST_CACHE[key] = (stamp, index)
        return index


def _manifest_status(
    root: Path,
    manifest: Path | None,
    index: dict[str, dict],
    names: list[str] | None = None,
) -> dict:
    """Classify the root/manifest pairing for the UI's warning banner.

    Path location alone is a weak test -- a manifest in a *parent* directory
    passes it while still belonging to another dataset. So when the caller knows
    which files the folder holds, the pairing is judged on evidence instead: if
    not one of them has a record, the manifest is the wrong one.
    """
    st = {
        "path": str(manifest) if manifest else "",
        "exists": bool(manifest and manifest.is_file()),
        "records": len(index),
        "matched": None,
        "checked": 0,
        "status": "none",
        "message": None,
    }

    if manifest is None:
        st["message"] = (
            "No manifest set - files will still move, but manual_root won't be "
            "written for them."
        )
        return st

    if not st["exists"]:
        st["status"] = "missing"
        st["message"] = (
            f"Manifest not found: {manifest}  -  files will still move, but "
            "manual_root won't be written for them."
        )
        return st

    if not _is_within(root, manifest.parent):
        st["status"] = "outside"
        st["message"] = (
            f"Root folder and manifest belong to different datasets. Root: {root} "
            f"- manifest: {manifest}. Moves under this root match no records "
            "there, so manual_root won't be written."
        )
        return st

    if names:
        matched = sum(1 for n in names if n in index)
        st["matched"] = matched
        st["checked"] = len(names)
        if matched == 0:
            st["status"] = "no_match"
            st["message"] = (
                f"None of the {len(names)} image(s) in this folder appear in "
                f"{manifest.name} ({len(index)} records) - it belongs to a "
                "different dataset, so manual_root won't be written."
            )
            return st

    st["status"] = "ok"
    return st


# --------------------------------------------------------------------------- #
# Commit: move files + patch the manifest
# --------------------------------------------------------------------------- #
def do_commit(payload: dict) -> dict:
    """Apply the queued reassignments: move files, then stamp ``manual_root``."""
    root = Path(payload["root"]).expanduser().resolve()
    manifest = _resolve_manifest(payload.get("manifest"))
    moves = payload.get("moves", [])

    report: dict = {
        "moved": [],
        "skipped": [],
        "manifest_updated": 0,
        "manifest_no_record": [],
        "manifest_path": str(manifest) if manifest else None,
        "manifest_written": False,
        "root_manifest_mismatch": _root_manifest_mismatch(root, manifest),
    }
    successful: dict[str, str] = {}  # filename -> destination folder

    for mv in moves:
        fname = mv.get("file")
        src_sub = mv.get("from")
        dst_sub = mv.get("to")
        if not (fname and src_sub and dst_sub) or src_sub == dst_sub:
            continue

        src = root / src_sub / fname
        dst_dir = root / dst_sub
        dst = dst_dir / fname

        if not (_is_within(src, root) and _is_within(dst, root)):
            report["skipped"].append({"file": fname, "reason": "path escapes root"})
            continue
        if not src.is_file():
            report["skipped"].append({"file": fname, "reason": "source file missing"})
            continue
        if dst.exists():
            report["skipped"].append({"file": fname, "reason": f"already exists in {dst_sub}"})
            continue

        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        except OSError as exc:
            report["skipped"].append({"file": fname, "reason": str(exc)})
            continue

        report["moved"].append({"file": fname, "from": src_sub, "to": dst_sub})
        successful[fname] = dst_sub

    # Patch manifest.jsonl -- rewrite only touched records, preserve the rest
    # byte-for-byte, and swap atomically so a crash can't corrupt the file.
    if manifest and manifest.is_file() and successful:
        updated: set[str] = set()
        out_lines: list[str] = []
        with manifest.open(encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    out_lines.append(line if line.endswith("\n") else line + "\n")
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError:
                    out_lines.append(line if line.endswith("\n") else line + "\n")
                    continue
                fn = obj.get("filename")
                if fn in successful:
                    obj["manual_root"] = successful[fn]
                    updated.add(fn)
                    out_lines.append(json.dumps(obj, ensure_ascii=False) + "\n")
                else:
                    out_lines.append(line if line.endswith("\n") else line + "\n")

        tmp = manifest.with_name(manifest.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            fh.writelines(out_lines)
        os.replace(tmp, manifest)

        report["manifest_updated"] = len(updated)
        report["manifest_written"] = True
        report["manifest_no_record"] = [f for f in successful if f not in updated]
        if not updated:
            report["root_manifest_mismatch"] = True

    return report


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "FigureLabeler/1.0"
    # Keep-alive: a review session fires hundreds of image requests, and a fresh
    # TCP connection per image is a good part of the "why is this so slow"
    # feeling. Every response below sets Content-Length, as HTTP/1.1 requires.
    protocol_version = "HTTP/1.1"
    timeout = 30  # reap idle keep-alive sockets instead of pinning a thread
    _responded = False  # reset per request; see _error()

    # -- small response helpers -------------------------------------------- #
    def _send_json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status, gzip_ok=True)

    def _send_bytes(
        self,
        data: bytes,
        content_type: str,
        status: int = 200,
        gzip_ok: bool = False,
        cache_control: str = "no-store",
    ) -> None:
        self._responded = True
        encoding = None
        accepts_gzip = "gzip" in self.headers.get("Accept-Encoding", "")
        if gzip_ok and accepts_gzip and len(data) >= GZIP_MIN_BYTES:
            # The image listings are very repetitive JSON: ~20x smaller for ~50 ms.
            data = gzip.compress(data, 5)
            encoding = "gzip"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if encoding:
            self.send_header("Content-Encoding", encoding)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(data)

    def _error(self, msg: str, status: int = 400) -> None:
        if self._responded:
            # Headers already went out (e.g. the socket died mid-body). A second
            # response would desync the keep-alive stream and leave the browser
            # waiting, so drop the connection instead.
            self.close_connection = True
            return
        self._send_json({"error": msg}, status)

    def log_message(self, *args) -> None:  # keep the console quiet
        pass

    # -- routing ----------------------------------------------------------- #
    def do_GET(self) -> None:
        self._responded = False
        parsed = urlparse(self.path)
        route = parsed.path
        qs = parse_qs(parsed.query)

        try:
            if route in ("/", "/index.html"):
                self._serve_index()
            elif route == "/api/defaults":
                self._api_defaults()
            elif route == "/api/listdir":
                self._api_listdir(qs)
            elif route == "/api/folders":
                self._api_folders(qs)
            elif route == "/api/images":
                self._api_images(qs)
            elif route == "/api/check":
                self._api_check(qs)
            elif route == "/img":
                self._serve_image(qs)
            else:
                self._error("not found", 404)
        except Exception as exc:  # noqa: BLE001 - surface any error to the UI
            self._error(f"{type(exc).__name__}: {exc}", 500)

    def do_POST(self) -> None:
        self._responded = False
        parsed = urlparse(self.path)
        if parsed.path != "/api/commit":
            self._error("not found", 404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._send_json(do_commit(payload))
        except Exception as exc:  # noqa: BLE001
            self._error(f"{type(exc).__name__}: {exc}", 500)

    # -- endpoints --------------------------------------------------------- #
    def _serve_index(self) -> None:
        if not INDEX_HTML.is_file():
            self._error("index.html not found next to server.py", 500)
            return
        self._send_bytes(INDEX_HTML.read_bytes(), "text/html; charset=utf-8")

    def _api_defaults(self) -> None:
        self._send_json(
            {
                "root": str(DEFAULT_ROOT) if DEFAULT_ROOT else "",
                "manifest": str(DEFAULT_MANIFEST) if DEFAULT_MANIFEST else "",
            }
        )

    def _api_listdir(self, qs: dict) -> None:
        """Server-side directory browser for the 'Browse...' picker."""
        raw = (qs.get("path", [""])[0]).strip()
        path = Path(raw).expanduser() if raw else (DEFAULT_ROOT or Path.cwd())
        path = path.resolve()
        if not path.is_dir():
            self._error(f"not a directory: {path}", 404)
            return
        dirs = []
        try:
            for entry in sorted(path.iterdir(), key=lambda p: p.name.lower()):
                if entry.is_dir():
                    dirs.append({"name": entry.name, "images": _count_images(entry)})
        except OSError as exc:
            self._error(str(exc), 500)
            return
        parent = str(path.parent) if path.parent != path else None
        self._send_json({"path": str(path), "parent": parent, "dirs": dirs})

    def _api_folders(self, qs: dict) -> None:
        root = Path(qs.get("root", [""])[0]).expanduser().resolve()
        if not root.is_dir():
            self._error(f"root is not a directory: {root}", 404)
            return
        with os.scandir(root) as it:
            subdirs = sorted((e for e in it if e.is_dir()), key=lambda e: e.name.lower())
        folders = [
            {"name": e.name, "count": _count_images(Path(e.path))} for e in subdirs
        ]
        self._send_json({"root": str(root), "folders": folders})

    def _api_check(self, qs: dict) -> None:
        """Validate a root/manifest pairing without shipping the image list.

        The UI calls this as soon as the manifest field changes, so the warning
        banner reacts immediately instead of waiting for the next full load.
        """
        raw_root = qs.get("root", [""])[0].strip()
        root = Path(raw_root).expanduser().resolve() if raw_root else None
        if root is None or not root.is_dir():
            self._error(f"root is not a directory: {raw_root}", 404)
            return
        manifest = _resolve_manifest(qs.get("manifest", [""])[0])
        index = _load_manifest_index(manifest)

        sub = qs.get("sub", [""])[0]
        names = None
        if sub:
            folder = (root / sub).resolve()
            if _is_within(folder, root) and folder.is_dir():
                names = _image_names(folder)
        self._send_json(
            {
                "root": str(root),
                "manifest": _manifest_status(root, manifest, index, names),
            }
        )

    def _api_images(self, qs: dict) -> None:
        root = Path(qs.get("root", [""])[0]).expanduser().resolve()
        sub = qs.get("sub", [""])[0]
        manifest_raw = qs.get("manifest", [""])[0].strip()
        folder = (root / sub).resolve()
        if not (_is_within(folder, root) and folder.is_dir()):
            self._error(f"not a folder inside root: {sub}", 404)
            return

        manifest = _resolve_manifest(manifest_raw)
        index = _load_manifest_index(manifest)

        names = _image_names(folder)
        images = []
        for name in names:
            rec = index.get(name, {})
            images.append(
                {
                    "file": name,
                    "meta": {
                        "routed_to": rec.get("routed_to"),
                        "manual_root": rec.get("manual_root"),
                        "raw_label": rec.get("raw_label"),
                        "raw_confidence": rec.get("raw_confidence"),
                        "mapped_label": rec.get("mapped_label"),
                        "llm_label": rec.get("llm_label"),
                        "source_pdf": rec.get("source_pdf"),
                        "page": rec.get("page"),
                        "width": rec.get("width"),
                        "height": rec.get("height"),
                        "in_manifest": name in index,
                    },
                }
            )
        state = _manifest_status(root, manifest, index, names)
        self._send_json(
            {
                "sub": sub,
                "images": images,
                "manifest": state,
                "manifest_found": state["exists"],
                "root_manifest_mismatch": state["status"] in ("outside", "no_match"),
                "warning": state["message"],
            }
        )

    def _serve_image(self, qs: dict) -> None:
        root = Path(qs.get("root", [""])[0]).expanduser().resolve()
        sub = qs.get("sub", [""])[0]
        file = qs.get("file", [""])[0]
        target = (root / sub / file).resolve()
        if not _is_within(target, root) or not target.is_file():
            self._error("image not found", 404)
            return
        if target.suffix.lower() not in IMAGE_EXTS:
            self._error("not an image", 400)
            return
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        # A crop never changes in place -- a commit moves it, which changes the
        # URL -- so caching is safe and makes stepping back and forth instant.
        self._send_bytes(
            target.read_bytes(), ctype, cache_control="private, max-age=3600"
        )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    global DEFAULT_ROOT, DEFAULT_MANIFEST

    repo = _repo_root()
    default_root = repo / "data" / "parsed" / "review"
    default_manifest = repo / "data" / "parsed" / "manifest.jsonl"

    ap = argparse.ArgumentParser(description="Figure labeling tool server.")
    ap.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="Folder that holds the class subfolders (default: data/parsed/review).",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest,
        help="Path to manifest.jsonl (default: data/parsed/manifest.jsonl).",
    )
    ap.add_argument("--port", type=int, default=8765, help="Port to listen on.")
    ap.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    ap.add_argument("--no-browser", action="store_true", help="Do not auto-open the browser.")
    args = ap.parse_args(argv)

    DEFAULT_ROOT = args.root.expanduser().resolve()
    DEFAULT_MANIFEST = args.manifest.expanduser().resolve()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Figure labeling tool running at {url}")
    print(f"  root folder : {DEFAULT_ROOT}")
    print(f"  manifest    : {DEFAULT_MANIFEST}")
    if _root_manifest_mismatch(DEFAULT_ROOT, DEFAULT_MANIFEST):
        print("  WARNING: the manifest is outside the root's folder -- commits here")
        print("           won't update it. Pass --manifest for this root's manifest.jsonl.")
    print("Press Ctrl+C to stop.")

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
