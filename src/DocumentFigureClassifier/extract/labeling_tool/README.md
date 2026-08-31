# Figure Labeling Tool

A browser-based tool for reviewing the cropped figures produced by
`extract_and_classify.py` and moving the mis-sorted ones into the right class
folder. It's a single-page HTML app served by a tiny, **zero-dependency**
Python server (stdlib only — no extra installs).

## What it does

- Pick a **root folder** that holds the class subfolders (default:
  `data/parsed/review`).
- Pick a **subfolder** to review (e.g. `bar` or `_review`).
- Optionally set a **confidence filter** (the trailing number in each filename,
  e.g. `…__table__0.61.png`) to show only images at or below a threshold — handy
  for focusing on the low-confidence cases the model was least sure about. The
  threshold is remembered across sessions.
- Flip through every image with the **arrow keys**. Leave an image alone to keep
  it where it is, or **click a destination folder** (or press its number key) to
  reassign it and jump to the next one.
- The destination buttons are **drag-to-reorder** — put the likely targets on
  top for the folder you're reviewing (e.g. `bar_stacked` / `bar_grouped` when
  reviewing `bar`). The order is remembered per subfolder.
- Each folder's **keyboard shortcut is rebindable**: click the key badge on a
  button and press the letter/number you want. Bindings are remembered per root
  and stay with the folder even after you reorder. Folders without a custom key
  fall back to their 1–9 position.
- **Undo** the last reassignment with `Ctrl+Z` (or the Undo button) — it reverts
  the change and jumps back to that image, the same as "keeping" it.
- **Nothing on disk changes until you press Commit.** Then the reassigned files
  are moved to their new folders and each moved file's record in
  `manifest.jsonl` gets a **`manual_root`** attribute set to the new folder.
  The original `routed_to` is left intact as provenance.

## Run it

From the project root:

```bash
uv run python -m DocumentFigureClassifier.extract.labeling_tool
```

Or run the script directly (no install needed):

```bash
python src/DocumentFigureClassifier/extract/labeling_tool/server.py
```

It opens `http://localhost:8765/` automatically. The root and manifest default
to `data/parsed/review` and `data/parsed/manifest.jsonl`; override with flags:

```bash
uv run python -m DocumentFigureClassifier.extract.labeling_tool --root data/parsed/review --manifest data/parsed/manifest.jsonl --port 8765
```

Flags: `--root`, `--manifest`, `--port` (default 8765), `--host` (default
`127.0.0.1`), `--no-browser`.

## Keyboard shortcuts

| Key | Action |
| --- | --- |
| `←` / `→` | previous / next image (keep = do nothing) |
| `Space` | next image |
| `1`…`9` / bound letters | reassign to that folder, then advance |
| `Ctrl`+`Z` | undo the last change and jump back to that image |
| `Backspace` / `Delete` | clear this image's pending assignment |
| `B` | cycle the preview background (checker / light / dark) |

Clicking a folder that's already assigned to the current image toggles the
assignment off. To change a folder's shortcut, click its key badge and press a
letter or number (`Backspace` clears it, `Esc` cancels).

## How files are matched to the manifest

Records are matched to files by the `filename` field (the crop's basename, which
is globally unique). On commit the tool:

1. Moves `root/<from>/<file>` → `root/<to>/<file>` (skips, never overwrites, if a
   file of that name already exists in the destination).
2. Rewrites `manifest.jsonl` atomically, adding `"manual_root": "<to>"` to the
   matching records and leaving every other line byte-for-byte unchanged.

Files are not renamed when moved, so their manifest link stays intact even
though the filename still encodes the *original* label. Use the folder the file
now lives in (or its `manual_root`) as the corrected ground truth.

## Notes

- The manifest normally sits one level **above** the class folders
  (`data/parsed/manifest.jsonl` vs `data/parsed/review/`); the tool handles that
  because the manifest path is configured independently of the root.
- Because root and manifest are configured independently, it's possible to pair
  a root with a manifest from a **different** dataset (e.g. a `data/parsed_new`
  root with the `data/parsed` manifest). Then the moved filenames match no
  records and `manual_root` is silently never written. The tool now warns about
  this — on startup, in the status line when you load a folder, in the commit
  confirmation, and in the commit report — whenever the manifest lives outside
  the root's own dataset folder.
- If the manifest can't be found, moves still happen — only the `manual_root`
  write is skipped, and the commit report says so.
- It's a local, single-user tool: it binds to localhost and reads/writes files
  under the root you choose.
