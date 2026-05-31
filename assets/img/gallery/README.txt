Adventure / album photos go in this folder.

Reference them from an album (Admin -> Collections -> Albums, or _albums/*.md) as
  gallery/<filename>
for the album cover and for each item in the photos list.

COPYRIGHT WATERMARK
-------------------
Every image placed in this folder gets "© <year> <author>" burned into the
bottom-left corner (and written to the file's Copyright metadata). This happens
automatically when you commit, via the git pre-commit hook.

  - One-time setup per clone:  git config core.hooksPath scripts/hooks
  - Requires Pillow:           python3 -m pip install --user Pillow
  - Run by hand any time:      python3 scripts/watermark_images.py
  - Re-stamp everything:       python3 scripts/watermark_images.py --force

The author comes from `author:` in _config.yml; override with --author / --year.
Already-stamped images are skipped (tracked in scripts/.watermark-manifest.json),
so running it repeatedly is safe.
