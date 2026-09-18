"""Fetch general-domain text for the v6 mixed corpus.

Two sources, both written to gitignored data/general/:

1. FineWeb-Edu sample shards (HuggingFaceFW/fineweb-edu, sample/10BT split).
   Parquet downloads via huggingface_hub; we keep the raw shards and let
   prepare_mix.py read them with pyarrow.

2. Project Gutenberg classics for the specific domains requested:
   philosophy, logic, mathematics-adjacent, sociology. Each title is
   HTTP-verified (200 + Gutenberg markers) and license header/footer is
   stripped; failures are logged and skipped, never fatal.

Usage:
    .venv/bin/python scripts/fetch_general.py --fineweb-shards 4 \
        --out data/general
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "HuggingFaceFW/fineweb-edu"
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"

# (gutenberg id, label) — philosophy / logic / sociology / math-adjacent.
# Not all ids are certain; the fetcher verifies and skips failures.
CLASSICS = [
    (1497, "Plato - Republic"),
    (2680, "Marcus Aurelius - Meditations"),
    (3207, "Hobbes - Leviathan"),
    (6762, "Spinoza - Ethics"),
    (1232, "Machiavelli - The Prince"),
    (12242, "Mill - On Liberty"),
    (11224, "Mill - Utilitarianism"),
    (4363, "Nietzsche - Beyond Good and Evil"),
    (1998, "Nietzsche - Thus Spake Zarathustra"),
    (3600, "Montaigne - Essays"),
    (61, "Marx & Engels - Communist Manifesto"),
    (9664, "Hume - Enquiry Concerning Human Understanding"),
    (5827, "Russell - The Problems of Philosophy"),
    (13846, "Descartes - Discourse on Method"),
    (10615, "Locke - Essay Concerning Human Understanding"),
    (3300, "Adam Smith - Wealth of Nations"),
    (33283, "Abbott - Flatland"),
    (4280, "Kant - Critique of Pure Reason"),
]

START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n",
                      re.IGNORECASE)
END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.IGNORECASE)


def strip_gutenberg(text: str) -> str | None:
    """Keep only the work proper (between the START/END markers)."""
    start = START_RE.search(text)
    if not start:
        return None
    body = text[start.end():]
    end = END_RE.search(body)
    if end:
        body = body[:end.start()]
    return body.strip() + "\n"


def fetch_classics(out_dir: Path) -> tuple[list[str], list[str]]:
    classics_dir = out_dir / "classics"
    classics_dir.mkdir(parents=True, exist_ok=True)
    fetched, skipped = [], []
    for gid, label in CLASSICS:
        dst = classics_dir / f"pg{gid}.txt"
        if dst.exists() and dst.stat().st_size > 10_000:
            fetched.append(label)
            continue
        url = GUTENBERG_URL.format(gid=gid)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "lm-playground-corpus/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}")
                raw = r.read().decode("utf-8", errors="replace")
            body = strip_gutenberg(raw)
            if body is None or len(body) < 10_000:
                raise RuntimeError("no Gutenberg markers or too short")
            dst.write_text(body, encoding="utf-8")
            fetched.append(label)
            print(f"  ok  pg{gid:<6} {label} ({len(body)/1e6:.1f} MB)")
        except Exception as e:  # noqa: BLE001 - graceful skip is the contract
            skipped.append(f"pg{gid} {label}: {e}")
            print(f"  SKIP pg{gid:<6} {label}: {e}")
    return fetched, skipped


def fetch_fineweb(n_shards: int, out_dir: Path) -> list[str]:
    fw_dir = out_dir / "fineweb"
    fw_dir.mkdir(parents=True, exist_ok=True)
    files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
             if f.startswith("sample/10BT/") and f.endswith(".parquet")]
    files.sort()
    if len(files) < n_shards:
        raise SystemExit(f"only {len(files)} shards listed in {REPO} sample/10BT")
    paths = []
    for f in files[:n_shards]:
        local = hf_hub_download(REPO, f, repo_type="dataset", local_dir=fw_dir)
        paths.append(local)
        print(f"  ok  {f}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fineweb-shards", type=int, default=4)
    parser.add_argument("--out", default="data/general")
    parser.add_argument("--skip-fineweb", action="store_true")
    parser.add_argument("--skip-classics", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet = []
    if not args.skip_fineweb:
        print("FineWeb-Edu sample shards:")
        parquet = fetch_fineweb(args.fineweb_shards, out_dir)
    fetched = skipped = []
    if not args.skip_classics:
        print("Gutenberg classics:")
        fetched, skipped = fetch_classics(out_dir)

    (out_dir / "fetch_report.json").write_text(json.dumps({
        "fineweb_shards": parquet,
        "classics_fetched": fetched,
        "classics_skipped": skipped,
    }, indent=2))
    print(f"done: {len(parquet)} fineweb shards, {len(fetched)} classics ok, "
          f"{len(skipped)} skipped")


if __name__ == "__main__":
    main()
