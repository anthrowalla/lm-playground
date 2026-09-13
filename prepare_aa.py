"""Extract American Anthropologist (JSTOR TEI XML) into trainable text.

One document per article div (types selectable via --types; MIS front
matter is excluded by default): a one-line metadata header, then the
article's pages concatenated with OCR furniture removed:

- [[START/END id]] markers and form feeds
- running heads: global masthead/section lines (corpus-frequency
  threshold) and per-article repeated all-caps lines
- page-number / roman-numeral-only lines
- reprint boilerplate (Kraus etc.)
- line wrapping (blank line = paragraph boundary) and end-of-line
  hyphenation (merge if the merged word is in the system dictionary,
  else keep the hyphen)

Output is plain text in the same <|bos|>...<|eos|> document format as
prepare_ethno.py, ready to be mixed into a corpus build.
"""

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

BOS, EOS = "<|bos|>", "<|eos|>"

HEADER_FIELDS = ["Title", "Author", "Published", "Volume", "Type"]
TYPE_LABEL = {"FLA": "Article", "BRV": "Book review", "NWS": "News", "EDI": "Editorial", "MIS": "Front matter"}

MARKER = re.compile(r"\[\[(START|END)[^\]]*\]\]")
INVALID_XML_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
PAGE_NO = re.compile(r"^[-–—\s]*([0-9]{1,4}|[ivxlcdm]{1,8})[-–—\s]*$", re.I)
BOILERPLATE = re.compile(
    r"KRAUS REPRINT|Reprinted by permission of the original publishers"
    r"|UNIVERSITY MICROFILMS|Reprinted with permission",
    re.I,
)
WS = re.compile(r"[ \t]+")


def norm_line(line: str) -> str:
    return WS.sub(" ", line.strip()).rstrip(".").upper()


def is_caps(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    return len(line) > 3 and len(letters) > 2 and sum(c.isupper() for c in letters) / len(letters) > 0.9


def load_dictionary() -> set[str]:
    words = set()
    for path in [Path("/usr/share/dict/american-english"), Path("/usr/share/dict/words")]:
        if path.exists():
            words |= {w.strip().lower() for w in path.read_text(encoding="utf-8", errors="ignore").splitlines() if w.strip()}
    return words


def join_hyphen(left: str, right: str, dictionary: set[str]) -> str:
    """Return the separator to use between a line ending in '-' and the next.

    'over-' + 'lapped' merges; 'well-' + 'known' keeps its hyphen. The
    system dictionary arbitrates; default is to merge.
    """
    m = re.search(r"([A-Za-z]+)-$", left)
    if m:
        first = re.match(r"([A-Za-z]+)\b", right)
        if first:
            stem, tail = m.group(1), first.group(1)
            if stem.lower() + tail.lower() in dictionary:
                return ""
            if (stem + "-" + tail).lower() in dictionary:
                return "-"
            return ""
    return " "


def clean_page(raw: str) -> list[str]:
    """One page's CDATA -> list of paragraph strings (not reflowed yet)."""
    raw = MARKER.sub("", raw).replace("\x0c", "\n")
    return [WS.sub(" ", ln.strip()) for ln in raw.splitlines()]


def iter_article_pages(elem):
    """Yield (article_div, pageList_div) pairs.

    Articles are divs with a direct bibl child, each followed by its
    pageList sibling. Review sections wrap the same pattern inside
    div type="group" containers, hence the recursion.
    """
    pending = None
    for child in elem:
        if child.find("bibl") is not None:
            pending = child
            continue
        if child.get("type") == "pageList" and pending is not None:
            yield pending, child
            pending = None
            continue
        yield from iter_article_pages(child)


def extract_articles(xml_path: Path, keep_types: set[str]):
    # JSTOR's CDATA embeds form feeds and other control chars that are
    # invalid in XML 1.0; map them to newlines before parsing.
    raw = INVALID_XML_CHARS.sub("\n", xml_path.read_text(encoding="utf-8"))
    root = ET.fromstring(raw)
    date = vol = None
    for idno in root.iter("idno"):
        if idno.get("type") == "volume" and vol is None:
            vol = (idno.text or "").strip()
    for d in root.iter("date"):
        if d.get("type") == "jstor" and date is None:
            date = (d.text or "").strip()  # YYYYMM
    if date and len(date) == 6:
        published = f"{date[:4]}-{date[4:]}"
    else:
        published = date or ""

    for body in root.iter("body"):
        for div, page_list in iter_article_pages(body):
            dtype = div.get("type")
            if dtype not in keep_types:
                continue
            title = author = ""
            bibl = div.find("bibl")
            t = bibl.find("title")
            if t is not None and t.text:
                title = WS.sub(" ", t.text.strip())
            a = bibl.find("author")
            if a is not None and a.text:
                author = WS.sub(" ", a.text.strip())
            pages = ["".join(p.itertext()) for p in page_list if p.get("type") == "page"]
            pages = [p for p in pages if p.strip()]
            if not pages:
                continue
            fields = {"Title": title, "Author": author, "Published": published,
                      "Volume": vol or "", "Type": TYPE_LABEL.get(dtype, dtype)}
            yield fields, pages


def furniture_filters(all_pages: list[list[str]]) -> set[str]:
    """Lines that recur as running heads across the corpus."""
    counts: dict[str, int] = {}
    for page in all_pages:
        for line in page:
            if is_caps(line):
                key = norm_line(line)
                counts[key] = counts.get(key, 0) + 1
    return {key for key, n in counts.items() if n >= 150}


def article_running_heads(pages: list[list[str]]) -> set[str]:
    n = len(pages)
    if n < 4:
        return set()
    counts: dict[str, int] = {}
    for page in pages:
        for line in page[:4]:
            if is_caps(line):
                counts[norm_line(line)] = counts.get(norm_line(line), 0) + 1
    return {key for key, c in counts.items() if c >= max(3, int(0.6 * n))}


def build_document(fields: dict, pages: list[list[str]], global_drop: set[str], dictionary: set[str]) -> str:
    art_drop = article_running_heads(pages)
    paragraphs: list[str] = []
    cur: list[str] = []

    def flush():
        if cur:
            paragraphs.append(" ".join(cur))
            cur.clear()

    for page in pages:
        pending_drop = True  # first caps line(s) of a page may be the running head block
        for line in page:
            line = line.strip()
            if not line:
                flush()
                pending_drop = True
                continue
            key = norm_line(line)
            if (
                key in global_drop
                or key in art_drop
                or PAGE_NO.match(line)
                or BOILERPLATE.search(line)
                or (pending_drop and is_caps(line) and len(line) < 60)
            ):
                continue
            pending_drop = False
            if cur:
                sep = join_hyphen(cur[-1], line, dictionary)
                if sep == " ":
                    cur.append(line)
                else:
                    cur[-1] = re.sub(r"-*$", "", cur[-1]) + ("-" if sep == "-" else "") + line
            else:
                cur.append(line)
    flush()

    paras = [WS.sub(" ", p).strip() for p in paragraphs]
    paras = [p for p in paras if p]
    header = " | ".join(f"{label}: {fields[label]}" for label in HEADER_FIELDS if fields.get(label))
    return f"{BOS}{header}\n\n" + "\n\n".join(paras) + f"\n{EOS}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml-dir", required=True, help="directory of JSTOR TEI issue XML files")
    parser.add_argument("--out", required=True, help="output corpus text file")
    parser.add_argument("--types", default="FLA,BRV,NWS,EDI", help="article div types to keep (comma-separated)")
    args = parser.parse_args()

    keep_types = {t.strip().upper() for t in args.types.split(",") if t.strip()}
    files = sorted(Path(args.xml_dir).glob("*.xml"))
    dictionary = load_dictionary()

    parsed = []
    all_pages: list[list[str]] = []
    for path in files:
        for fields, pages in extract_articles(path, keep_types):
            cleaned = [clean_page(p) for p in pages]
            parsed.append((fields, cleaned))
            all_pages.extend(cleaned)
    print(f"articles: {len(parsed):,} from {len(files)} issues")

    global_drop = furniture_filters(all_pages)
    print(f"global running-head lines dropped: {len(global_drop)}")

    n_chars = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for fields, pages in parsed:
            doc = build_document(fields, pages, global_drop, dictionary)
            f.write(doc)
            n_chars += len(doc)
    print(f"corpus: {n_chars:,} chars -> {args.out}")


if __name__ == "__main__":
    main()
