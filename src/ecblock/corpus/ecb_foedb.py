"""Client for the ECB 'FoeDB' publications database.

The ECB press site (pubbydate) is a JS app backed by a static sharded JSON DB at
/foedb/dbs/foedb/publications.en/. It indexes EVERY ECB publication (~20k records:
speeches, interviews, press-conference statements, monetary policy accounts, blog
posts, working papers, press releases, ...).

Layout (reverse-engineered from foedb.min.js):
  versions.json                         -> [{"version","hash"}]
  {version}/{hash}/metadata.json        -> {header, total_records, chunk_size, ...}
  {version}/{hash}/data/0/chunk_{n}.json-> flat array of (chunk_size * len(header)) values

Each record's documentTypes[0] is the relative URL of the publication; its path
prefix identifies the category, e.g.:
  /press/inter/...     interviews (incl. media interviews republished by the ECB)
  /press/key/...       speeches by Executive Board members
  /press/accounts/...  monetary policy accounts (minutes)
  /press/pressconf/... press-conference pages (statement + Q&A)
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

from ..config import RAW

HOST = "https://www.ecb.europa.eu"
DB_BASE = f"{HOST}/foedb/dbs/foedb/publications.en"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
CACHE = RAW / "foedb"


@dataclass
class PubRecord:
    id: int
    date: str
    year: int
    type_id: int
    boardmember: str | None
    authors: str | None
    url: str          # absolute URL to the .en.html publication
    title: str = ""   # publicationProperties.Title (clean, e.g. "Interview with Nikkei")

    @property
    def path(self) -> str:
        return self.url.replace(HOST, "")


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = UA
    return s


def _get(sess, url, *, as_json=True, retries=4):
    for k in range(retries):
        try:
            r = sess.get(url, timeout=60)
            if r.status_code == 200:
                return r.json() if as_json else r.text
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 ** k)
    return None


def fetch_records(use_cache: bool = True) -> list[PubRecord]:
    CACHE.mkdir(parents=True, exist_ok=True)
    sess = _session()

    ver = _get(sess, f"{DB_BASE}/versions.json")
    version, vhash = ver[0]["version"], ver[0]["hash"]
    base = f"{DB_BASE}/{version}/{vhash}"

    meta = _get(sess, f"{base}/metadata.json")
    header = meta["header"]
    total = meta["total_records"]
    chunk_size = meta["chunk_size"]
    group_size = meta.get("chunk_group_size", 1000)
    n_fields = len(header)
    n_chunks = -(-total // chunk_size)  # ceil

    idx = {f: i for i, f in enumerate(header)}
    out: list[PubRecord] = []
    for c in range(n_chunks):
        # all chunks shard into data/{group}/ where group counts chunks per dir
        group = c // group_size
        cache_f = CACHE / f"chunk_{c}.json"
        if use_cache and cache_f.exists():
            flat = json.loads(cache_f.read_text(encoding="utf-8"))
        else:
            flat = _get(sess, f"{base}/data/{group}/chunk_{c}.json")
            if flat is None:
                continue
            cache_f.write_text(json.dumps(flat), encoding="utf-8")
        for i in range(0, len(flat), n_fields):
            row = flat[i:i + n_fields]
            if len(row) < n_fields:
                continue
            docs = row[idx["documentTypes"]]
            url = None
            if isinstance(docs, list) and docs:
                # prefer the .en.html publication page
                html = [d for d in docs if isinstance(d, str) and d.endswith(".en.html")]
                url = (html[0] if html else docs[0])
            if not url:
                continue
            ts = row[idx["pub_timestamp"]]
            d = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d") if ts else ""
            props = row[idx["publicationProperties"]]
            title = props.get("Title", "") if isinstance(props, dict) else ""
            out.append(PubRecord(
                id=row[idx["id"]],
                date=d,
                year=row[idx["year"]],
                type_id=row[idx["type"]],
                boardmember=row[idx["boardmember"]],
                authors=row[idx["Authors"]],
                url=url if url.startswith("http") else HOST + url,
                title=title.strip() if isinstance(title, str) else "",
            ))
    return out


def filter_by_path(records: list[PubRecord], prefix: str) -> list[PubRecord]:
    return [r for r in records if r.path.startswith(prefix)]


# ---- text extraction from an ECB publication HTML page ----
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def extract_text(html: str) -> str:
    m = re.search(r"<main\b.*?</main>", html, re.S | re.I)
    seg = m.group(0) if m else html
    seg = re.sub(r"<(script|style|nav|footer|header)\b.*?</\1>", " ", seg, flags=re.S | re.I)
    # drop related/share boxes by class is hard generically; rely on <main> scope
    txt = _TAG.sub(" ", seg)
    import html as _h
    txt = _h.unescape(txt)
    return _WS.sub(" ", txt).strip()


def fetch_text(url: str, sess: requests.Session | None = None) -> str:
    sess = sess or _session()
    html = _get(sess, url, as_json=False)
    return extract_text(html) if html else ""


if __name__ == "__main__":
    import collections
    recs = fetch_records()
    print(f"Total publications indexed: {len(recs)}")
    cats = collections.Counter(r.path.split("/")[2] if r.path.startswith("/press/")
                               else r.path.split("/")[1] for r in recs)
    print("Top path categories:", cats.most_common(15))
    inter = filter_by_path(recs, "/press/inter/")
    print(f"\nInterviews: {len(inter)}  ({min(r.date for r in inter)} .. {max(r.date for r in inter)})")
    for r in inter[:6]:
        print(" ", r.date, "|", r.boardmember, "|", r.path)
