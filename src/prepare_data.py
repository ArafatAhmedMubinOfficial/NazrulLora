#!/usr/bin/env python3
"""Build deduplicated Kazi Nazrul Islam + Basic Bangla datasets for 2-stage Colab training."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

from datasets import load_dataset
from transformers import AutoTokenizer

BASE_MODEL_ID = "unsloth/gemma-3-4b-it"

SYSTEM_PROMPT = (
    "আপনি জাতীয় কবি কাজী নজরুল ইসলামের সাহিত্যিক ভাবধারা, ছন্দ, ওজস্বী ভাষা ও সৃজনশীল শৈলীতে "
    "দক্ষ একজন বাংলা এআই লেখক। ব্যবহারকারীর প্রম্পট অনুযায়ী কাজী নজরুলের ভঙ্গিতে কবিতা, "
    "গান (নজরুলগীতি), গল্প, প্রবন্ধ, নাটক বা গদ্য রচনা করুন এবং সাধারণ বাংলা প্রশ্নেরও নির্ভুল উত্তর দিন।"
)

GENRE_BN = {
    "poem": "কবিতা",
    "song": "নজরুলগীতি",
    "novel": "উপন্যাস",
    "story": "ছোটগল্প",
    "essay": "প্রবন্ধ",
    "play": "নাটক",
    "letter": "পত্র",
    "prose": "গদ্য রচনা",
}


def normalize_text(text: str) -> str:
    """Normalize whitespace and line breaks while preserving Bangla punctuation and stanza breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_ekpatagolpo_text(text: str) -> str:
    """Strip Ekpatagolpo UI controls, arrow glyphs, and pagination footers anywhere in text."""
    text = re.sub(r"[⤶⤷⟲]+", " ", text)
    text = re.sub(r"A\s*[−\-–—]\s*A\s*\+", " ", text)
    text = re.sub(r"\b\d+\s+Mins?\s+Read\b", " ", text, flags=re.I)
    text = re.sub(r"(?:\s+\d{1,3}){3,}\s*$", "", text)
    return normalize_text(text)


def doc_fingerprint(text: str) -> str:
    """Content fingerprint ignoring punctuation/whitespace for deduplication."""
    stripped = re.sub(r"[\s।|,.;:!?\-\—\–\'\"‘’“”\(\)\[\]॥]+", "", text)
    return hashlib.md5(stripped[:350].encode("utf-8")).hexdigest()


def bangla_ratio(text: str) -> float:
    """Fraction of non-whitespace characters in the Bengali Unicode block."""
    non_ws = re.sub(r"\s+", "", text)
    if not non_ws:
        return 0.0
    bn = len(re.findall(r"[\u0980-\u09FF]", non_ws))
    return bn / len(non_ws)


def infer_ekpatagolpo_genre(title: str, url: str, text: str) -> str:
    """Infer literary genre from Ekpatagolpo title/URL/body cues."""
    combined = f"{title} {url}".lower()
    if "গানের মালা" in title or "নজরুল-গীতিকা" in title or "বুলবুল" in title or "সুর-সাকী" in title or "গীতি" in title:
        return "song"
    if "নাটক" in title or "পুতুলের বিয়ে" in title or "ঝিলিমিলি" in title or "আলেয়া" in title:
        return "play"
    if "প্রবন্ধ" in title or "যুগবাণী" in title or "রুদ্র-মঙ্গল" in title or "দুর্দিনের যাত্রী" in title or "রাজবন্দীর জবানবন্দী" in title:
        return "essay"
    if "পত্র" in title or "চিঠি" in title:
        return "letter"
    if "কুহেলিকা" in title or "বাঁধনহারা" in title or "বাঁধন-হারা" in title or "মৃত্যুক্ষুধা" in title:
        return "novel"
    if "ছোটো গল্প" in title or "ব্যথার দান" in title or "রিক্তের বেদন" in title or "শিউলি মালা" in title:
        return "story"
    if "রুবাইয়াত" in title or "হাফিজ" in title or "সর্বহারা" in title or "অগ্নি-বীণা" in title or "বিষের বাঁশী" in title:
        return "poem"
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines and sum(len(ln) for ln in lines) / len(lines) < 55:
        return "poem"
    return "prose"


def is_nazrul_ekpatagolpo(row: dict) -> bool:
    """Identify authentic Kazi Nazrul Islam writings while excluding biographies/about pages."""
    title = (row.get("title") or "").strip()
    author = (row.get("author") or "").strip()
    url = (row.get("url") or "").lower()
    if any(x in url for x in ["/tag/", "/category/", "/author/"]):
        return False
    if any(x in title for x in ["জীবনী", "পরিচিতি", "মূল্যায়ন", "স্মৃতিকথা", "প্রবন্ধাবলি – নজরুল বিষয়ক"]):
        return False
    if "নজরুল ইসলাম বাবু" in title or "দেওয়ান নজরুল" in title:
        return False
    if "কাজী নজরুল" in title or "কাজী নজরুল" in author or "kazi-nazrul" in url:
        return True
    if title.startswith(("নজরুল-গীতিকা", "নজরুল গীতিকা", "নজরুল প্রবন্ধসমগ্র")):
        return True
    return False


def collect_nazrul_corpus(cache_dir: Path) -> List[dict]:
    """Download, clean, and deduplicate all clean digital Kazi Nazrul Islam writings."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    seen: Dict[str, dict] = {}
    raw_counts = Counter()

    def register(source: str, title: str, text: str, kind: str) -> None:
        text = normalize_text(text)
        if len(text) < 45 or bangla_ratio(text) < 0.55:
            return
        raw_counts[source] += 1
        fp = doc_fingerprint(text)
        if fp not in seen or len(text) > len(seen[fp]["text"]):
            seen[fp] = {
                "source": source,
                "title": normalize_text(title),
                "text": text,
                "kind": kind,
            }

    # 1. Ekpatagolpo (plays, stories, essays, letters, songs, novels, poems)
    ek_cache = cache_dir / "ekpatagolpo_nazrul.jsonl"
    if ek_cache.exists():
        print(f"[data] Loading cached {ek_cache}", flush=True)
        with ek_cache.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                register("Ekpatagolpo", rec["title"], rec["text"], rec["kind"])
    else:
        print("[data] Streaming sayurio/ekpatagolpo-scrape-bangla-literature...", flush=True)
        ds_ek = load_dataset("sayurio/ekpatagolpo-scrape-bangla-literature", split="train", streaming=True)
        ek_rows = []
        for idx, row in enumerate(ds_ek, 1):
            if is_nazrul_ekpatagolpo(row):
                txt = clean_ekpatagolpo_text(row.get("content") or "")
                title = (row.get("title") or "").replace("– কাজী নজরুল ইসলাম", "").replace("- কাজী নজরুল ইসলাম", "").strip()
                kind = infer_ekpatagolpo_genre(row.get("title") or "", row.get("url") or "", txt)
                if len(txt) >= 60:
                    rec = {"title": title, "text": txt, "kind": kind}
                    ek_rows.append(rec)
                    register("Ekpatagolpo", title, txt, kind)
            if idx % 15000 == 0:
                print(f"  ekpatagolpo scanned {idx} rows ({len(ek_rows)} Nazrul docs)...", flush=True)
        with ek_cache.open("w", encoding="utf-8") as f:
            for r in ek_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  saved {len(ek_rows)} Ekpatagolpo Nazrul docs.", flush=True)

    # 2. Bangla Kobita Archive (sayurio/bangla-kobita-scrape-bangla-literature)
    kobita_file = cache_dir / "bangla_kobita.jsonl"
    if not kobita_file.exists():
        print("[data] Downloading sayurio/bangla-kobita-scrape-bangla-literature...", flush=True)
        urllib.request.urlretrieve(
            "https://huggingface.co/datasets/sayurio/bangla-kobita-scrape-bangla-literature/resolve/main/data.jsonl",
            kobita_file,
        )
    with kobita_file.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "kazinazrulislam" in line.lower():
                try:
                    obj = json.loads(line)
                    u = (obj.get("url") or "").lower().rstrip("/")
                    if "/kazinazrulislam/" in u and not u.endswith("/kazinazrulislam"):
                        title = (obj.get("title") or "").replace(" - কাজী নজরুল ইসলাম", "").strip()
                        register("BanglaKobita", title, obj.get("content") or "", "poem")
                except Exception:
                    continue

    # 3. Bengali-Poem-Dataset-JSON
    bpd_file = cache_dir / "Bengali-Poem-Dataset.json"
    if not bpd_file.exists():
        print("[data] Downloading Bengali-Poem-Dataset.json...", flush=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/mohammad-sheikh-shahinur-rahman/Bengali-Poem-Dataset-JSON/main/Bengali-Poem-Dataset.json",
            bpd_file,
        )
    bpd = json.loads(bpd_file.read_text(encoding="utf-8"))
    for r in bpd.get("records", []):
        if "নজরুল" in (r.get("poet") or ""):
            register("BengaliPoemJSON", r.get("title") or "", r.get("text") or "", "poem")

    # 4. rishiraj/bengali-poems
    print("[data] Loading rishiraj/bengali-poems...", flush=True)
    for r in load_dataset("rishiraj/bengali-poems", split="train"):
        if "নজরুল" in (r.get("poet") or ""):
            register("RishirajPoems", "", r.get("poem") or "", "poem")

    # 5. Shakil2448868/kazi-nazrul-islam-synthetic-prompt-bangla (clean song lyrics + titles)
    print("[data] Loading Shakil2448868 Nazrul songs...", flush=True)
    for r in load_dataset("Shakil2448868/kazi-nazrul-islam-synthetic-prompt-bangla", split="train"):
        register("ShakilSongs", r.get("Title") or "", r.get("Song") or "", "song")

    # 6. BanglaMusicStylo (Kazi Nazrul Islam folder)
    ms_dir = cache_dir / "music_stylo" / "Data" / "Kazi Nazrul Islam"
    if not ms_dir.exists():
        import subprocess
        rar_file = cache_dir / "music_stylo.rar"
        if not rar_file.exists():
            print("[data] Downloading BanglaMusicStylo...", flush=True)
            urllib.request.urlretrieve(
                "https://github.com/rafayetrafi/BanglaMusicStylo-A-Stylometric-Dataset-of-Bangla-Music-Lyrics/raw/master/BanglaStylometry%20Dataset.rar",
                rar_file,
            )
        subprocess.run(["apt-get", "-qq", "update"], check=False)
        subprocess.run(["apt-get", "-qq", "install", "-y", "unrar"], check=False)
        ms_extract = cache_dir / "music_stylo"
        ms_extract.mkdir(parents=True, exist_ok=True)
        subprocess.run(["unrar", "x", "-o+", "-idq", str(rar_file), str(ms_extract)], check=False)
    if ms_dir.exists():
        for fn in sorted(ms_dir.iterdir()):
            if fn.stat().st_size == 0:
                continue
            try:
                with zipfile.ZipFile(fn) as z:
                    root = ET.fromstring(z.read("word/document.xml"))
                    paras = [
                        "".join(n.text or "" for n in p.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")).strip()
                        for p in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p")
                    ]
                    txt = "\n".join(p for p in paras if p)
            except Exception:
                txt = fn.read_bytes().decode("utf-8", errors="ignore")
            title = re.sub(r"^song\d+_", "", fn.name).replace(".docx", "").strip()
            register("MusicStylo", title, txt, "song")

    # 7. NazrulRachanabali 3 novels (Kuhelika, Bandhanhara, Mrityukshudha)
    novels_cache = cache_dir / "nazrul_novels.jsonl"
    if novels_cache.exists():
        with novels_cache.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                register("NazrulNovels", rec["title"], rec["text"], "novel")
    else:
        print("[data] Fetching NazrulRachanabali novel chapters...", flush=True)
        req = urllib.request.Request(
            "https://api.github.com/repos/ashikurrahmanshaon/NazrulRachanabali/git/trees/HEAD?recursive=1",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        tree = json.loads(urllib.request.urlopen(req, timeout=20).read().decode())["tree"]
        novels = []
        for f in tree:
            p = f["path"]
            if p.startswith("app/src/main/assets/") and p.endswith(".html"):
                raw_url = "https://raw.githubusercontent.com/ashikurrahmanshaon/NazrulRachanabali/master/" + urllib.parse.quote(p)
                h = urllib.request.urlopen(raw_url, timeout=20).read().decode("utf-8", errors="replace")
                h = re.sub(r"<(br|p|div|h\d)[^>]*>", "\n", h, flags=re.I)
                txt = html.unescape(re.sub(r"<[^>]+>", "", h))
                txt = normalize_text(txt)
                title = p.split("/")[-1].replace(".html", "").replace("_", " ")
                novels.append({"title": title, "text": txt})
                register("NazrulNovels", title, txt, "novel")
        with novels_cache.open("w", encoding="utf-8") as f:
            for r in novels:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    docs = list(seen.values())
    print(f"[data] Raw counts per source: {dict(raw_counts)}", flush=True)
    print(f"[data] Unique deduplicated clean Nazrul docs: {len(docs)}", flush=True)
    return docs


def collect_basic_bangla_corpus(cache_dir: Path, max_wiki_docs: int = 1200, max_sahitya_docs: int = 450, max_poem_docs: int = 1000) -> List[dict]:
    """Collect clean general Bangla Wikipedia and classic literature documents for language grounding."""
    docs: List[dict] = []

    # 1. Clean Bangla Wikipedia (streaming first shard to stay fast)
    wiki_cache = cache_dir / "basic_wiki.jsonl"
    if wiki_cache.exists():
        with wiki_cache.open("r", encoding="utf-8") as f:
            for line in f:
                docs.append(json.loads(line))
    else:
        print("[data] Sampling clean Bangla Wikipedia articles...", flush=True)
        ds_wiki = load_dataset("sayurio/bangla-wikipedia", data_files=["jsonl/data_001.jsonl"], split="train", streaming=True)
        wiki_rows = []
        skip_prefixes = ("মিডিয়াউইকি:", "উইকিপিডিয়া:", "টেমপ্লেট:", "বিষয়শ্রেণী:", "চিত্র:", "সাহায্য:", "মডিউল:", "প্রবেশদ্বার:")
        for row in ds_wiki:
            title = (row.get("title") or "").strip()
            content = normalize_text(row.get("content") or "")
            if title.startswith(skip_prefixes) or len(content) < 350 or bangla_ratio(content) < 0.72:
                continue
            wiki_rows.append({"title": title, "text": content[:2500], "kind": "wiki"})
            if len(wiki_rows) >= max_wiki_docs:
                break
        with wiki_cache.open("w", encoding="utf-8") as f:
            for r in wiki_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        docs.extend(wiki_rows)

    # 2. Classic Bangla literature (barunsaha/bangla_sahitya)
    print("[data] Sampling barunsaha/bangla_sahitya...", flush=True)
    ds_sah = list(load_dataset("barunsaha/bangla_sahitya", split="train"))
    rng = random.Random(42)
    rng.shuffle(ds_sah)
    added_sah = 0
    for row in ds_sah:
        txt = normalize_text(row.get("text") or "")
        if len(txt) < 150 or bangla_ratio(txt) < 0.72:
            continue
        docs.append({"title": row.get("title") or "", "text": txt[:2500], "kind": "classic_bn"})
        added_sah += 1
        if added_sah >= max_sahitya_docs:
            break

    # 3. General Bengali poetry from Bengali-Poem-Dataset.json
    bpd_file = cache_dir / "Bengali-Poem-Dataset.json"
    if bpd_file.exists():
        bpd = json.loads(bpd_file.read_text(encoding="utf-8"))
        others = [r for r in bpd.get("records", []) if "নজরুল" not in (r.get("poet") or "") and len(r.get("text") or "") >= 120]
        rng.shuffle(others)
        for r in others[:max_poem_docs]:
            txt = normalize_text(r.get("text") or "")
            if bangla_ratio(txt) >= 0.72:
                docs.append({"title": r.get("title") or "", "text": txt[:2000], "kind": "classic_poem"})

    print(f"[data] Basic Bangla grounding docs: {len(docs)}", flush=True)
    return docs


def chunk_by_paragraphs(text: str, max_chars: int = 1050) -> List[str]:
    """Split long texts into coherent paragraph-aligned chunks for CPT and SFT excerpts."""
    if len(text) <= max_chars:
        return [text]
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) <= 1:
        paras = [ln.strip() for ln in text.splitlines() if ln.strip()]
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for p in paras:
        if len(p) > max_chars:
            # Split extra-long single paragraph by sentences/dandas
            sents = re.split(r"(?<=[।?!॥])\s+", p)
            for s in sents:
                if cur and cur_len + len(s) + 1 > max_chars:
                    chunks.append("\n".join(cur))
                    cur, cur_len = [s], len(s)
                else:
                    cur.append(s)
                    cur_len += len(s) + 1
            continue
        if cur and cur_len + len(p) + 2 > max_chars:
            chunks.append("\n\n".join(cur))
            cur, cur_len = [p], len(p)
        else:
            cur.append(p)
            cur_len += len(p) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


THEME_LEXICON = [
    (["বিদ্রোহ", "স্বাধীন", "শিকল", "কারা", "অন্যায়", "শোষ", "রক্ত", "অগ্নি", "জাগ", "যুদ্ধ", "রণ"], "অন্যায় ও পরাধীনতার বিরুদ্ধে বিদ্রোহ ও জাগরণ"),
    (["নারী", "মা ", "জননী", "রমনী", "লজ্জা", "সম্মান", "অপমান", "ভোগ", "দেহ", "পণ্য", "কলঙ্ক"], "নারীর মর্যাদা, সমাজের ভণ্ডামি ও নৈতিক অবক্ষয়"),
    (["সাম্য", "মানুষ", "কুলি", "মজুর", "গরিব", "দারিদ্র্য", "ক্ষুধা", "ধনী", "ভণ্ড", "পাপ"], "সাম্যবাদ, মানবতা ও সমাজের ভণ্ডামি"),
    (["তরুণ", "যৌবন", "নবীন", "উষা", "প্রভাত", "আশা", "পথিক", "চলো"], "তারুণ্যের শক্তি ও নবজাগরণ"),
    (["বর্ষা", "মেঘ", "বৃষ্টি", "বাদল", "ঝড়", "শ্রাবণ", "আষাঢ়", "বিজলি"], "বর্ষা, মেঘ, ঝড় ও প্রকৃতির রূপ"),
    (["প্রেম", "প্রিয়া", "বিরহ", "ভালোবাসা", "বাঁশি", "নয়ন", "আঁখি", "চাঁদ", "ফুল"], "প্রেম, বিরহ ও হৃদয়ের আকুতি"),
    (["আল্লাহ", "খোদা", "রসুল", "ঈদ", "শ্যামা", "কালী", "মসজিদ", "মন্দির", "স্রষ্টা"], "আধ্যাত্মিকতা, ভক্তি ও মানবধর্ম"),
]


def fetch_wikipedia_craft_articles(cache_dir: Path) -> List[dict]:
    """Fetch Bengali Wikipedia articles on Bengali meter, poetic ornaments, and Nazrul's craft."""
    craft_cache = cache_dir / "bangla_poetics_craft.jsonl"
    if craft_cache.exists():
        with craft_cache.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    print("[data] Fetching Bengali poetics & Nazrul literary craft articles...", flush=True)
    titles = [
        ("কাজী_নজরুল_ইসলাম", "কাজী নজরুল ইসলামের সাহিত্যিক শৈলী ও জীবন"),
        ("নজরুলগীতি", "নজরুলগীতির সুর, তাল ও রচনাশৈলী"),
        ("বিদ্রোহী_(কবিতা)", "বিদ্রোহী কবিতার রচনাকৌশল ও ছন্দ"),
        ("অলংকার_(সাহিত্য)", "বাংলা কাব্যে অলংকার, উপমা ও রূপকের ব্যবহার"),
        ("ছন্দ", "কবিতার ছন্দ ও মাত্রাবিন্যাস"),
        ("কবিতা", "কবিতার গঠন, মিল ও কাব্যকলা"),
        ("বাংলা_কবিতা", "বাংলা কবিতার ধারা ও ঐতিহ্য"),
    ]
    articles: List[dict] = []
    for page, desc in titles:
        u = "https://bn.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
            {"action": "parse", "page": page, "prop": "text", "redirects": 1, "format": "json"}
        )
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "NazrulResearchBot/1.0"})
            d = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
            html_str = d.get("parse", {}).get("text", {}).get("*", "")
            paras = re.findall(r"<p>(.*?)</p>", html_str, flags=re.S)
            clean_paras = []
            for p in paras:
                txt = re.sub(r"\[[^\]]{1,5}\]", "", html.unescape(re.sub(r"<[^>]+>", "", p))).strip()
                if len(txt) >= 50:
                    clean_paras.append(txt)
            full_txt = normalize_text("\n\n".join(clean_paras))
            if len(full_txt) >= 300:
                articles.append({"title": page.replace("_", " "), "desc": desc, "text": full_txt, "kind": "craft"})
        except Exception as e:
            print(f"  wiki craft warn ({page}): {e}", flush=True)
    with craft_cache.open("w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"[data] Collected {len(articles)} poetics & Nazrul craft articles.", flush=True)
    return articles


def _bangla_end_key(line: str) -> str:
    """Extract normalized trailing Bengali phonetic key (last 2 chars ignoring punctuation)."""
    cleaned = re.sub(r"[^ঀ-৿]", "", line)
    if len(cleaned) < 2:
        return ""
    return cleaned[-2:]


def extract_rhyme_pairs_hint(lines: List[str]) -> str:
    """Summarize actual end-rhyme ending syllables found in a rhymed stanza."""
    keys = [_bangla_end_key(ln) for ln in lines if ln.strip()]
    pairs = []
    for i in range(len(keys) - 1):
        if keys[i] and (keys[i] == keys[i + 1] or (i + 2 < len(keys) and keys[i] == keys[i + 2])):
            if keys[i] not in pairs:
                pairs.append(keys[i])
    if not pairs:
        return ""
    return " / ".join(f"-{k}" for k in pairs[:4])


def score_verse_rhyme(lines: List[str]) -> float:
    """Estimate fraction of lines participating in AABB or ABAB end-rhymes."""
    if len(lines) < 4:
        return 0.0
    keys = [_bangla_end_key(ln) for ln in lines]
    matched = 0
    for i, k in enumerate(keys):
        if not k:
            continue
        neighbors = [keys[j] for j in (i - 2, i - 1, i + 1, i + 2) if 0 <= j < len(keys) and j != i]
        if k in neighbors:
            matched += 1
    return matched / len(keys)


def infer_themes(text: str, title: str) -> List[str]:
    """Extract descriptive Bengali themes present in a poem or song."""
    combined = f"{title} {text}"
    found = [desc for kws, desc in THEME_LEXICON if any(kw in combined for kw in kws)]
    return found or ["জীবন, প্রকৃতি ও মানবিক অনুভূতি"]


def build_craft_sft_examples(craft_articles: List[dict]) -> List[dict]:
    """Build SFT Q&A pairs teaching Bengali poetics, meter, ornaments, and Nazrul's style."""
    examples: List[dict] = []
    for art in craft_articles:
        chunks = chunk_by_paragraphs(art["text"], max_chars=950)
        for i, ch in enumerate(chunks[:6]):
            if len(ch) < 120:
                continue
            if i == 0:
                q = f"{art['desc']} সম্পর্কে বিস্তারিত আলোচনা করো।"
            else:
                q = f"'{art['title']}' বিষয়ের একটি গুরুত্বপূর্ণ অংশ সহজ ও প্রামাণিকভাবে ব্যাখ্যা করো।"
            examples.append({"prompt": q, "response": ch, "domain": "poetics_craft"})
    return examples


def build_nazrul_sft_examples(docs: List[dict], seed: int = 42) -> List[dict]:
    """Generate clean, diverse prompt -> Nazrul response training pairs (rhyme & craft focused)."""
    rng = random.Random(seed)
    examples: List[dict] = []

    poem_prompts_bn = [
        "কাজী নজরুল ইসলামের রচনাশৈলীতে ছন্দ ও অন্ত্যমিল বজায় রেখে '{title}' শিরোনামে একটি কবিতা লেখো।",
        "বিদ্রোহী কবি কাজী নজরুলের ভঙ্গিতে মিলযুক্ত ছন্দে '{title}' নিয়ে একটি প্রাণবন্ত কবিতা রচনা করো।",
        "কাজী নজরুলের কাব্যিক ছন্দ ও ভাষায় নিচের বিষয় নিয়ে একটি ছন্দোবদ্ধ কবিতা লিখে দাও—বিষয়: {theme}।",
        "কাজী নজরুল ইসলামের ভঙ্গিতে একটি সমাজ-ভাবনা ও আবেগময় কবিতা লেখো যেখানে {theme} ফুটে উঠেছে। ছন্দ ও অন্ত্যমিল অবশ্যই রাখবে।",
    ]
    poem_prompts_en = [
        "Write a rhymed Bangla poem in the style of Kazi Nazrul Islam titled '{title}'.",
        "Compose a rhythmic Bengali poem in Kazi Nazrul Islam's voice about {theme}.",
    ]
    song_prompts_bn = [
        "কাজী নজরুল ইসলামের ভঙ্গিতে ছন্দ ও অন্ত্যমিলসহ '{title}' শীর্ষক একটি নজরুলগীতি রচনা করো।",
        "নজরুলগীতির সুরেলা ছন্দে {theme} নিয়ে একটি গান লেখো।",
        "কবি কাজী নজরুলের গানের ধাঁচে '{title}' নিয়ে একটি গীতি-রচনা তৈরি করো।",
    ]
    prose_prompts_bn = [
        "কাজী নজরুল ইসলামের গদ্যশৈলীতে '{title}' বিষয়ে একটি সাহিত্যিক রচনা লেখো।",
        "নজরুলের ওজস্বী ও আবেগঘন ভাষায় '{title}' নিয়ে একটি গদ্য অনুচ্ছেদ রচনা করো।",
    ]

    for doc in docs:
        title = doc["title"] or "ভাবনা"
        kind = doc["kind"]
        genre_label = GENRE_BN.get(kind, "সাহিত্য রচনা")
        chunks = chunk_by_paragraphs(doc["text"], max_chars=900)
        themes = infer_themes(doc["text"], title)
        theme_str = themes[0]

        first_chunk = chunks[0]
        if len(first_chunk) >= 80:
            if kind == "song":
                p = rng.choice(song_prompts_bn).format(title=title, theme=theme_str)
                examples.append({"prompt": p, "response": first_chunk, "domain": "nazrul_song"})
            elif kind == "poem":
                lines = [ln.strip() for ln in first_chunk.splitlines() if ln.strip()]
                rhyme = score_verse_rhyme(lines)
                rhyme_hint = extract_rhyme_pairs_hint(lines)
                repeats = 2 if rhyme >= 0.4 else 1
                for _ in range(repeats):
                    p = rng.choice(poem_prompts_bn + poem_prompts_en).format(title=title, theme=theme_str)
                    examples.append({"prompt": p, "response": first_chunk, "domain": "nazrul_poem"})
                if len(lines) >= 8 and rhyme >= 0.35:
                    stanza = "\n".join(lines[:10])
                    p2 = f"কাজী নজরুলের ভঙ্গিতে অন্ত্যমিল ও ছন্দ ঠিক রেখে নিচের বিষয়ে একটি সংক্ষিপ্ত কবিতা লেখো: {theme_str}।"
                    examples.append({"prompt": p2, "response": stanza, "domain": "nazrul_poem_rhymed"})
                    if rhyme_hint and rng.random() < 0.35:
                        craft_resp = (
                            f"[ছন্দ ও মিল-পরিকল্পনা: {theme_str} ভাবনা, অন্ত্যমিল ({rhyme_hint}), ওজস্বী ও গীতল ছন্দ]\n\n"
                            f"{stanza}"
                        )
                        p3 = f"কাজী নজরুলের লেখার কৌশল ও অন্ত্যমিল ({rhyme_hint}) ব্যবহার করে '{theme_str}' বিষয়ে একটি ছন্দোবদ্ধ কবিতা লেখো।"
                        examples.append({"prompt": p3, "response": craft_resp, "domain": "nazrul_craft_blueprint"})
            elif kind in ("essay", "story", "novel", "play", "letter", "prose"):
                p = f"কাজী নজরুল ইসলামের {genre_label} শৈলীতে '{title}' বিষয়ে একটি গদ্য লেখা রচনা করো।"
                if rng.random() < 0.4:
                    p = rng.choice(prose_prompts_bn).format(title=title)
                examples.append({"prompt": p, "response": first_chunk, "domain": f"nazrul_{kind}"})

        # 2. Line/passage continuation prompt (limit prose chunks so poems/songs dominate SFT)
        max_ch = 3 if kind in ("poem", "song") else 1
        for ch in chunks[:max_ch]:
            lines = [ln for ln in ch.splitlines() if ln.strip()]
            if len(lines) >= 4:
                split_idx = min(3, max(2, len(lines) // 3))
                prefix = "\n".join(lines[:split_idx])
                suffix = "\n".join(lines[split_idx:])
                if len(prefix) >= 25 and len(suffix) >= 60:
                    cont_prompt = rng.choice([
                        f"কাজী নজরুলের ছন্দ ও অন্ত্যমিল অনুসরণ করে নিচের লেখাটি এগিয়ে নিয়ে যাও:\n\n{prefix}",
                        f"নিচের পংক্তিগুলো কাজী নজরুল ইসলামের কাব্যধারায় মিল রেখে সম্পূর্ণ করো:\n\n{prefix}",
                        f"Continue this Bangla {genre_label} in Kazi Nazrul Islam's literary voice:\n\n{prefix}",
                    ])
                    examples.append({"prompt": cont_prompt, "response": suffix, "domain": f"nazrul_cont_{kind}"})
            elif len(ch) >= 260 and kind not in ("poem", "song"):
                sents = re.split(r"(?<=[।?!॥])\s+", ch)
                if len(sents) >= 4:
                    prefix = " ".join(sents[:2])
                    suffix = " ".join(sents[2:])
                    if len(suffix) >= 80:
                        cont_prompt = f"কাজী নজরুল ইসলামের গদ্যভঙ্গিতে নিচের অংশটির পরের অংশ লেখো:\n\n{prefix}"
                        examples.append({"prompt": cont_prompt, "response": suffix, "domain": f"nazrul_cont_{kind}"})

    # 3. Add factual literary & biographical QA pairs about Kazi Nazrul Islam
    nazrul_qa = [
        ("বাংলাদেশের জাতীয় কবি কে এবং তাঁর সাহিত্যকর্মের মূল সুর কী?",
         "বাংলাদেশের জাতীয় কবি কাজী নজরুল ইসলাম। তিনি 'বিদ্রোহী কবি' হিসেবে খ্যাত। তাঁর সাহিত্যকর্মের মূল সুর হলো অন্যায়, শোষণ ও পরাধীনতার বিরুদ্ধে বিদ্রোহ, সাম্য ও মানবতার জয়গান, এবং প্রেম, প্রকৃতি ও আধ্যাত্মিক সুরের অপূর্ব সমন্বয়।"),
        ("কাজী নজরুল ইসলামের জন্ম ও মৃত্যু কবে এবং কোথায়?",
         "কাজী নজরুল ইসলাম ১৮৯৯ সালের ২৪শে মে (১১ই জ্যৈষ্ঠ ১৩০৬ বঙ্গাব্দ) পশ্চিমবঙ্গের বর্ধমান জেলার চুরুলিয়া গ্রামে জন্মগ্রহণ করেন এবং ১৯৭৬ সালের ২৯শে আগস্ট ঢাকায় মৃত্যুবরণ করেন।"),
        ("কাজী নজরুল ইসলামের বিখ্যাত কয়েকটি কাব্যগ্রন্থের নাম বলো।",
         "কাজী নজরুল ইসলামের উল্লেখযোগ্য কাব্যগ্রন্থগুলোর মধ্যে রয়েছে অগ্নি-বীণা, বিষের বাঁশী, ভাঙ্গার গান, সাম্যবাদী, সর্বহারা, দোলনচাঁপা, ছায়ানট, ফণি-মনসা, সিন্ধু-হিন্দোল, চক্রবাক, সন্ধ্যা ও প্রলয়-শিখা।"),
        ("কাজী নজরুল ইসলামের তিনটি উপন্যাস কী কী?",
         "কাজী নজরুল ইসলামের তিনটি বিখ্যাত উপন্যাস হলো 'বাঁধন-হারা', 'মৃত্যুক্ষুধা' এবং 'কুহেলিকা'।"),
        ("নজরুলগীতি বলতে কী বোঝায়?",
         "কাজী নজরুল ইসলামের রচিত ও সুরারোপিত প্রায় চার হাজারেরও বেশি গান 'নজরুলগীতি' বা 'নজরুল সংগীত' নামে পরিচিত। এতে রাগ-রাগিণী, গজল, শ্যামাসংগীত, হামদ-নাত, দেশাত্মবোধক এবং গণসংগীতের অসাধারণ বৈচিত্র্য রয়েছে।"),
        ("কাজী নজরুল ইসলামের বিখ্যাত প্রবন্ধ ও গল্পগ্রন্থগুলোর নাম লেখো।",
         "নজরুলের উল্লেখযোগ্য প্রবন্ধগ্রন্থের মধ্যে রয়েছে যুগবাণী, রুদ্র-মঙ্গল, দুর্দিনের যাত্রী ও রাজবন্দীর জবানবন্দী। তাঁর বিখ্যাত গল্পগ্রন্থ হলো ব্যথার দান, রিক্তের বেদন এবং শিউলি মালা।"),
        ("Who is the national poet of Bangladesh?",
         "Kazi Nazrul Islam (1899–1976), widely celebrated as the Bidrohi Kobi (Rebel Poet), is the National Poet of Bangladesh. His poems, Nazrul Geeti songs, novels, and essays champion freedom, equality, justice, and human dignity."),
    ]
    for q, a in nazrul_qa:
        for _ in range(4):
            examples.append({"prompt": q, "response": a, "domain": "nazrul_qa"})

    rng.shuffle(examples)
    return examples


def collect_general_bangla_sft(max_examples: int = 2000, seed: int = 42) -> List[dict]:
    """Collect clean Bangla general instruction-following examples for reasoning grounding."""
    print("[data] Streaming BanglaLLM/bangla-alpaca-orca general instructions...", flush=True)
    ds = load_dataset("BanglaLLM/bangla-alpaca-orca", split="train", streaming=True)
    rng = random.Random(seed)
    out: List[dict] = []
    for idx, row in enumerate(ds):
        if idx > 25000 and len(out) >= max_examples:
            break
        # Subsample every few rows for topic diversity while streaming fast
        if idx % 3 != 0:
            continue
        inst = normalize_text(row.get("instruction") or "")
        inp = normalize_text(row.get("input") or "")
        resp = normalize_text(row.get("output") or "")
        if not inst or not resp or len(resp) < 40 or len(resp) > 1200 or len(inst) > 500:
            continue
        if bangla_ratio(inst + resp) < 0.65:
            continue
        prompt = f"{inst}\n\n{inp}" if inp else inst
        out.append({"prompt": prompt, "response": resp, "domain": "general_bangla"})
        if len(out) >= max_examples:
            break
    rng.shuffle(out)
    print(f"[data] General Bangla SFT examples: {len(out)}", flush=True)
    return out


def format_cpt_document(doc: dict, rng: random.Random) -> str:
    """Format a Stage 1 CPT document with optional genre/title conditioning."""
    if doc["kind"] in ("wiki", "classic_bn", "classic_poem", "craft"):
        if doc.get("title") and rng.random() < 0.5:
            return f"শিরোনাম: {doc['title']}\n\n{doc['text']}"
        return doc["text"]
    genre = GENRE_BN.get(doc["kind"], "সাহিত্য রচনা")
    if doc.get("title") and rng.random() < 0.65:
        return f"শিরোনাম: {doc['title']}\nধরন: {genre}\nরচনা: কাজী নজরুল ইসলাম\n\n{doc['text']}"
    return f"কাজী নজরুল ইসলাম — {genre}\n\n{doc['text']}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Nazrul + Basic Bangla training datasets")
    parser.add_argument("--cache-dir", type=Path, default=Path("/content/probe_data"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--nazrul-upsample", type=int, default=2)
    parser.add_argument("--skip-cpt", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    naz_docs = collect_nazrul_corpus(args.cache_dir)
    # Stratified train/val split by genre
    by_kind: Dict[str, List[dict]] = {}
    for d in naz_docs:
        by_kind.setdefault(d["kind"], []).append(d)
    naz_train_docs, naz_val_docs = [], []
    for kind, lst in sorted(by_kind.items()):
        rng.shuffle(lst)
        n_val = max(1, int(len(lst) * args.val_ratio))
        naz_val_docs.extend(lst[:n_val])
        naz_train_docs.extend(lst[n_val:])

    def write_jsonl(path: Path, rows: List[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cpt_train_records: List[dict] = []
    cpt_val_records: List[dict] = []
    basic_docs: List[dict] = []
    naz_cpt_tokens = 0
    basic_cpt_tokens = 0

    craft_articles = fetch_wikipedia_craft_articles(args.cache_dir)

    if not args.skip_cpt:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        basic_docs = collect_basic_bangla_corpus(args.cache_dir) + craft_articles

        for _ in range(args.nazrul_upsample):
            for d in naz_train_docs:
                for ch in chunk_by_paragraphs(d["text"], max_chars=1350):
                    chunk_doc = {**d, "text": ch}
                    txt = format_cpt_document(chunk_doc, rng)
                    n_tok = len(tok.encode(txt, add_special_tokens=False))
                    naz_cpt_tokens += n_tok
                    cpt_train_records.append({"text": txt, "group": f"nazrul_{d['kind']}", "tokens": n_tok})

        for d in basic_docs:
            for ch in chunk_by_paragraphs(d["text"], max_chars=1350):
                chunk_doc = {**d, "text": ch}
                txt = format_cpt_document(chunk_doc, rng)
                n_tok = len(tok.encode(txt, add_special_tokens=False))
                basic_cpt_tokens += n_tok
                cpt_train_records.append({"text": txt, "group": d["kind"], "tokens": n_tok})

        rng.shuffle(cpt_train_records)

        for d in naz_val_docs:
            for ch in chunk_by_paragraphs(d["text"], max_chars=1350):
                txt = f"শিরোনাম: {d['title']}\nধরন: {GENRE_BN.get(d['kind'], 'রচনা')}\nরচনা: কাজী নজরুল ইসলাম\n\n{ch}" if d["title"] else ch
                cpt_val_records.append({"text": txt, "group": f"nazrul_{d['kind']}"})

        write_jsonl(args.out_dir / "cpt_train.jsonl", cpt_train_records)
        write_jsonl(args.out_dir / "cpt_val.jsonl", cpt_val_records)

    # Build Stage 2 SFT pairs (poetry, craft guides, rhyme blueprints + general Bangla)
    naz_sft_train = build_nazrul_sft_examples(naz_train_docs, seed=args.seed)
    craft_sft = build_craft_sft_examples(craft_articles)
    naz_sft_val = build_nazrul_sft_examples(naz_val_docs, seed=args.seed + 1)
    gen_sft = collect_general_bangla_sft(max_examples=1500, seed=args.seed)

    sft_train = naz_sft_train + craft_sft * 3 + gen_sft
    rng.shuffle(sft_train)
    sft_val = naz_sft_val[:200]

    write_jsonl(args.out_dir / "sft_train.jsonl", sft_train)
    write_jsonl(args.out_dir / "sft_val.jsonl", sft_val)

    genre_stats = {}
    for k, lst in sorted(by_kind.items()):
        genre_stats[k] = {
            "docs": len(lst),
            "chars": sum(len(x["text"]) for x in lst),
        }

    stats = {
        "unique_nazrul_docs": len(naz_docs),
        "nazrul_train_docs": len(naz_train_docs),
        "nazrul_val_docs": len(naz_val_docs),
        "nazrul_genres": genre_stats,
        "basic_bangla_docs": len(basic_docs),
        "cpt_train_records": len(cpt_train_records),
        "cpt_nazrul_tokens_total": naz_cpt_tokens,
        "cpt_basic_bangla_tokens_total": basic_cpt_tokens,
        "sft_train_nazrul_pairs": len(naz_sft_train),
        "sft_train_general_bangla_pairs": len(gen_sft),
        "sft_val_pairs": len(sft_val),
    }
    (args.out_dir / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[data] Dataset summary:", json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
