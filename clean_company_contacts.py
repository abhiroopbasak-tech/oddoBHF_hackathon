import argparse
import csv
import html as htmllib
import os
import re
from urllib.parse import urlparse

EMAIL_RE = re.compile(r"([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})")

# Common "role" mailboxes (useful for outreach; not personal)
GENERIC_LOCALPART_HINTS = {
    "info", "contact", "hello", "press", "media", "pr", "presse",
    "ir", "investor", "investors", "careers", "jobs", "support",
    "sales", "bizdev", "business", "partnership", "office"
}

# Robust-ish list for second-level public suffixes (small, practical subset)
SLL_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk",
    "com.au", "net.au", "org.au",
    "co.jp", "ne.jp",
    "com.br", "com.mx", "com.tr",
}

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def strip_tags(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<!--.*?-->", " ", s, flags=re.S)
    s = re.sub(r"<script.*?>.*?</script>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<style.*?>.*?</style>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = htmllib.unescape(s)
    return normalize_ws(s)

def get_netloc(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""

def registrable_domain(netloc: str) -> str:
    """Naive 'registrable domain' extraction without external deps."""
    netloc = (netloc or "").lower().strip()
    if not netloc:
        return ""
    parts = [p for p in netloc.split(".") if p]
    if len(parts) <= 2:
        return netloc
    last2 = ".".join(parts[-2:])
    last3 = ".".join(parts[-3:])
    # handle known second-level suffixes like co.uk
    if last2 in SLL_SUFFIXES and len(parts) >= 3:
        return ".".join(parts[-3:])
    if ".".join(parts[-2:]) in SLL_SUFFIXES and len(parts) >= 3:
        return ".".join(parts[-3:])
    # else: default last2
    return last2

def is_generic_email(email: str) -> bool:
    email = (email or "").strip().lower()
    if "@" not in email:
        return False
    local = email.split("@", 1)[0]
    # allow patterns like info, info-de, info_de, info1 etc
    local_base = re.split(r"[\.\-_0-9]+", local)[0]
    return local_base in GENERIC_LOCALPART_HINTS

def clean_email(raw_email: str) -> str:
    e = (raw_email or "").strip()
    if not e:
        return ""
    e = htmllib.unescape(e)
    e = e.replace("\\u003e", "").replace("u003e", "").replace("&gt;", "").replace(">", "")
    e = e.strip().strip('"').strip("'")
    e = e.replace("mailto:", "").strip()
    # remove stray punctuation around it
    e = e.strip(" ,;:()[]{}<>")
    # quick validation
    m = EMAIL_RE.search(e)
    return m.group(1) if m else ""

def decode_cfemail(cfhex: str) -> str:
    """
    Decode Cloudflare email obfuscation.
    cfhex is hex string from data-cfemail.
    """
    try:
        cfhex = cfhex.strip()
        if not re.fullmatch(r"[0-9a-fA-F]+", cfhex) or len(cfhex) < 4:
            return ""
        data = bytes.fromhex(cfhex)
        key = data[0]
        out = bytes([b ^ key for b in data[1:]])
        email = out.decode("utf-8", errors="ignore")
        email = clean_email(email)
        return email
    except Exception:
        return ""

def extract_emails_from_text(text: str) -> list[str]:
    text = text or ""
    # decode cfemail occurrences first
    emails = []
    for m in re.finditer(r'data-cfemail\s*=\s*"([0-9a-fA-F]+)"', text):
        dec = decode_cfemail(m.group(1))
        if dec:
            emails.append(dec)
    # then normal email regex
    for m in EMAIL_RE.finditer(text):
        emails.append(clean_email(m.group(1)))
    # dedupe in order
    seen = set()
    out = []
    for e in emails:
        el = e.lower()
        if el and el not in seen:
            seen.add(el)
            out.append(e)
    return out

def classify_contact_type(url: str, email: str, evidence: str) -> str:
    s = (url or "").lower() + " " + (email or "").lower() + " " + (evidence or "").lower()

    # strong IR indicators
    if any(k in s for k in ["investor-relations", "investor relations", "/ir", "/investors", "aktionaer", "shareholder", "finanzberichte", "annual report"]):
        return "investor_relations"

    # press/media
    if any(k in s for k in ["press", "presse", "media", "newsroom", "kommunikation", "pr "]):
        return "press"

    # careers
    if any(k in s for k in ["careers", "jobs", "stellen", "karriere", "recruit", "bewerb"]):
        return "careers"

    # support
    if any(k in s for k in ["support", "help", "hilfe", "faq", "service-center", "kundenservice"]):
        return "support"

    # b2b / partnerships / sales
    if any(k in s for k in ["sales", "partner", "partnership", "bizdev", "business", "enterprise", "b2b", "firmenkunden", "unternehmen"]):
        return "sales_partnership"

    # legal / impressum
    if any(k in s for k in ["impressum", "legal", "privacy", "datenschutz", "terms"]):
        return "legal"

    # generic contact
    if any(k in s for k in ["contact", "kontakt", "get in touch", "ansprechpartner"]):
        return "contact"

    return "other"

def score_row(contact_type: str, email: str, third_party: bool, generic: bool, contact_url: str, evidence_clean: str) -> int:
    score = 0
    if email:
        score += 5
    if email and not third_party:
        score += 4
    if third_party and email:
        score -= 6
    if generic and email:
        score += 2

    # type weighting
    type_bonus = {
        "investor_relations": 3,
        "press": 2,
        "sales_partnership": 2,
        "contact": 1,
        "legal": 0,
        "support": 0,
        "careers": 0,
        "other": 0,
    }
    score += type_bonus.get(contact_type, 0)

    # prefer short paths (often official)
    try:
        path_len = len((urlparse(contact_url).path or "").strip("/"))
        if path_len <= 30:
            score += 1
    except Exception:
        pass

    # evidence keyword
    if evidence_clean and any(k in evidence_clean.lower() for k in ["e-mail", "email", "mail", "kontakt", "contact"]):
        score += 1

    return score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="Path to company_contacts.csv")
    ap.add_argument("--outdir", default="out_clean", help="Output directory")
    ap.add_argument("--keep_third_party_emails", action="store_true", help="Keep third-party emails (default: remove)")
    ap.add_argument("--max_rows_per_company", type=int, default=50, help="Cap rows per company in clean output")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.infile, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    clean_rows = []
    # Track how many rows per company we keep
    kept_count = {}

    for r in raw_rows:
        company = normalize_ws(r.get("company", ""))
        company_website = normalize_ws(r.get("company_website", ""))
        contact_url = normalize_ws(r.get("contact_url", ""))
        raw_email = r.get("email", "") or ""
        evidence_raw = r.get("evidence", "") or ""
        source = normalize_ws(r.get("source", ""))

        if not company or not contact_url:
            continue

        company_netloc = get_netloc(company_website) or get_netloc(contact_url)
        company_regdom = registrable_domain(company_netloc)

        evidence_clean = strip_tags(evidence_raw)[:240]

        # Clean primary email field
        email = clean_email(raw_email)

        # If no email, try extracting from evidence
        extracted = []
        if not email:
            extracted = extract_emails_from_text(evidence_raw)
            if extracted:
                email = extracted[0]  # pick first

        email_domain = ""
        if email and "@" in email:
            email_domain = email.split("@", 1)[1].lower()

        email_regdom = registrable_domain(email_domain)
        third_party = bool(email and company_regdom and email_regdom and email_regdom != company_regdom)

        # If third-party emails are not wanted, drop the email but keep the contact path
        dropped_email = False
        if third_party and email and not args.keep_third_party_emails:
            email = ""
            email_domain = ""
            email_regdom = ""
            dropped_email = True
            third_party = False  # email removed, row is no longer "third-party email"

        generic = is_generic_email(email) if email else False

        contact_type = classify_contact_type(contact_url, email, evidence_clean)

        score = score_row(contact_type, email, third_party=False, generic=generic, contact_url=contact_url, evidence_clean=evidence_clean)

        row = {
            "company": company,
            "company_website": company_website,
            "company_domain": company_regdom,
            "contact_url": contact_url,
            "contact_type": contact_type,
            "email": email,
            "email_domain": email_regdom,
            "is_generic": str(generic),
            "dropped_third_party_email": str(dropped_email),
            "evidence_clean": evidence_clean,
            "source": source,
            "score": str(score),
        }

        # Per-company cap (prevents giant noisy dumps)
        c = kept_count.get(company, 0)
        if c >= args.max_rows_per_company:
            continue
        kept_count[company] = c + 1
        clean_rows.append(row)

    # Deduplicate: (company, contact_url, email)
    dedup = {}
    for r in clean_rows:
        key = (r["company"].lower(), r["contact_url"].lower(), (r["email"] or "").lower())
        dedup[key] = r
    clean_rows = list(dedup.values())

    # Build BEST contacts per company
    by_company = {}
    for r in clean_rows:
        by_company.setdefault(r["company"], []).append(r)

    best_rows = []
    for company, rows in by_company.items():
        # prefer rows that have an email; then by score
        rows_sorted = sorted(
            rows,
            key=lambda x: (
                0 if x.get("email") else 1,
                -int(x.get("score", "0")),
                len((urlparse(x.get("contact_url", "")).path or "")),
            )
        )
        best = rows_sorted[0]

        # include a few alternates (useful for marketing ops)
        alts = rows_sorted[1:4]
        alt_emails = [a["email"] for a in alts if a.get("email")]
        alt_urls = [a["contact_url"] for a in alts if a.get("contact_url")]

        best_rows.append({
            "company": company,
            "company_website": best.get("company_website", ""),
            "company_domain": best.get("company_domain", ""),
            "best_contact_type": best.get("contact_type", ""),
            "best_contact_email": best.get("email", ""),
            "best_contact_url": best.get("contact_url", ""),
            "best_score": best.get("score", "0"),
            "alt_emails": " | ".join(alt_emails),
            "alt_urls": " | ".join(alt_urls),
            "evidence_clean": best.get("evidence_clean", ""),
            "source": best.get("source", ""),
        })

    clean_path = os.path.join(args.outdir, "company_contacts_clean.csv")
    best_path = os.path.join(args.outdir, "company_best_contacts.csv")

    with open(clean_path, "w", encoding="utf-8", newline="") as f:
        fn = [
            "company","company_website","company_domain","contact_url","contact_type",
            "email","email_domain","is_generic","dropped_third_party_email",
            "evidence_clean","source","score"
        ]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for r in sorted(clean_rows, key=lambda x: (x["company"].lower(), -int(x["score"]))):
            w.writerow({k: r.get(k, "") for k in fn})

    with open(best_path, "w", encoding="utf-8", newline="") as f:
        fn = [
            "company","company_website","company_domain",
            "best_contact_type","best_contact_email","best_contact_url","best_score",
            "alt_emails","alt_urls","evidence_clean","source"
        ]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for r in sorted(best_rows, key=lambda x: x["company"].lower()):
            w.writerow({k: r.get(k, "") for k in fn})

    print("Done.")
    print("Wrote:")
    print(" -", clean_path)
    print(" -", best_path)
    print(f"Stats: raw_rows={len(raw_rows)} clean_rows={len(clean_rows)} companies={len(best_rows)}")

if __name__ == "__main__":
    main()
