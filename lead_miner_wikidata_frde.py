# lead_miner_wikidata_frde.py
# Purpose: Build a large, clean lead list (FR+DE founders/CEOs) from Wikidata,
#          add actionable contact paths from company websites (robots-respecting),
#          generate clusters + graph outputs (CSV + JSON + GraphML for Gephi).
#
# Install:
#   pip install requests networkx tqdm
#
# Run:
#   python .\lead_miner_wikidata_frde.py --max_leads 2000 --outdir out
# Optional:
#   python .\lead_miner_wikidata_frde.py --max_leads 2000 --outdir out --include_non_generic_emails
#   python .\lead_miner_wikidata_frde.py --max_leads 2000 --outdir out --ai_cluster_labels

import argparse, csv, json, os, time, re, random
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict
from urllib.parse import quote_plus, urlparse, urljoin

import requests
import networkx as nx
from tqdm import tqdm

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

FR = "Q142"
DE = "Q183"
BUSINESS_ENTERPRISE = "Q4830453"  # business enterprise

# Wikidata properties used:
# P112 = founded by
# P169 = chief executive officer
# P17  = country
# P452 = industry
# P856 = official website
# P6634 = LinkedIn personal ID
# P4264 = LinkedIn company ID

CONTACT_KEYWORDS = [
    "contact", "kontakt", "impressum", "legal", "about", "team", "leadership",
    "press", "presse", "media", "newsroom", "investor", "investors", "ir",
    "careers", "jobs", "support", "help", "sales", "partnership", "unternehmen"
]

GENERIC_EMAIL_HINTS = [
    "info", "contact", "hello", "press", "media", "pr", "presse",
    "ir", "investor", "investors", "careers", "jobs", "support",
    "sales", "bizdev", "business", "partnership", "kontakt"
]

ROLE_NORMALIZE = {
    "founded": "Founder",
    "ceo_of": "CEO",
}

_EMAIL_RE = re.compile(r"([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})")


# ----------------------------
# Helpers
# ----------------------------
def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def looks_like_person(name: str) -> bool:
    name = normalize_ws(name)
    if not name:
        return False
    parts = name.split()
    if len(parts) < 2:
        return False
    if len(name) > 80:
        return False
    # avoid org-like punctuation / handles
    if re.search(r"[0-9@#$/\\]|_", name):
        return False
    # basic capitalization check
    if not (parts[0][0].isupper() and parts[1][0].isupper()):
        return False
    return True

def wikidata_entity_url(qid_or_url: str) -> str:
    if not qid_or_url:
        return ""
    if qid_or_url.startswith("http"):
        return qid_or_url
    return f"https://www.wikidata.org/wiki/{qid_or_url}"

def google_search_link(person: str, company: str) -> str:
    q = f"{person} {company}".strip()
    return "https://www.google.com/search?q=" + quote_plus(q)

def linkedin_person_url(linkedin_id: str) -> str:
    linkedin_id = (linkedin_id or "").strip()
    return f"https://www.linkedin.com/in/{linkedin_id}" if linkedin_id else ""

def linkedin_company_url(company_id: str) -> str:
    company_id = (company_id or "").strip()
    return f"https://www.linkedin.com/company/{company_id}" if company_id else ""

def ensure_http(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return "https://" + url

def same_domain(a: str, b: str) -> bool:
    try:
        return urlparse(a).netloc.lower() == urlparse(b).netloc.lower()
    except Exception:
        return False


# ----------------------------
# Wikidata SPARQL
# ----------------------------
def build_sparql(limit: int, offset: int) -> str:
    # Deterministic-ish ordering helps pagination behave sanely.
    return f"""
    SELECT ?person ?personLabel ?company ?companyLabel ?ccountryLabel ?industryLabel
           ?personLinkedIn ?companyLinkedIn ?personWebsite ?companyWebsite
           ?relType
    WHERE {{
      {{
        ?company wdt:P31/wdt:P279* wd:{BUSINESS_ENTERPRISE} .
        ?company wdt:P17 ?ccountry .
        FILTER(?ccountry IN (wd:{FR}, wd:{DE})) .
        ?company wdt:P112 ?person .
        BIND("founded" AS ?relType)
        OPTIONAL {{ ?company wdt:P452 ?industry . }}
        OPTIONAL {{ ?person wdt:P6634 ?personLinkedIn . }}
        OPTIONAL {{ ?company wdt:P4264 ?companyLinkedIn . }}
        OPTIONAL {{ ?person wdt:P856 ?personWebsite . }}
        OPTIONAL {{ ?company wdt:P856 ?companyWebsite . }}
      }}
      UNION
      {{
        ?company wdt:P31/wdt:P279* wd:{BUSINESS_ENTERPRISE} .
        ?company wdt:P17 ?ccountry .
        FILTER(?ccountry IN (wd:{FR}, wd:{DE})) .
        ?company wdt:P169 ?person .
        BIND("ceo_of" AS ?relType)
        OPTIONAL {{ ?company wdt:P452 ?industry . }}
        OPTIONAL {{ ?person wdt:P6634 ?personLinkedIn . }}
        OPTIONAL {{ ?company wdt:P4264 ?companyLinkedIn . }}
        OPTIONAL {{ ?person wdt:P856 ?personWebsite . }}
        OPTIONAL {{ ?company wdt:P856 ?companyWebsite . }}
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,fr,de". }}
    }}
    ORDER BY ?person ?company
    LIMIT {limit}
    OFFSET {offset}
    """

def sparql_query(query: str, ua: str, session: requests.Session, timeout: int, retries: int) -> Dict[str, Any]:
    last_err = None
    for attempt in range(retries):
        try:
            r = session.get(
                WIKIDATA_SPARQL,
                params={"query": query, "format": "json"},
                headers={
                    "User-Agent": ua,
                    "Accept": "application/sparql-results+json",
                },
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            # exponential backoff + jitter
            sleep_s = min(60, (2 ** attempt)) + random.random()
            time.sleep(sleep_s)
    raise last_err

def parse_bindings(data: Dict[str, Any]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for b in data.get("results", {}).get("bindings", []):
        row: Dict[str, str] = {}
        for k, v in b.items():
            row[k] = v.get("value", "")
        rows.append(row)
    return rows


# ----------------------------
# Robots handling (requests-based, cached)
# ----------------------------
class RobotsCache:
    def __init__(self, user_agent: str, session: requests.Session, timeout: int = 20):
        self.ua = user_agent
        self.session = session
        self.timeout = timeout
        self._cache: Dict[str, Optional[re.Pattern]] = {}  # base -> compiled disallow regex (very simple)

    def _fetch_robots_text(self, base: str) -> str:
        robots_url = urljoin(base, "/robots.txt")
        try:
            r = self.session.get(robots_url, headers={"User-Agent": self.ua}, timeout=self.timeout, allow_redirects=True)
            if r.status_code >= 400:
                return ""
            return r.text or ""
        except Exception:
            return ""

    def _compile_rules(self, robots_txt: str) -> Optional[re.Pattern]:
        # Minimal parser: only respects "User-agent: *" Disallow lines.
        # If we can’t parse, we choose to be conservative (return None => disallow all).
        if not robots_txt.strip():
            return re.compile(r"^$")  # no disallow
        lines = [ln.strip() for ln in robots_txt.splitlines()]
        active = False
        disallows: List[str] = []
        for ln in lines:
            if not ln or ln.startswith("#"):
                continue
            m = re.match(r"(?i)^user-agent:\s*(.+)$", ln)
            if m:
                ua = m.group(1).strip()
                active = (ua == "*" )
                continue
            if not active:
                continue
            m = re.match(r"(?i)^disallow:\s*(.*)$", ln)
            if m:
                path = (m.group(1) or "").strip()
                if path == "":
                    continue
                # Escape regex special chars except *
                path_re = re.escape(path).replace(r"\*", ".*")
                disallows.append(path_re)
        if not disallows:
            return re.compile(r"^$")  # nothing disallowed
        return re.compile(r"^(?:" + "|".join(disallows) + ")")

    def allowed(self, url: str) -> bool:
        try:
            p = urlparse(url)
            if not p.scheme.startswith("http"):
                return False
            base = f"{p.scheme}://{p.netloc}"
            if base not in self._cache:
                txt = self._fetch_robots_text(base)
                self._cache[base] = self._compile_rules(txt)
            rule = self._cache[base]
            if rule is None:
                return False
            path = p.path or "/"
            return rule.search(path) is None
        except Exception:
            return False


# ----------------------------
# Contact harvesting
# ----------------------------
def fetch_html(session: requests.Session, url: str, ua: str, timeout: int = 25) -> str:
    try:
        r = session.get(url, headers={"User-Agent": ua}, timeout=timeout, allow_redirects=True)
        if r.status_code >= 400:
            return ""
        return r.text or ""
    except Exception:
        return ""

def extract_links(html: str, base_url: str, limit: int = 250) -> List[str]:
    if not html:
        return []
    links: List[str] = []
    for m in re.finditer(r'href\s*=\s*["\']([^"\']+)["\']', html, flags=re.I):
        href = (m.group(1) or "").strip()
        if not href or href.startswith("#"):
            continue
        if href.startswith("mailto:") or href.startswith("tel:"):
            continue
        u = urljoin(base_url, href)
        links.append(u)
        if len(links) >= limit:
            break
    # de-dup
    out: List[str] = []
    seen = set()
    for u in links:
        ul = u.lower()
        if ul in seen:
            continue
        seen.add(ul)
        out.append(u)
    return out

def pick_contact_pages(home_url: str, links: List[str], max_pages: int) -> List[str]:
    home_url = ensure_http(home_url)
    cand: List[str] = []
    for u in links:
        if not same_domain(home_url, u):
            continue
        path = (urlparse(u).path or "").lower()
        if any(k in path for k in CONTACT_KEYWORDS):
            cand.append(u)
    cand.sort(key=lambda x: len(urlparse(x).path or ""))
    return cand[:max_pages]

def extract_emails(html: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if not html:
        return out
    for m in _EMAIL_RE.finditer(html):
        email = m.group(1)
        start = max(0, m.start() - 60)
        end = min(len(html), m.end() + 60)
        snippet = re.sub(r"\s+", " ", html[start:end])
        out.append((email, snippet[:200]))
    # de-dup (case-insensitive)
    seen = set()
    ded: List[Tuple[str, str]] = []
    for e, snip in out:
        el = e.lower()
        if el in seen:
            continue
        seen.add(el)
        ded.append((e, snip))
    return ded

def is_generic_email(email: str) -> bool:
    email = (email or "").lower()
    if "@" not in email:
        return False
    local = email.split("@", 1)[0]
    return any(h in local for h in GENERIC_EMAIL_HINTS)

def classify_contact_type(url: str, email: str) -> str:
    s = (url or "").lower() + " " + (email or "").lower()
    if any(k in s for k in ["investor", "investors", "ir"]):
        return "investor_relations"
    if any(k in s for k in ["press", "presse", "media", "newsroom", "pr"]):
        return "press"
    if any(k in s for k in ["sales", "partner", "partnership", "bizdev", "business"]):
        return "sales_partnership"
    if any(k in s for k in ["careers", "jobs"]):
        return "careers"
    if any(k in s for k in ["support", "help"]):
        return "support"
    if any(k in s for k in ["impressum", "legal"]):
        return "legal"
    if "contact" in s or "kontakt" in s:
        return "contact"
    return "other"

def harvest_company_contacts(
    people: List[Dict[str, Any]],
    ua: str,
    robots: RobotsCache,
    sleep_s: float,
    max_pages_per_company: int,
    include_non_generic: bool,
    html_timeout: int = 25,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, str]]]:
    session = requests.Session()

    # unique companies with websites
    company_sites: Dict[str, str] = {}
    for p in people:
        comp = (p.get("company") or "").split("|")[0].strip()
        site = ensure_http(p.get("company_website") or "")
        if comp and site and comp not in company_sites:
            company_sites[comp] = site

    company_contacts_rows: List[Dict[str, Any]] = []
    best_contact_by_company: Dict[str, Dict[str, str]] = {}

    priority = {
        "investor_relations": 1,
        "press": 2,
        "sales_partnership": 3,
        "contact": 4,
        "legal": 5,
        "support": 6,
        "careers": 7,
        "other": 8,
    }

    def add_row(company: str, company_site: str, contact_url: str, email: str, evidence: str):
        ct = classify_contact_type(contact_url, email)
        company_contacts_rows.append({
            "company": company,
            "company_website": company_site,
            "contact_url": contact_url,
            "contact_type": ct,
            "email": email,
            "is_generic": str(is_generic_email(email)) if email else "",
            "evidence": evidence,
            "source": "company_site",
        })

    for comp, site in tqdm(company_sites.items(), desc="Harvesting company contact paths"):
        if not robots.allowed(site):
            continue

        html = fetch_html(session, site, ua=ua, timeout=html_timeout)
        if not html:
            time.sleep(sleep_s)
            continue

        # homepage emails
        for email, ev in extract_emails(html):
            if include_non_generic or is_generic_email(email):
                add_row(comp, site, site, email, ev)

        # contact pages
        links = extract_links(html, site)
        contact_pages = pick_contact_pages(site, links, max_pages=max_pages_per_company)

        visited = {site}
        for u in contact_pages:
            if u in visited:
                continue
            visited.add(u)
            if not robots.allowed(u):
                continue
            h2 = fetch_html(session, u, ua=ua, timeout=html_timeout)
            if not h2:
                time.sleep(sleep_s)
                continue

            emails = extract_emails(h2)
            if emails:
                for email, ev in emails:
                    if include_non_generic or is_generic_email(email):
                        add_row(comp, site, u, email, ev)
            else:
                add_row(comp, site, u, "", "contact page discovered (no email found)")
            time.sleep(sleep_s)

        # pick best contact for this company
        rows = [r for r in company_contacts_rows if r["company"] == comp]
        if rows:
            def score(r: Dict[str, Any]):
                has_email = 0 if r.get("email") else 1
                generic_pen = 0 if r.get("is_generic") == "True" else 1
                pr = priority.get(r.get("contact_type") or "other", 99)
                path_len = len(urlparse(r.get("contact_url","")).path or "")
                return (has_email, generic_pen, pr, path_len)

            best = sorted(rows, key=score)[0]
            best_contact_by_company[comp] = {
                "best_contact_email": best.get("email", ""),
                "best_contact_url": best.get("contact_url", ""),
                "best_contact_type": best.get("contact_type", ""),
            }

        time.sleep(sleep_s)

    return company_contacts_rows, best_contact_by_company


# ----------------------------
# Lead fetch + edges
# ----------------------------
def fetch_leads(
    max_leads: int,
    page_size: int,
    ua: str,
    sleep_s: float,
    sparql_timeout: int,
    sparql_retries: int,
    debug: bool
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    session = requests.Session()

    people: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    offset = 0

    pbar = tqdm(total=max_leads, desc="Collecting leads (unique people)", unit="people")

    while len(people) < max_leads:
        q = build_sparql(limit=page_size, offset=offset)
        try:
            data = sparql_query(q, ua=ua, session=session, timeout=sparql_timeout, retries=sparql_retries)
        except Exception as e:
            if debug:
                print(f"[Wikidata] query failed at offset={offset}: {e}")
            break

        rows = parse_bindings(data)
        if not rows:
            break

        if debug:
            print(f"[Wikidata] rows={len(rows)} offset={offset}")

        for r in rows:
            person = (r.get("personLabel") or "").strip()
            company = (r.get("companyLabel") or "").strip()
            if not looks_like_person(person) or not company:
                continue

            relType = (r.get("relType") or "").strip()
            role = ROLE_NORMALIZE.get(relType, relType)

            person_url = wikidata_entity_url(r.get("person", ""))
            company_url = wikidata_entity_url(r.get("company", ""))

            ccountry = (r.get("ccountryLabel") or "").strip()
            industry = (r.get("industryLabel") or "").strip()

            person_li = linkedin_person_url(r.get("personLinkedIn", ""))
            comp_li = linkedin_company_url(r.get("companyLinkedIn", ""))

            person_site = ensure_http(r.get("personWebsite", ""))
            company_site = ensure_http(r.get("companyWebsite", ""))

            if person_url and person_url not in people:
                people[person_url] = {
                    "person_name": person,
                    "role": role,
                    "company": company,
                    "company_country": ccountry,
                    "industry": industry,
                    "wikidata_person": person_url,
                    "wikidata_company": company_url,
                    "person_website": person_site,
                    "company_website": company_site,
                    "linkedin_profile": person_li,
                    "company_linkedin": comp_li,
                    "info_link": google_search_link(person, company),
                    "sources": "wikidata",
                }
                pbar.update(1)
                if len(people) >= max_leads:
                    # still record edge for this record
                    pass
            elif person_url and person_url in people:
                # multi-company: append if new
                existing = people[person_url].get("company", "")
                if company not in existing.split("|"):
                    people[person_url]["company"] = (existing + " | " + company).strip(" |")

            # relationship edge (always record, even if person already exists)
            edges.append({
                "subject_type": "person",
                "subject_name": person,
                "predicate": relType,
                "object_type": "company",
                "object_name": company,
                "reason": "Wikidata statement",
                "evidence": "Wikidata statement",
                "confidence": 0.95,
                "source": "wikidata",
                "subject_wikidata": person_url,
                "object_wikidata": company_url,
            })

            if len(people) >= max_leads:
                break

        offset += page_size
        time.sleep(sleep_s)

    pbar.close()
    return list(people.values()), edges


# ----------------------------
# Similarity edges (bounded per person)
# ----------------------------
def build_similarity_edges(people: List[Dict[str, Any]], max_neighbors: int = 5) -> List[Dict[str, Any]]:
    # Build inverted indices
    by_industry: Dict[str, List[str]] = defaultdict(list)
    by_country: Dict[str, List[str]] = defaultdict(list)

    for p in people:
        name = p["person_name"]
        ind = (p.get("industry") or "").strip()
        ctry = (p.get("company_country") or "").strip()
        if ind:
            by_industry[ind].append(name)
        if ctry:
            by_country[ctry].append(name)

    # For each person, accumulate candidate scores from shared features (industry, country)
    candidates: Dict[str, Counter] = defaultdict(Counter)

    def add_feature_groups(groups: Dict[str, List[str]], weight: int):
        for _, names in groups.items():
            if len(names) < 2:
                continue
            # limit extremely large groups by sampling (prevents heavy bias + blowups)
            if len(names) > 2000:
                names = random.sample(names, 2000)
            for n in names:
                for m in names:
                    if n != m:
                        candidates[n][m] += weight

    add_feature_groups(by_industry, weight=2)
    add_feature_groups(by_country, weight=1)

    # Create bounded edges: top-N neighbors per person
    edges: List[Dict[str, Any]] = []
    seen_pairs = set()

    for a, ctr in candidates.items():
        for b, score in ctr.most_common(max_neighbors):
            key = tuple(sorted((a, b)))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            edges.append({
                "subject_type": "person",
                "subject_name": a,
                "predicate": "similar_to",
                "object_type": "person",
                "object_name": b,
                "reason": "Shared features (industry/country)",
                "evidence": "Derived similarity",
                "confidence": min(0.9, 0.45 + 0.1 * score),
                "source": "derived",
                "subject_wikidata": "",
                "object_wikidata": "",
            })

    return edges


# ----------------------------
# Clustering
# ----------------------------
def cluster_people(people: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    G = nx.Graph()

    def pnode(n: str) -> str: return f"p:{n}"
    def cnode(n: str) -> str: return f"c:{n}"

    # People + company bipartite edges
    for p in people:
        pname = p["person_name"]
        G.add_node(pnode(pname), kind="person")
        for company in [c.strip() for c in (p.get("company") or "").split("|") if c.strip()]:
            G.add_node(cnode(company), kind="company")
            G.add_edge(pnode(pname), cnode(company), weight=2.0, kind="works")

    # Similarity edges (person-person)
    for e in edges:
        if e.get("predicate") == "similar_to" and e.get("subject_type") == "person" and e.get("object_type") == "person":
            G.add_edge(pnode(e["subject_name"]), pnode(e["object_name"]), weight=1.0, kind="similar")

    # Community detection, fallback to connected components
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        clusters = list(greedy_modularity_communities(G, weight="weight"))
    except Exception:
        clusters = [set(c) for c in nx.connected_components(G)]

    person_to_cluster: Dict[str, str] = {}
    cluster_meta: Dict[str, Dict[str, Any]] = {}
    person_lookup = {p["person_name"]: p for p in people}

    for i, nodes in enumerate(clusters, start=1):
        cid = f"C{i:04d}"
        inds: List[str] = []
        ctrys: List[str] = []
        members: List[str] = []
        companies: List[str] = []

        for n in nodes:
            if n.startswith("p:"):
                pname = n[2:]
                members.append(pname)
                person_to_cluster[pname] = cid
                p = person_lookup.get(pname, {})
                if p.get("industry"): inds.append(p["industry"])
                if p.get("company_country"): ctrys.append(p["company_country"])
            elif n.startswith("c:"):
                companies.append(n[2:])

        if members:
            cluster_meta[cid] = {
                "size": len(members),
                "top_industries": [x for x, _ in Counter(inds).most_common(3)],
                "top_countries": [x for x, _ in Counter(ctrys).most_common(3)],
                "top_companies": [x for x, _ in Counter(companies).most_common(5)],
                "sample_members": members[:10],
            }

    return person_to_cluster, cluster_meta


# ----------------------------
# Optional AI labeling
# ----------------------------
def ai_label_clusters(cluster_meta: Dict[str, Dict[str, Any]], api_key: str, model: str) -> Dict[str, str]:
    if not OpenAI:
        return {}
    client = OpenAI(api_key=api_key)
    labels: Dict[str, str] = {}

    for cid, meta in tqdm(cluster_meta.items(), desc="AI labeling clusters"):
        payload = {
            "cluster_id": cid,
            "size": meta.get("size", 0),
            "top_industries": meta.get("top_industries", []),
            "top_countries": meta.get("top_countries", []),
            "top_companies": meta.get("top_companies", []),
            "sample_members": meta.get("sample_members", []),
        }
        try:
            # Keep it simple: no strict JSON mode needed for a single label
            resp = client.responses.create(
                model=model,
                input=("Give a short label (max 8 words) for this cluster:\n"
                       + json.dumps(payload, ensure_ascii=False)),
                max_output_tokens=40,
            )
            label = (getattr(resp, "output_text", "") or "").strip().strip('"')
            if label:
                labels[cid] = label[:80]
        except Exception:
            continue

    return labels


# ----------------------------
# Output
# ----------------------------
def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_leads", type=int, default=2000)
    ap.add_argument("--page_size", type=int, default=200)  # safer than 500 for Wikidata
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--sleep", type=float, default=0.8, help="delay between Wikidata pages")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--sparql_timeout", type=int, default=60)
    ap.add_argument("--sparql_retries", type=int, default=6)

    ap.add_argument("--max_similarity_neighbors", type=int, default=5)

    ap.add_argument("--contact_sleep", type=float, default=1.0)
    ap.add_argument("--max_pages_per_company", type=int, default=6)
    ap.add_argument("--include_non_generic_emails", action="store_true",
                    help="Also keep personal emails if found on company pages (default keeps only role emails like info@, press@, etc.)")

    ap.add_argument("--ai_cluster_labels", action="store_true", help="Use OpenAI to label clusters (optional).")
    ap.add_argument("--ai_model", default="gpt-4o-mini")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ua = "LeadMiner/4.0 (mailto:you@example.com) Wikidata+CompanyContacts"

    # 1) Fetch leads
    people, base_edges = fetch_leads(
        max_leads=args.max_leads,
        page_size=args.page_size,
        ua=ua,
        sleep_s=args.sleep,
        sparql_timeout=args.sparql_timeout,
        sparql_retries=args.sparql_retries,
        debug=args.debug,
    )

    if not people:
        print("No people found. Try reducing --page_size to 100 and increasing --sparql_timeout.")
        return

    # 2) Similarity edges (bounded)
    sim_edges = build_similarity_edges(people, max_neighbors=args.max_similarity_neighbors)
    all_edges = base_edges + sim_edges

    # 3) Clusters
    person_to_cluster, cluster_meta = cluster_people(people, all_edges)

    for p in people:
        cid = person_to_cluster.get(p["person_name"], "")
        p["cluster_id"] = cid
        meta = cluster_meta.get(cid, {})
        if meta:
            inds = meta.get("top_industries", [])
            ctr = meta.get("top_countries", [])
            comps = meta.get("top_companies", [])
            p["cluster_reason"] = (
                f"Shared group: industries={', '.join(inds[:2])}; "
                f"countries={', '.join(ctr[:2])}; "
                f"companies={', '.join(comps[:2])}"
            )
        else:
            p["cluster_reason"] = ""

    # 4) Optional AI labels
    api_key = (os.getenv("OPENAI_API_KEY", "") or "").strip()
    if args.ai_cluster_labels and api_key:
        labels = ai_label_clusters(cluster_meta, api_key=api_key, model=args.ai_model)
        for p in people:
            cid = p.get("cluster_id", "")
            p["cluster_label"] = labels.get(cid, "")
    else:
        for p in people:
            p["cluster_label"] = ""

    # 5) Harvest contact paths (robots-respecting)
    session = requests.Session()
    robots = RobotsCache(user_agent=ua, session=session)

    company_contacts, best_by_company = harvest_company_contacts(
        people,
        ua=ua,
        robots=robots,
        sleep_s=args.contact_sleep,
        max_pages_per_company=args.max_pages_per_company,
        include_non_generic=args.include_non_generic_emails,
    )

    # attach best contact to each person
    for p in people:
        comp = (p.get("company") or "").split("|")[0].strip()
        best = best_by_company.get(comp, {})
        p["best_contact_email"] = best.get("best_contact_email", "")
        p["best_contact_url"] = best.get("best_contact_url", "")
        p["best_contact_type"] = best.get("best_contact_type", "")

    # 6) Build graph outputs (GraphML for Gephi)
    G = nx.Graph()

    # nodes
    for p in people:
        G.add_node(
            p["person_name"],
            kind="person",
            role=p.get("role", ""),
            company=p.get("company", ""),
            cluster=p.get("cluster_id", ""),
            country=p.get("company_country", ""),
            industry=p.get("industry", ""),
        )
        for company in [c.strip() for c in (p.get("company") or "").split("|") if c.strip()]:
            if not G.has_node(company):
                G.add_node(company, kind="company")

    # edges
    for e in all_edges:
        s = e.get("subject_name", "")
        o = e.get("object_name", "")
        if not s or not o:
            continue
        G.add_edge(
            s, o,
            predicate=e.get("predicate", ""),
            source=e.get("source", ""),
            confidence=float(e.get("confidence", 0.0) or 0.0)
        )

    # 7) Write outputs
    people_csv = os.path.join(args.outdir, "people.csv")
    edges_csv = os.path.join(args.outdir, "edges.csv")
    company_contacts_csv = os.path.join(args.outdir, "company_contacts.csv")
    bundle_json = os.path.join(args.outdir, "bundle.json")
    graphml_path = os.path.join(args.outdir, "graph.graphml")

    write_csv(
        people_csv,
        people,
        [
            "person_name","role","company","company_country","industry",
            "person_website","company_website","linkedin_profile","company_linkedin",
            "wikidata_person","wikidata_company","info_link",
            "cluster_id","cluster_label","cluster_reason",
            "best_contact_email","best_contact_url","best_contact_type",
            "sources"
        ],
    )

    write_csv(
        edges_csv,
        all_edges,
        [
            "subject_type","subject_name","predicate","object_type","object_name",
            "reason","evidence","confidence","source","subject_wikidata","object_wikidata"
        ],
    )

    write_csv(
        company_contacts_csv,
        company_contacts,
        ["company","company_website","contact_url","contact_type","email","is_generic","evidence","source"]
    )

    bundle = {
        "summary": {
            "people_count": len(people),
            "edges_count": len(all_edges),
            "clusters_count": len({p["cluster_id"] for p in people if p.get("cluster_id")}),
            "company_contacts_rows": len(company_contacts),
            "source": "wikidata (+ derived similarity) + company websites (robots-respecting)",
        },
        "clusters": cluster_meta,
        "people_sample": people[:50],
    }

    with open(bundle_json, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)

    nx.write_graphml(G, graphml_path)

    print("Done.")
    print(json.dumps(bundle["summary"], indent=2))
    print("Wrote:")
    print("-", people_csv)
    print("-", edges_csv)
    print("-", company_contacts_csv)
    print("-", bundle_json)
    print("-", graphml_path)
    print("\nTo visualize: open out/graph.graphml in Gephi (free) or Cytoscape (free).")


if __name__ == "__main__":
    main()
