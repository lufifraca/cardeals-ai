# When to Use an LLM (and When Not To): Extracting Structured Data from Messy HTML

This is a walkthrough of one design decision in **CarDeals AI** — how it pulls lease and finance
offers out of 11 dealership websites that all structure their pages differently.

The interesting part isn't the scraping. It's the question underneath it, which is one a lot of
developers are stuck on right now:

> **When should I use an LLM, and when should I just write a parser?**

The easy answer in 2026 is "use GPT-4 for everything." This walkthrough is about why that's the
wrong default, and what a hybrid approach looks like instead. You can clone the repo and run it on a
single dealer to watch raw HTML turn into clean JSON at the end.

---

## 1. The problem: 11 sites, no two alike

Each dealership runs on a different website platform — Octane, DealerOn, DealerInspire, Dealer.com,
and a few others. They all show the same *kind* of thing (a lease special: a car, a monthly payment,
a term, a due-at-signing amount), but the HTML is completely different.

Here's the same piece of information — "$329/mo for 36 months" — on two different platforms.

**DealerInspire** wraps everything in semantic-ish classes:

```html
<!-- illustrative: DealerInspire special card -->
<div class="vehicle-special">
  <span class="special-title">2025 Toyota RAV4 LE</span>
  <span class="lease-payment">$329</span>
  <span class="lease-term">36 months</span>
  <span class="due-at-signing">$3,999 due at signing</span>
</div>
```

**Octane** buries it in nested divs with payment and term smashed into one string:

```html
<!-- illustrative: Octane offer block -->
<div class="offer">
  <div class="offer__headline">RAV4 LE Lease</div>
  <div class="offer__terms">
    <p>$329/mo &middot; 36 mo &middot; $3,999 down</p>
  </div>
</div>
```

A single parser can't read both. The payment is in a `<span class="lease-payment">` on one site and
inside a `<p>` of comma-separated text on the other. Multiply that by 11 sites and you have 11
different shapes for the same five fields.

This is the exact situation where people reach for an LLM. So let's look at why doing *only* that is
a trap.

---

## 2. The tempting wrong default: "just send it all to GPT-4"

The naive pipeline is one function:

```python
# the approach we did NOT ship
def extract(html: str) -> list[Offer]:
    cleaned = strip_tags_and_scripts(html)
    return gpt4_extract(cleaned)   # one LLM call per page, every page, every day
```

It works. It's also a bad default, for three concrete reasons:

1. **Cost.** This scraper runs on a daily cron across 11 dealers. Sending every full page to GPT-4
   every day is a recurring bill for work that, on most of those sites, a deterministic parser could
   do for free. You're paying per token to re-derive structure you already understand.

2. **Non-determinism.** The model occasionally reformats a payment, drops a trailing zero, or
   "helpfully" infers a number that isn't on the page. For data that drives a price comparison, a
   confidently wrong `$329` vs `$392` is worse than a missing row.

3. **No source of truth.** When the model is the only thing standing between raw HTML and your
   database, you have no cheap, exact baseline to check it against.

The lesson isn't "LLMs are bad at this." They're great at it. The lesson is that an LLM is the
**expensive, fuzzy tool you reach for when the cheap, exact tool runs out** — not the first thing
you grab.

---

## 3. The hybrid approach: CSS first, GPT-4 as a fallback

The shipped pipeline routes each dealer through deterministic extraction first, and only falls back
to GPT-4 when that fails.

```
detect platform
   │
   ├── known platform?  ──►  CSS extractor (free, exact, deterministic)
   │                              │
   │                          got offers? ──► validate ──► save
   │                              │
   │                          empty / failed ─┐
   │                                          ▼
   └── unknown platform? ─────────────►  GPT-4 extraction (cleaned HTML → schema)
                                              │
                                          validate ──► save
```

### 3a. The deterministic path

For the 3 most common platforms, a small extractor knows exactly where the fields live. This is the
"cheap and exact" tool.

```python
# from scraper/css_extractors.py  (replace with your real extractor)
from bs4 import BeautifulSoup

def extract_dealerinspire(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    offers = []
    for card in soup.select(".vehicle-special"):
        offers.append({
            "title":   _text(card, ".special-title"),
            "payment": _money(card, ".lease-payment"),
            "term":    _months(card, ".lease-term"),
            "down":    _money(card, ".due-at-signing"),
            "source":  "css:dealerinspire",
        })
    return offers
```

No tokens, no latency, no guessing. For these platforms the structure is known, so we encode that
knowledge directly. This handles the bulk of the 11 dealers.

### 3b. The LLM fallback

For the long tail — unrecognized platforms, or a known platform that quietly changed its markup —
the page goes to GPT-4 with cleaned HTML and a strict schema. This is the "expensive but flexible"
tool, used only where the cheap one can't reach.

```python
# from scraper/extractor.py  (replace with your real call)
OFFER_SCHEMA = {
    "name": "extract_offers",
    "parameters": {
        "type": "object",
        "properties": {
            "offers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title":   {"type": "string"},
                        "payment": {"type": "number"},   # monthly, USD
                        "term":    {"type": "integer"},  # months
                        "down":    {"type": "number"},
                    },
                    "required": ["title", "payment", "term"],
                },
            }
        },
        "required": ["offers"],
    },
}

def gpt4_extract(cleaned_html: str) -> list[dict]:
    resp = client.chat.completions.create(
        model="gpt-4-turbo",
        tools=[{"type": "function", "function": OFFER_SCHEMA}],
        tool_choice={"type": "function", "function": {"name": "extract_offers"}},
        messages=[
            {"role": "system", "content":
                "Extract ONLY lease/finance offers explicitly present in the HTML. "
                "Never infer or estimate a number. If a field is absent, omit it."},
            {"role": "user", "content": cleaned_html},
        ],
    )
    return _parse_tool_call(resp)
```

Two details that matter more than the prompt wording:

- **A schema, not free text.** Function calling forces the model to return typed fields, so the
  output drops straight into the same shape the CSS path produces. Both paths converge on one
  internal model.
- **An explicit "don't guess" instruction.** The system message tells the model to omit missing
  fields rather than invent them. This won't make it perfect — which is why step 4 exists — but it
  removes the most common failure.

### 3c. The routing

The decision of which tool to use is small and boring on purpose:

```python
# from scraper/extractor.py  (replace with your real routing)
def extract_offers(html: str, platform: str) -> list[dict]:
    if platform in CSS_EXTRACTORS:
        offers = CSS_EXTRACTORS[platform](html)
        if offers:                      # cheap path worked
            return offers
    return gpt4_extract(clean_html(html))   # fall back only when it didn't
```

That `if offers:` is the whole idea: **try free and exact, fall back to paid and flexible only when
you have to.**

---

## 4. The guardrail: never trust either path blindly

Both paths — CSS *and* GPT-4 — feed into the same validation layer before anything is saved. A
selector can match the wrong element; a model can hallucinate. So nothing reaches the database
without passing the same checks.

```python
# from scraper/validators.py  (replace with your real rules)
KNOWN_MODELS = {"RAV4", "Camry", "Corolla", "CR-V", "Civic", "Accord", ...}

def validate(offer: dict) -> dict | None:
    if not (50 <= offer["payment"] <= 2000):     # implausible monthly payment
        return None
    if not (12 <= offer["term"] <= 60):           # implausible lease term
        return None
    if not _matches_known_model(offer["title"]):  # not a car we recognize
        return None

    offer["confidence"] = _score(offer)           # how sure are we?
    return offer if offer["confidence"] >= 0.6 else None
```

A `$2/mo` lease on a "2025 Toaster XL" gets dropped whether a buggy selector produced it or the model
dreamed it up. The confidence score is also stored, so the frontend can sort and flag lower-trust
rows instead of presenting everything as equally certain.

This is the part that makes the LLM safe to use at all: it isn't the last word, it's a *candidate
generator* whose output gets checked.

---

## 5. Run it yourself

Clone the repo and run the scraper against a single dealer to watch the whole flow — fetch, route,
extract, validate — turn a live dealer page into structured JSON:

```bash
cd scraper
pip install -r requirements.txt
playwright install chromium
python main.py --dealer dealerinspire-toyota --once --print
```

You'll see which path each offer came from (`css:*` vs `gpt4`), the validated fields, and the
confidence score — the same records that get written to the `offers` table.

---

## Takeaways

- **An LLM is a fallback, not a default.** Use deterministic parsing wherever the structure is known;
  reserve the model for the long tail you can't economically hand-code.
- **Make both paths converge on one schema** so the rest of your system doesn't care which one ran.
- **Validate everything, regardless of source.** Treat model output as a candidate to be checked, not
  a fact to be trusted.
- **Cost and determinism are design inputs**, not afterthoughts — especially for anything that runs on
  a schedule.

If you take one thing from this: the skill isn't "prompting GPT-4." It's knowing the boundary where
a parser stops being worth it and a model starts — and building a pipeline that lives on both sides
of that line.
