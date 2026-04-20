"""Extract offers from dealer page HTML using GPT-4."""

import json
import logging
import re
from typing import Optional

from bs4 import BeautifulSoup
from openai import OpenAI

from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY, timeout=60.0)

EXTRACTION_PROMPT = """Extract all current vehicle lease and finance offers from this dealership page.

Return a JSON array. Each offer should have this structure:
{{
  "year": 2025,
  "make": "Toyota",
  "model": "RAV4",
  "trim": "LE",
  "offer_type": "lease",
  "monthly_payment": 299.00,
  "down_payment": 3499.00,
  "term_months": 36,
  "annual_mileage": 12000,
  "apr": null,
  "msrp": null,
  "selling_price": null,
  "offer_end_date": "2025-02-28",
  "disclaimer": "Plus tax, title, license. On approved credit...",
  "confidence": 0.95
}}

Rules:
- Only extract CURRENT offers (not expired)
- Use null for any field you cannot determine
- The "make" field should be the car manufacturer (Toyota, Honda, Tesla, etc.)
- monthly_payment and down_payment are in dollars (not cents)
- Set confidence between 0.0 and 1.0:
  - 0.9+ : All key fields clearly stated
  - 0.7-0.9 : Most fields clear, some inferred
  - 0.5-0.7 : Several fields uncertain or missing
  - Below 0.5 : Don't include the offer
- If no valid offers found, return an empty array []
- Return ONLY the JSON array, no explanation or markdown

Dealership page content:
{page_text}"""


# Known models for image matching (all makes)
ALL_MODELS = [
    # Toyota
    "4Runner", "bZ4X", "Camry", "Corolla", "Corolla Cross",
    "Crown", "GR86", "GR Corolla", "GR Supra", "Grand Highlander",
    "Highlander", "Land Cruiser", "Mirai", "Prius", "RAV4",
    "Sequoia", "Sienna", "Tacoma", "Tundra", "Venza",
    # Honda
    "Accord", "Civic", "CR-V", "HR-V", "Passport", "Pilot",
    "Prologue", "Ridgeline", "Odyssey", "Insight",
    # Tesla
    "Model 3", "Model Y", "Model S", "Model X", "Cybertruck"
]


def clean_html(html: str) -> str:
    """
    Clean HTML and extract text content.
    Removes scripts, styles, and extracts visible text.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style elements
    for element in soup(["script", "style", "nav", "footer", "header", "meta", "link"]):
        element.decompose()

    # Get text with newline separators
    text = soup.get_text(separator="\n", strip=True)

    # Clean up excessive whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = "\n".join(lines)

    return text


def extract_vehicle_images(html: str, base_url: str = "") -> dict[str, str]:
    """
    Extract vehicle images from HTML and map them to model names.

    Returns a dict mapping lowercase model names to image URLs.
    """
    from urllib.parse import urljoin

    soup = BeautifulSoup(html, "html.parser")
    model_images: dict[str, str] = {}

    # Find all images
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
        if not src:
            continue

        # Make URL absolute
        if src.startswith("//"):
            src = "https:" + src
        elif src.startswith("/"):
            src = urljoin(base_url, src)
        elif not src.startswith("http"):
            src = urljoin(base_url, src)

        # Skip tiny images, icons, logos
        width = img.get("width")
        height = img.get("height")
        if width and height:
            try:
                if int(width) < 100 or int(height) < 100:
                    continue
            except ValueError:
                pass

        # Check image URL and alt text for model names
        img_text = f"{src} {img.get('alt', '')} {img.get('title', '')}".lower()

        for model in ALL_MODELS:
            model_lower = model.lower().replace(" ", "")
            # Check if model name appears in image URL or alt text
            if model_lower in img_text.replace(" ", "").replace("-", "").replace("_", ""):
                # Prefer larger/better quality images
                if model_lower not in model_images or "large" in src.lower() or "full" in src.lower():
                    model_images[model_lower] = src
                    logger.debug(f"Found image for {model}: {src[:80]}...")

    logger.info(f"Extracted images for {len(model_images)} models: {list(model_images.keys())}")
    return model_images


def truncate_text(text: str, max_chars: int = 15000) -> str:
    """Truncate text to max characters, trying to break at sentence boundaries."""
    if len(text) <= max_chars:
        return text

    # Try to find a good break point
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.8:
        return truncated[:last_period + 1]
    return truncated


def parse_json_response(response_text: str) -> list[dict]:
    """
    Parse JSON from GPT response, handling markdown code blocks.
    """
    # Strip markdown code fences if present
    text = response_text.strip()
    if text.startswith("```"):
        # Remove opening fence
        text = re.sub(r"^```json?\s*\n?", "", text)
        # Remove closing fence
        text = re.sub(r"\n?```\s*$", "", text)

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        elif isinstance(result, dict) and "offers" in result:
            return result["offers"]
        else:
            logger.warning(f"Unexpected JSON structure: {type(result)}")
            return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        logger.debug(f"Raw response: {text[:500]}")
        return []


def has_offer_indicators(text: str) -> tuple[bool, int]:
    """
    Check if page text contains enough offer-related keywords to justify GPT extraction.
    This prevents wasting tokens on pages without real offers.

    Returns:
        Tuple of (has_enough_indicators, count_found)
    """
    from config import OFFER_KEYWORDS, MIN_OFFER_KEYWORDS

    text_lower = text.lower()
    found = sum(1 for keyword in OFFER_KEYWORDS if keyword.lower() in text_lower)
    return found >= MIN_OFFER_KEYWORDS, found


def extract_offers(html: str, dealer_name: str, base_url: str = "",
                   dealer_slug: str = "", default_make: str = "Toyota") -> list[dict]:
    """
    Extract offers from dealer page HTML.

    Uses a hybrid approach:
    1. Try CSS-based extraction first (fast, free)
    2. Try auto-detecting platform from HTML
    3. Fall back to GPT-4 only if CSS extraction fails

    Args:
        html: Raw HTML content from dealer page
        dealer_name: Name of the dealer (for logging)
        base_url: Base URL for resolving relative image URLs
        dealer_slug: Dealer slug for CSS extractor lookup
        default_make: Default make for this dealer (Toyota, Honda, etc.)

    Returns:
        List of offer dictionaries
    """
    logger.info(f"Extracting offers from {dealer_name}...")

    from css_extractors import has_css_extractor, extract_with_css, detect_platform

    # Try CSS extraction first (fast, no AI tokens)
    can_css = dealer_slug and has_css_extractor(dealer_slug)
    # Also try auto-detection even without a slug override
    if not can_css:
        can_css = detect_platform(html) is not None

    if can_css:
        logger.info(f"Trying CSS extractor for {dealer_slug or 'auto-detected'}")
        offers = extract_with_css(dealer_slug, html, base_url, default_make=default_make)
        if offers:
            logger.info(f"CSS extraction successful: {len(offers)} offers (0 AI tokens used)")
            return offers
        logger.warning(f"CSS extraction returned no results, falling back to LLM")

    # Extract vehicle images first (for LLM fallback)
    model_images = extract_vehicle_images(html, base_url)

    # Clean and truncate HTML
    text = clean_html(html)
    text = truncate_text(text, max_chars=15000)

    logger.info(f"Cleaned text: {len(text):,} characters")

    if len(text) < 100:
        logger.warning(f"Page text too short for {dealer_name}, skipping extraction")
        return []

    # Pre-check: Look for offer indicators before calling GPT
    has_offers, keyword_count = has_offer_indicators(text)
    if not has_offers:
        logger.warning(f"Page lacks offer indicators ({keyword_count} found) for {dealer_name}, skipping GPT extraction")
        return []

    logger.info(f"Found {keyword_count} offer indicators, proceeding with GPT extraction")

    # Build prompt
    prompt = EXTRACTION_PROMPT.format(page_text=text)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a data extraction assistant. Extract vehicle offers from dealer websites and return structured JSON."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=4000,
        )

        response_text = response.choices[0].message.content
        logger.debug(f"GPT response: {response_text[:500]}")

        # Parse JSON
        offers = parse_json_response(response_text)

        # Attach images to offers based on model name
        for offer in offers:
            model = offer.get("model", "").lower().replace(" ", "")
            if model in model_images:
                offer["image_url"] = model_images[model]
                logger.debug(f"Attached image to {offer.get('model')}")

        # Count offers with images
        offers_with_images = sum(1 for o in offers if o.get("image_url"))
        logger.info(f"Attached images to {offers_with_images}/{len(offers)} offers")

        # Log token usage
        usage = response.usage
        logger.info(
            f"GPT-4 usage: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion = {usage.total_tokens} total tokens"
        )

        logger.info(f"Extracted {len(offers)} raw offers from {dealer_name}")
        return offers

    except Exception as e:
        logger.error(f"GPT-4 extraction failed for {dealer_name}: {e}")
        return []


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    from fetcher import fetch_page

    url = "https://www.longotoyota.com/new-vehicle-specials/"
    html = fetch_page(url)

    if html:
        offers = extract_offers(html, "Longo Toyota", base_url=url)
        print(f"\nExtracted {len(offers)} offers:")
        for offer in offers[:3]:  # Show first 3
            print(json.dumps(offer, indent=2))
    else:
        print("Failed to fetch page")
