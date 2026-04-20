"""AI agent service for natural language car deal searches using GPT-4 with function calling."""

import json
import logging
from typing import Optional

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from schemas import OfferResponse, SearchParams
from services.offer_search import search_offers

logger = logging.getLogger(__name__)

settings = get_settings()
client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=30.0)

# System prompt for the AI agent
SYSTEM_PROMPT = """You are CarDealsAI, a helpful assistant that helps users find car lease and finance deals in the Los Angeles area.

You have access to a database of current vehicle offers from dealers in LA:

**Toyota Dealers (5):**
- Longo Toyota (El Monte)
- Toyota of Downtown LA (Los Angeles)
- North Hollywood Toyota (North Hollywood)
- Culver City Toyota (Culver City)
- AutoNation Toyota Cerritos (Cerritos)

**Honda Dealers (4):**
- Airport Marina Honda (Los Angeles)
- Galpin Honda (Mission Hills)
- Goudy Honda (Alhambra)
- Norm Reeves Honda Cerritos (Cerritos)
- Scott Robinson Honda (Torrance)

When users ask about car deals, use the search_offers function to find relevant offers. You can filter by:
- make: Brand name (Toyota, Honda)
- model: Model name (RAV4, Camry, Civic, CR-V, etc.)
- max_monthly_payment: Maximum monthly payment they want to pay
- offer_type: "lease" or "finance"
- max_down_payment: Maximum down payment

After getting search results, provide a helpful response that:
1. Summarizes what you found
2. Highlights the best deals
3. Mentions the make, model, dealer and key terms (monthly payment, down payment, term)
4. Encourages them to check the source link for full details

If search results include a "SIMILAR OFFERS" section, it means the exact query had no matches but we found related deals. Present those as helpful suggestions — explain why the original search had no exact matches, then recommend the similar deals the user might be interested in.

If they ask about brands other than Toyota or Honda, or locations outside LA, politely explain that you currently only have Toyota and Honda deals in the Los Angeles area.

**General car questions:** You can also answer general questions about car models, trims, features, reliability, comparisons between models, fuel economy, safety ratings, etc. For example: "What's the difference between RAV4 LE and XLE?", "Is the Camry reliable?", "Which Toyota SUV is best for families?". Answer these directly using your knowledge without calling search_offers. If relevant, mention that you can also search for current deals on that model.

Keep responses concise but helpful (under 250 words). Use natural, conversational language."""

# Function definition for GPT-4
SEARCH_FUNCTION = {
    "name": "search_offers",
    "description": "Search for current vehicle offers (lease and finance deals) from LA dealers. Use this whenever the user asks about car deals, prices, or specific models. Supports Toyota and Honda.",
    "parameters": {
        "type": "object",
        "properties": {
            "make": {
                "type": "string",
                "enum": ["Toyota", "Honda"],
                "description": "Vehicle manufacturer/brand to filter by. Leave empty to search all makes."
            },
            "model": {
                "type": "string",
                "description": "Model name to filter by. Toyota: RAV4, Camry, Corolla, Tacoma, Highlander, Prius, Tundra, 4Runner. Honda: Civic, Accord, CR-V, HR-V, Pilot, Prologue, Odyssey, Ridgeline. Leave empty to search all models."
            },
            "max_monthly_payment": {
                "type": "number",
                "description": "Maximum monthly payment in dollars (e.g., 350 for under $350/month)"
            },
            "offer_type": {
                "type": "string",
                "enum": ["lease", "finance"],
                "description": "Type of offer: 'lease' for lease deals, 'finance' for financing/APR deals"
            },
            "max_down_payment": {
                "type": "number",
                "description": "Maximum down payment in dollars"
            }
        },
        "required": []
    }
}


def format_offer_for_display(offer: OfferResponse) -> str:
    """Format an offer for display in the AI response."""
    parts = [f"**{offer.year} {offer.make} {offer.model}"]
    if offer.trim:
        parts[0] += f" {offer.trim}"
    parts[0] += "**"

    if offer.offer_type == "lease" and offer.monthly_payment:
        parts.append(f"${offer.monthly_payment}/mo lease")
        if offer.down_payment:
            parts.append(f"(${offer.down_payment:,.0f} due at signing)")
        if offer.term_months:
            parts.append(f"for {offer.term_months} months")
    elif offer.offer_type == "finance" and offer.apr:
        parts.append(f"{offer.apr}% APR financing")
        if offer.term_months:
            parts.append(f"up to {offer.term_months} months")

    parts.append(f"at {offer.dealer_name}")

    return " ".join(parts)


def format_offers_summary(offers: list[OfferResponse]) -> str:
    """Create a summary of offers for the AI to use."""
    if not offers:
        return "No offers found matching the criteria."

    lines = [f"Found {len(offers)} offer(s):\n"]
    for i, offer in enumerate(offers[:10], 1):
        lines.append(f"{i}. {format_offer_for_display(offer)}")

    return "\n".join(lines)


async def execute_search(
    db: AsyncSession,
    function_args: dict
) -> tuple[list[OfferResponse], dict]:
    """Execute the search_offers function with parsed arguments."""
    # Build SearchParams from function arguments
    params = SearchParams(
        make=function_args.get("make"),
        model=function_args.get("model"),
        max_monthly_payment=function_args.get("max_monthly_payment"),
        offer_type=function_args.get("offer_type"),
        max_down_payment=function_args.get("max_down_payment"),
        limit=10,
        sort_by="monthly_payment"
    )

    # Execute search
    offers, filters_applied = await search_offers(db, params)
    return offers, filters_applied


async def execute_fallback_search(
    db: AsyncSession,
    original_args: dict
) -> list[OfferResponse]:
    """Try broader searches when the original query returns no results."""
    # Strategy 1: Keep make, drop model and payment filters
    if original_args.get("model") or original_args.get("max_monthly_payment"):
        params = SearchParams(
            make=original_args.get("make"),
            offer_type=original_args.get("offer_type"),
            limit=5,
            sort_by="monthly_payment"
        )
        offers, _ = await search_offers(db, params)
        if offers:
            return offers

    # Strategy 2: Search all offers with no filters
    params = SearchParams(limit=5, sort_by="monthly_payment")
    offers, _ = await search_offers(db, params)
    return offers


async def process_chat(
    db: AsyncSession,
    message: str,
    history: Optional[list[dict]] = None,
) -> tuple[str, list[OfferResponse], Optional[dict]]:
    """
    Process a chat message using GPT-4 with function calling.

    Args:
        db: Database session
        message: Current user message
        history: Optional list of previous messages [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        Tuple of (response_text, offers_list, search_params_used)
    """
    logger.info(f"Processing chat: {message[:100]}...")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Add conversation history (last 6 messages max to control costs)
    if history:
        for msg in history[-6:]:
            if msg.get("role") in ("user", "assistant") and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": message})

    tools = [{"type": "function", "function": SEARCH_FUNCTION}]

    try:
        # First call to GPT-4 - may call function
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=1000,
        )

        assistant_message = response.choices[0].message
        logger.info(f"GPT-4 response: finish_reason={response.choices[0].finish_reason}")

        # Check if GPT-4 wants to call a function
        if assistant_message.tool_calls:
            tool_call = assistant_message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            logger.info(f"Function call: {function_name} with args: {function_args}")

            if function_name == "search_offers":
                # Execute the search
                offers, filters_applied = await execute_search(db, function_args)

                # If no results, try a broader fallback search
                fallback_offers = []
                if not offers:
                    logger.info("No exact matches, trying fallback search...")
                    fallback_offers = await execute_fallback_search(db, function_args)

                # Format results for GPT-4
                if offers:
                    offers_summary = format_offers_summary(offers)
                elif fallback_offers:
                    offers_summary = (
                        "No offers found matching the exact criteria.\n\n"
                        "SIMILAR OFFERS you can recommend:\n"
                        + format_offers_summary(fallback_offers)
                    )
                    offers = fallback_offers  # Return these to the frontend
                else:
                    offers_summary = "No offers found matching the criteria, and no similar offers are available right now."

                # Add the function call and result to messages
                messages.append(assistant_message.model_dump())
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": offers_summary
                })

                # Get final response from GPT-4
                final_response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000,
                )

                response_text = final_response.choices[0].message.content
                logger.info(f"Final response length: {len(response_text)} chars, {len(offers)} offers")

                return response_text, offers, function_args

        # No function call - direct response (general questions, greetings, etc.)
        response_text = assistant_message.content or "I'm here to help you find car deals in LA. I can search Toyota and Honda offers, or answer questions about car models. What are you looking for?"
        logger.info(f"Direct response (no function call): {len(response_text)} chars")
        return response_text, [], None

    except Exception as e:
        logger.exception(f"Error processing chat: {e}")
        return (
            "I'm sorry, I encountered an error while searching for deals. Please try again or use the search filters directly.",
            [],
            None
        )
