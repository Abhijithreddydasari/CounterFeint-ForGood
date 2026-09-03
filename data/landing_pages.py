"""
Simulated landing page investigation data.

When the agent investigates 'landing_page' for an ad, this module
generates the findings text based on the ad's ground truth label.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List


@dataclass
class LandingPageData:
    domain: str
    domain_age_days: int
    ssl_valid: bool
    registrar: str
    content_summary: str
    redirect_chain: List[str]
    similarity_to_known_scams: float  # 0.0-1.0
    key_claims: List[str]
    suspicious_elements: List[str]

    def to_investigation_text(self) -> str:
        ssl_status = "Valid SSL certificate" if self.ssl_valid else "NO SSL / expired certificate"
        redirects = " -> ".join(self.redirect_chain) if self.redirect_chain else "No redirects (direct)"
        suspicious = "\n".join(f"  - {s}" for s in self.suspicious_elements) if self.suspicious_elements else "  None detected"
        claims = "\n".join(f"  - {c}" for c in self.key_claims) if self.key_claims else "  None"

        lines = [
            f"Domain: {self.domain}",
            f"Domain age: {self.domain_age_days} days",
            f"SSL: {ssl_status}",
            f"Registrar: {self.registrar}",
            f"WHOIS privacy: {'Enabled' if 'privacy' in self.registrar.lower() or 'proxy' in self.registrar.lower() else 'Disabled (registrant info public)'}",
            f"Redirect chain: {redirects}",
            f"Similarity to known scam templates: {self.similarity_to_known_scams:.0%}",
            f"Key claims on page:\n{claims}",
            f"Suspicious elements:\n{suspicious}",
            f"Content summary: {self.content_summary}",
        ]
        return "\n".join(lines)


_LEGIT_REGISTRARS = ["GoDaddy", "Namecheap", "Google Domains", "Cloudflare Registrar", "AWS Route 53"]
_SUSPICIOUS_REGISTRARS = ["NameSilo", "Epik", "Njalla (privacy)", "Tucows (privacy proxy)", "Openprovider"]

_LEGIT_DOMAINS_SUFFIXES = [".com", ".co", ".org", ".io", ".net"]
_SCAM_DOMAINS_SUFFIXES = [".shop", ".store", ".xyz", ".click", ".top", ".buzz", ".site"]


def generate_landing_page(
    rng: random.Random,
    ad_id: str,
    is_fraud: bool,
    fraud_type: str = "",
    *,
    domain_override: str | None = None,
    registrar_override: str | None = None,
    stealth: bool = False,
) -> LandingPageData:
    """Generate simulated landing page investigation data.

    ``stealth=True`` (task_3 / task_4): fraud pages look like ordinary
    businesses. No ``.xyz``, no "similarity 87%", no "NO SSL". The tell
    is buried and incomplete — investigator must cross-check payment /
    targeting, not read a label off the landing page.
    """

    base_word = rng.choice(["deal", "offer", "shop", "store", "buy", "get", "best", "top", "pro", "elite"])

    if is_fraud and stealth:
        brand = rng.choice(
            ["northpeak", "clearharbor", "fieldnote", "lumenpath", "riveroak", "stackline", "brightmill"]
        )
        suffix = rng.choice(_LEGIT_DOMAINS_SUFFIXES)
        domain = domain_override or f"{brand}{rng.randint(10, 99)}{suffix}"
        domain_age = rng.randint(200, 1800)
        ssl_valid = rng.random() < 0.88
        registrar = registrar_override or rng.choice(_LEGIT_REGISTRARS + _SUSPICIOUS_REGISTRARS[:2])
        similarity = round(rng.uniform(0.04, 0.22), 2)
        redirect_chain = []
        if rng.random() < 0.25:
            redirect_chain = [domain, f"www.{domain}"]
        suspicious = _get_stealth_suspicious_elements(rng)
        claims = _get_legit_claims(rng)
        content = _get_stealth_fraud_content_summary(rng, fraud_type)
    elif is_fraud:
        suffix = rng.choice(_SCAM_DOMAINS_SUFFIXES)
        domain = domain_override or f"{base_word}{rng.randint(10, 9999)}{suffix}"
        domain_age = rng.randint(1, 60)
        ssl_valid = rng.random() < 0.4
        registrar = registrar_override or rng.choice(_SUSPICIOUS_REGISTRARS)
        similarity = round(rng.uniform(0.5, 0.95), 2)

        redirect_count = rng.randint(1, 4)
        redirect_chain = [domain]
        for _ in range(redirect_count):
            redirect_chain.append(f"redirect{rng.randint(1, 999)}{rng.choice(['.click', '.top', '.xyz'])}")
        redirect_chain.append(f"final-{rng.randint(1, 999)}{rng.choice(['.site', '.store'])}")

        suspicious = _get_fraud_suspicious_elements(rng, fraud_type)
        claims = _get_fraud_claims(rng, fraud_type)
        content = _get_fraud_content_summary(rng, fraud_type)
    else:
        suffix = rng.choice(_LEGIT_DOMAINS_SUFFIXES)
        brand = rng.choice(["homenest", "stylehaven", "taskflow", "fitlife", "sparkle", "codeacademy", "nutri"])
        domain = domain_override or f"{brand}{suffix}"
        domain_age = rng.randint(365, 5000)
        ssl_valid = True
        registrar = registrar_override or rng.choice(_LEGIT_REGISTRARS)
        similarity = round(rng.uniform(0.0, 0.15), 2)
        redirect_chain = []
        suspicious = []
        claims = _get_legit_claims(rng)
        content = _get_legit_content_summary(rng)

    return LandingPageData(
        domain=domain,
        domain_age_days=domain_age,
        ssl_valid=ssl_valid,
        registrar=registrar,
        content_summary=content,
        redirect_chain=redirect_chain,
        similarity_to_known_scams=similarity,
        key_claims=claims,
        suspicious_elements=suspicious,
    )


def _get_fraud_suspicious_elements(rng: random.Random, fraud_type: str) -> List[str]:
    common = [
        "No physical address listed",
        "No contact phone number",
        "Privacy policy copied from template",
        "Terms of service link returns 404",
    ]
    type_specific = {
        "fake_giveaway": ["Countdown timer with fake urgency", "Requests personal info before any value provided"],
        "counterfeit": ["Product images appear stolen from official brand site", "Price comparison shows 90%+ discount vs retail"],
        "miracle_cure": ["FDA disclaimer buried in footer", "Before/after photos appear digitally altered"],
        "advance_fee_scam": ["Wire transfer requested upfront", "No company registration found"],
        "fake_crypto": ["Smart contract not verified on-chain", "Team photos are stock images"],
        "fake_endorsement": ["Celebrity quote not found in any verified source", "Affiliate tracking parameters in URL"],
        "brand_impersonation": ["Domain mimics well-known brand with character substitution", "Logo appears edited from official assets"],
        "gray_area": ["Citations reference unpublished or retracted studies", "Testimonials lack verifiable details"],
        "coordinated_network": ["Identical page template used across multiple domains", "Contact form submits to third-party aggregator"],
    }
    elements = list(common)
    elements.extend(type_specific.get(fraud_type, []))
    rng.shuffle(elements)
    return elements[: rng.randint(2, min(5, len(elements)))]


def _get_fraud_claims(rng: random.Random, fraud_type: str) -> List[str]:
    by_type = {
        "fake_giveaway": ["100% free, no purchase necessary", "Guaranteed winner", "Act now — expires in 2 hours"],
        "counterfeit": ["Authentic products", "Direct from manufacturer", "90% below retail price"],
        "miracle_cure": ["Clinically proven results", "Works in 7 days or less", "Endorsed by doctors"],
        "advance_fee_scam": ["Guaranteed returns", "Low processing fee required", "Confidential opportunity"],
        "fake_crypto": ["12-15% monthly returns guaranteed", "Audited by top security firms", "Risk-free investment"],
        "fake_endorsement": ["As seen on TV", "Celebrity recommended", "Limited exclusive offer"],
        "brand_impersonation": ["Official authorized retailer", "Factory direct pricing", "Same quality, lower price"],
        "gray_area": ["Clinically studied ingredients", "Doctor recommended", "30-day money-back guarantee"],
        "coordinated_network": ["Limited stock available", "Thousands of 5-star reviews", "Fast free shipping"],
    }
    return by_type.get(fraud_type, ["Special limited offer", "Act now"])


def _get_fraud_content_summary(rng: random.Random, fraud_type: str) -> str:
    summaries = {
        "fake_giveaway": "Landing page features a large countdown timer and a form requesting name, email, phone, and address. No clear sponsor or rules disclosure.",
        "counterfeit": "Product catalog showing luxury branded items at extreme discounts. Stock photos. Payment accepted via wire transfer and crypto only.",
        "miracle_cure": "Long-form sales page with testimonials, before/after images, and pseudoscientific explanations. Multiple urgency CTAs.",
        "advance_fee_scam": "Simple page with a letter-style appeal requesting wire transfer. Grammar errors. No company details.",
        "fake_crypto": "Professional-looking platform with dashboard screenshots. Whitepaper link leads to a generic PDF. Team bios use stock photos.",
        "fake_endorsement": "News article-style page with celebrity photos. Comments section appears pre-populated with positive feedback.",
        "brand_impersonation": "Near-replica of the official brand website. URL uses character substitution. Cart and checkout flow functional but data destination unclear.",
        "gray_area": "Well-designed supplement product page. Ingredient list present but efficacy claims exceed scientific evidence. Has a real return policy.",
        "coordinated_network": "Standard e-commerce template. Products appear legitimate but company details are vague. Identical template to other flagged sites.",
    }
    return summaries.get(fraud_type, "Generic landing page with minimal content and heavy use of urgency language.")


def _get_legit_claims(rng: random.Random) -> List[str]:
    options = [
        "Free shipping on orders over $50",
        "14-day free trial",
        "Money-back guarantee",
        "Established since 2010",
        "4.5-star average customer rating",
        "Licensed and insured",
        "BBB accredited",
        "Certified organic ingredients",
    ]
    rng.shuffle(options)
    return options[: rng.randint(2, 4)]


def _get_stealth_suspicious_elements(rng: random.Random) -> List[str]:
    """Ambiguous findings. None of these alone justify a reject."""
    pool = [
        "Checkout posts to a third-party processor on a different /24",
        "About page last updated 11 days ago; rest of site dated 2024",
        "Support inbox is a catch-all at a freemail domain",
        "Two product SKUs reuse the same manufacturer photo EXIF",
        "Return policy exists but the RMA form 302s to a URL shortener",
        "CDN cert is valid; origin cert expires in 19 days",
    ]
    rng.shuffle(pool)
    n = rng.randint(0, 2)
    return pool[:n]


def _get_stealth_fraud_content_summary(rng: random.Random, fraud_type: str) -> str:
    options = [
        "Standard Shopify-style catalog, about page, and shipping FAQ. Company address is a virtual office suite.",
        "SaaS marketing site with pricing tiers and a demo form. Footer legal entity does not match the WHOIS registrant.",
        "Service-business site with team photos and a booking widget. Photos reverse-search to a stock library.",
        "Well-structured storefront. Payment processor MID is shared with two unrelated verticals in the same week.",
    ]
    return rng.choice(options)


def _get_legit_content_summary(rng: random.Random) -> str:
    options = [
        "Well-structured business website with clear product catalog, about page, contact information, and shipping policy.",
        "Professional service website with team bios, case studies, client testimonials, and clear pricing.",
        "E-commerce store with detailed product descriptions, customer reviews, FAQ section, and accessible support.",
        "SaaS landing page with feature breakdown, pricing tiers, integration docs, and live demo option.",
    ]
    return rng.choice(options)
