"""
Synthetic advertiser profile generation.

Each ad has an associated advertiser with history data
that becomes available when the agent investigates 'advertiser_history'.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List


@dataclass
class AdvertiserProfile:
    advertiser_id: str
    account_name: str
    account_age_days: int
    total_spend_usd: float
    previous_violations: int
    previous_bans: int
    ad_volume_last_30d: int
    historical_approval_rate: float
    payment_method_id: str
    payment_method_type: str
    country: str
    verified_business: bool
    account_created_date: str = ""
    spend_velocity: str = ""
    ad_submission_pattern: str = ""

    def to_investigation_text(self) -> str:
        status = "Verified Business" if self.verified_business else "Unverified"
        violation_note = ""
        if self.previous_violations > 0:
            violation_note = f" ({self.previous_violations} policy violations on record, {self.previous_bans} previous bans)"
        elif self.previous_bans > 0:
            violation_note = f" ({self.previous_bans} previous bans on record)"

        lines = [
            f"Advertiser: {self.account_name} ({status})",
            f"Account age: {self.account_age_days} days",
            f"Account created: {self.account_created_date}" if self.account_created_date else None,
            f"Country: {self.country}",
            f"Total historical spend: ${self.total_spend_usd:,.2f}",
            f"Ads submitted in last 30 days: {self.ad_volume_last_30d}",
            f"Historical approval rate: {self.historical_approval_rate:.0%}{violation_note}",
            f"Payment method: {self.payment_method_type} (ID: {self.payment_method_id})",
            f"Spend velocity: {self.spend_velocity}" if self.spend_velocity else None,
            f"Submission pattern: {self.ad_submission_pattern}" if self.ad_submission_pattern else None,
        ]
        return "\n".join(l for l in lines if l is not None)


_LEGIT_NAMES = [
    "HomeNest LLC", "StyleHaven Inc", "Chef's Choice Store", "GlowNatural Co",
    "ArtisanHide Crafts", "TaskFlow Pro", "InvoiceSimple Inc", "PeopleFirst HR",
    "VaultBackup Solutions", "MailCraft Inc", "Mike's Plumbing", "Sal's Pizzeria",
    "SparkleHome Services", "Williams Tax Group", "PawPals Chicago",
    "CodeAcademy Pro", "StateU Online", "PrepMaster Tutoring",
    "LensArt Studio", "LinguaViva Education", "FitLife Fitness",
    "ZenFlow Yoga", "NutriPlan Co", "SwiftStride Athletics", "FitVisit",
]

_SCAM_NAMES = [
    "Digital Marketing Solutions LLC", "Global Deals Marketplace",
    "Premium Offers International", "Quick Rewards Corp",
    "NextGen Trading Ltd", "Elite Ventures Group",
    "Horizon Brands LLC", "TrustPoint Commerce",
    "Alpha Innovations Inc", "PrimeEdge Solutions",
    "Quantum Returns Ltd", "FuturePath Holdings",
    "BlueChip Partners", "ClearView Enterprises",
    "Apex Growth Strategies", "SilverLine Dynamics",
]

_COUNTRIES_LEGIT = ["United States", "United Kingdom", "Canada", "Australia", "Germany"]
_COUNTRIES_MIXED = [
    "United States", "United Kingdom", "Netherlands", "Singapore",
    "Hong Kong", "United Arab Emirates", "Estonia", "Georgia",
]


def generate_advertiser_profile(
    rng: random.Random,
    ad_id: str,
    is_fraud: bool,
    *,
    payment_method_id: str | None = None,
    ring_created_date: str | None = None,
    stealth: bool = False,
) -> AdvertiserProfile:
    """Generate a synthetic advertiser profile for a single ad."""
    from datetime import date, timedelta

    if is_fraud and stealth:
        account_name = rng.choice(_SCAM_NAMES + _LEGIT_NAMES[:8])
        account_age = rng.randint(120, 900)
        total_spend = round(rng.uniform(800, 40000), 2)
        violations = rng.choices([0, 1], weights=[85, 15])[0]
        bans = 0
        ad_volume = rng.randint(2, 18)
        approval_rate = round(rng.uniform(0.82, 0.97), 2)
        country = rng.choice(_COUNTRIES_LEGIT + _COUNTRIES_MIXED[:3])
        verified = rng.random() < 0.55
        pmt_type = rng.choice(["credit_card", "corporate_card", "bank_account", "prepaid_card"])
    elif is_fraud:
        account_name = rng.choice(_SCAM_NAMES)
        account_age = rng.randint(1, 90)
        total_spend = round(rng.uniform(0, 500), 2)
        violations = rng.choices([0, 1, 2, 3], weights=[40, 30, 20, 10])[0]
        bans = rng.choices([0, 1, 2], weights=[60, 30, 10])[0]
        ad_volume = rng.randint(5, 80)
        approval_rate = round(rng.uniform(0.3, 0.75), 2)
        country = rng.choice(_COUNTRIES_MIXED)
        verified = rng.random() < 0.15
        pmt_type = rng.choice(["prepaid_card", "crypto", "virtual_card", "wire_transfer", "credit_card"])
    else:
        account_name = rng.choice(_LEGIT_NAMES)
        account_age = rng.randint(180, 2500)
        total_spend = round(rng.uniform(5000, 500000), 2)
        violations = 0
        bans = 0
        ad_volume = rng.randint(1, 20)
        approval_rate = round(rng.uniform(0.9, 1.0), 2)
        country = rng.choice(_COUNTRIES_LEGIT)
        verified = rng.random() < 0.85
        pmt_type = rng.choice(["credit_card", "bank_account", "corporate_card"])

    if payment_method_id is None:
        payment_method_id = f"pmt_{rng.randint(100000, 999999)}"

    # Temporal signals
    if ring_created_date:
        created_date = ring_created_date
    else:
        created = date(2026, 4, 6) - timedelta(days=account_age)
        created_date = created.isoformat()

    if is_fraud:
        spend_per_day = total_spend / max(account_age, 1)
        if spend_per_day > 20:
            spend_velocity = f"${spend_per_day:,.0f}/day avg — ramped from $0 to ${total_spend:,.0f} in {account_age} days"
        else:
            spend_velocity = f"${spend_per_day:,.0f}/day avg over account lifetime"

        if ad_volume > 20:
            submission_pattern = f"{ad_volume} ads in 30 days (burst: {rng.randint(8, ad_volume)} in a single 24h window)"
        else:
            submission_pattern = f"{ad_volume} ads in 30 days (steady cadence)"
    else:
        spend_per_day = total_spend / max(account_age, 1)
        spend_velocity = f"${spend_per_day:,.0f}/day avg — consistent growth over {account_age} days"
        submission_pattern = f"{ad_volume} ads in 30 days (steady cadence)"

    return AdvertiserProfile(
        advertiser_id=f"adv_{ad_id}",
        account_name=account_name,
        account_age_days=account_age,
        total_spend_usd=total_spend,
        previous_violations=violations,
        previous_bans=bans,
        ad_volume_last_30d=ad_volume,
        historical_approval_rate=approval_rate,
        payment_method_id=payment_method_id,
        payment_method_type=pmt_type,
        country=country,
        verified_business=verified,
        account_created_date=created_date,
        spend_velocity=spend_velocity,
        ad_submission_pattern=submission_pattern,
    )
