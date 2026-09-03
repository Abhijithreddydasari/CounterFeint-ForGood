"""
Synthetic ad queue generation.

Generates a complete queue of ads for a given task configuration,
including all pre-generated investigation data. When the agent
investigates, the environment just reveals pre-computed data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .advertiser_profiles import AdvertiserProfile, generate_advertiser_profile
from .fraud_patterns import FRAUD_TEMPLATES, LEGIT_TEMPLATES, AdTemplate
from .landing_pages import LandingPageData, generate_landing_page
from .network_generator import FraudRing, generate_fraud_networks

# Decoy pools: values that can appear in both legit and fraud ads,
# making naive pattern-matching unreliable.
_DECOY_REGISTRARS = ["NameSilo", "Cloudflare Registrar", "GoDaddy", "Tucows (privacy proxy)"]
_DECOY_PAYMENT_TYPES = ["credit_card", "prepaid_card", "corporate_card"]
_COMMON_TARGETING_SEGMENTS = [
    "Adults 25-54, interests: shopping, lifestyle",
    "Adults 18-45, interests: technology, gadgets",
    "Adults 30-55, interests: finance, investing",
]

# Shown to the investigator instead of fake_* / network_* category names.
_SURFACE_CATEGORY: Dict[str, str] = {
    "fake_giveaway": "ecommerce",
    "miracle_cure": "fitness",
    "counterfeit_goods": "ecommerce",
    "advance_fee": "saas",
    "fake_crypto": "saas",
    "celebrity_endorsement_fraud": "fitness",
    "clone_brand": "ecommerce",
    "gray_area_supplements": "fitness",
    "network_crypto": "saas",
    "network_ecommerce": "ecommerce",
    "network_fintech": "saas",
    "network_health": "fitness",
}


# Curriculum escalation category pools. `_TASK_1_FRAUD_POOL` is the novice
# fraudster's toolkit (only two obvious scam templates + legit camouflage),
# `_TASK_2_FRAUD_POOL` adds mid-tier deceptive patterns, and `task_3` uses
# the server-side default which includes the network_* ring categories.
_LEGIT_CAMOUFLAGE = ("ecommerce", "saas", "local_service", "education", "fitness")

_TASK_1_ALLOWED_CATEGORIES: List[str] = list(_LEGIT_CAMOUFLAGE) + [
    "fake_giveaway",
    "miracle_cure",
]

_TASK_2_ALLOWED_CATEGORIES: List[str] = _TASK_1_ALLOWED_CATEGORIES + [
    "counterfeit_goods",
    "advance_fee",
    "fake_crypto",
    "celebrity_endorsement_fraud",
    "clone_brand",
    "gray_area_supplements",
]


@dataclass
class TaskConfig:
    task_id: str
    name: str
    difficulty: str
    queue_size: int
    action_budget: int
    n_legit: int
    n_fraud: int
    n_escalate: int
    include_networks: bool
    n_fraud_rings: int
    allowed_difficulties: List[str]
    description: str

    max_rounds: Optional[int] = None
    max_proposals: Optional[int] = None
    max_fraudster_actions_per_turn: Optional[int] = None
    max_investigator_actions_per_turn: Optional[int] = None
    allowed_fraud_categories: Optional[List[str]] = None
    # 0 = current (labels on the surface). 1 = hide risk_signals, mix stealth
    # landing pages. 2 = camouflage category + stealth LP + quiet profiles.
    # 3 = task_4: chain rings, payment-only shared signal, legit decoys.
    stealth_level: int = 0
    ring_topologies: Optional[List[str]] = None
    ring_signal_keys: Optional[List[str]] = None


TASK_CONFIGS: Dict[str, TaskConfig] = {
    "task_1": TaskConfig(
        task_id="task_1",
        name="Basic Ad Triage",
        difficulty="easy",
        queue_size=5,
        action_budget=25,
        n_legit=2,
        n_fraud=3,
        n_escalate=0,
        include_networks=False,
        n_fraud_rings=0,
        allowed_difficulties=["easy"],
        description=(
            "Learn the investigation loop. Queue of 5 ads with obviously "
            "fraudulent or clearly legitimate signals. Generous budget of 25 "
            "actions (5 per ad). Novice Fraudster: only fake-giveaway and "
            "miracle-cure templates allowed. Capped at 3 proposals so the "
            "queue never exceeds 8 ads (~3 actions per ad even after the "
            "Fraudster maxes out)."
        ),
        max_rounds=4,
        max_proposals=3,
        max_fraudster_actions_per_turn=3,
        max_investigator_actions_per_turn=6,
        allowed_fraud_categories=_TASK_1_ALLOWED_CATEGORIES,
    ),
    "task_2": TaskConfig(
        task_id="task_2",
        name="Sophisticated Fraud Under Budget Pressure",
        difficulty="medium",
        queue_size=12,
        action_budget=30,
        n_legit=5,
        n_fraud=5,
        n_escalate=2,
        include_networks=False,
        n_fraud_rings=0,
        allowed_difficulties=["easy", "medium"],
        description=(
            "Triage under budget constraints. Mix of legit ads, sophisticated "
            "scams, and gray-area cases. 12 ads but only 30 actions (~2.5 per ad). "
            "Agent must prioritize which ads to investigate deeply. "
            "Mid-tier Fraudster: adds counterfeit, clone-brand, advance-fee, "
            "crypto, celebrity-endorsement, and gray-area supplement templates."
        ),
        max_rounds=4,
        max_proposals=6,
        max_fraudster_actions_per_turn=3,
        max_investigator_actions_per_turn=6,
        allowed_fraud_categories=_TASK_2_ALLOWED_CATEGORIES,
        stealth_level=1,
    ),
    "task_3": TaskConfig(
        task_id="task_3",
        name="Coordinated Fraud Network Detection",
        difficulty="hard",
        queue_size=20,
        action_budget=35,
        n_legit=6,
        n_fraud=10,
        n_escalate=4,
        include_networks=True,
        n_fraud_rings=3,
        allowed_difficulties=["hard"],
        description=(
            "Full challenge including coordinated fraud rings. 20 ads with 3 "
            "hidden fraud networks using varied topologies (cliques, chains, "
            "hub-and-spoke). Queue copy is camouflaged (network_* templates "
            "only). Landing pages no longer print scam-similarity scores. "
            "Budget of 35 actions (~1.75 per ad). Ring member ads look "
            "borderline individually — the agent must cross-reference "
            "payment IDs across ads. Sophisticated Fraudster: 5 rounds, "
            "7 proposals."
        ),
        max_rounds=5,
        max_proposals=7,
        max_fraudster_actions_per_turn=3,
        max_investigator_actions_per_turn=7,
        allowed_fraud_categories=None,
        stealth_level=2,
    ),
    # Held-out generalisation eval — same template universe + ring topologies
    # as task_3, but a strictly tighter budget regime (25 ads / 30 actions =
    # ~1.2 actions/ad vs task_3's ~1.75) and one extra ring.  No training
    # seeds in TRAINING_SEED_TIERS — this task's seeds (4001..4005 in
    # eval_suite.EVAL_SEEDS) are reserved for measuring whether the trained
    # Investigator generalises beyond the budget distribution it was trained
    # on, not just to fresh seeds within the same budget.  See
    # ANALYSIS.md §3.1 and ROUND_2_Q5_REALISM_REWARDS_TRAINING.md §5.1.
    "task_3_unseen": TaskConfig(
        task_id="task_3_unseen",
        name="Networks Under Tighter Budget (Held-out Eval)",
        difficulty="hard",
        queue_size=25,
        action_budget=30,
        n_legit=8,
        n_fraud=12,
        n_escalate=5,
        include_networks=True,
        n_fraud_rings=4,
        allowed_difficulties=["hard"],
        description=(
            "Held-out generalisation eval. Same fraud + escalate templates "
            "and ring topologies as task_3, but the budget regime is "
            "deliberately unseen: 25 ads with only 30 actions (~1.2/ad vs "
            "task_3's ~1.75) and 4 hidden rings instead of 3. Used by "
            "eval_suite.run_before_after to test whether the Investigator "
            "learned the underlying detection skill or just over-fit to the "
            "training budget distribution. Never appears in TRAINING_SEED_TIERS."
        ),
        max_rounds=5,
        max_proposals=8,
        max_fraudster_actions_per_turn=3,
        max_investigator_actions_per_turn=7,
        allowed_fraud_categories=None,
        stealth_level=2,
    ),
    "task_4": TaskConfig(
        task_id="task_4",
        name="Camouflaged Chain Rings Under Injection Pressure",
        difficulty="hard",
        queue_size=24,
        action_budget=22,
        n_legit=8,
        n_fraud=12,
        n_escalate=4,
        include_networks=True,
        n_fraud_rings=4,
        allowed_difficulties=["hard"],
        description=(
            "Long-horizon eval. Every ad looks like ordinary ecommerce/SaaS. "
            "Landing pages have valid SSL, aged .com domains, and scam-"
            "similarity < 25%. Rings are chains: A shares a payment ID with "
            "B, B with C — A never shares with C. Two legit ads reuse a ring "
            "registrar as decoys. Fraudster keeps injecting mid-episode "
            "(10 extra proposals). Budget ~0.9 actions/ad. Single-ad "
            "keyword matching cannot solve this; the investigator must "
            "compare payment IDs across turns."
        ),
        max_rounds=6,
        max_proposals=10,
        max_fraudster_actions_per_turn=3,
        max_investigator_actions_per_turn=6,
        allowed_fraud_categories=list(_LEGIT_CAMOUFLAGE),
        stealth_level=3,
        ring_topologies=["chain"],
        ring_signal_keys=["payment_method"],
    ),
}


@dataclass
class CampaignProfile:
    """Campaign-level metadata associated with an ad."""
    objective: str          # e.g. "conversions", "traffic", "awareness", "app_installs"
    bid_strategy: str       # e.g. "lowest_cost", "cost_cap", "bid_cap"
    daily_budget_usd: float
    ad_set_count: int
    placements: List[str]

    def to_investigation_text(self, account_age_days: int) -> str:
        budget_age_ratio = (
            self.daily_budget_usd / max(account_age_days, 1)
        )
        placements_str = ", ".join(self.placements)

        lines = [
            f"Campaign Objective: {self.objective}",
            f"Bid Strategy: {self.bid_strategy}",
            f"Daily Budget: ${self.daily_budget_usd:,.2f} "
            f"(account is {account_age_days} days old — "
            f"budget/age ratio: ${budget_age_ratio:,.2f}/day)",
            f"Active Ad Sets: {self.ad_set_count}",
            f"Placements: {placements_str}",
        ]

        warnings = []
        if budget_age_ratio > 50:
            warnings.append(
                "Budget-to-account-age ratio exceeds typical thresholds."
            )
        if self.ad_set_count > 15:
            warnings.append(
                f"High ad set count ({self.ad_set_count}) — "
                "possible policy evasion testing via creative variation."
            )
        if self.objective in ("traffic", "awareness") and self.bid_strategy == "lowest_cost":
            warnings.append(
                f"Optimizing for {self.objective} with lowest-cost bidding "
                "— common in spray-and-pray fraud campaigns."
            )
        if "Audience Network" in self.placements and len(self.placements) <= 2:
            warnings.append(
                "Heavy reliance on Audience Network placement — "
                "higher bot traffic exposure."
            )

        if warnings:
            for w in warnings:
                lines.append(f"  WARNING: {w}")
        else:
            lines.append("Budget and pacing consistent with historical account behavior.")

        return "\n".join(lines)


@dataclass
class Ad:
    ad_id: str
    ad_copy: str
    category: str
    targeting_summary: str
    initial_risk_signals: List[str]
    ground_truth_label: str  # "fraud", "legit", or "escalate"
    fraud_type: str
    severity: float
    difficulty: str


@dataclass
class GeneratedEpisode:
    """All pre-generated data for one episode."""
    task_config: TaskConfig
    ads: List[Ad]
    advertiser_profiles: Dict[str, AdvertiserProfile]
    campaign_profiles: Dict[str, CampaignProfile]
    landing_pages: Dict[str, LandingPageData]
    fraud_rings: List[FraudRing]
    ad_to_rings: Dict[str, List[str]]
    investigation_data: Dict[str, Dict[str, str]]


def generate_episode(seed: int, task_id: str = "task_1") -> GeneratedEpisode:
    """Generate a complete episode with all pre-computed investigation data."""
    rng = random.Random(seed)
    config = TASK_CONFIGS[task_id]

    ads = _generate_ad_queue(rng, config)

    fraud_ad_ids = [a.ad_id for a in ads if a.ground_truth_label == "fraud"]

    fraud_rings: List[FraudRing] = []
    ad_to_rings: Dict[str, List[str]] = {}
    ring_shared_payments: Dict[str, str] = {}

    if config.include_networks and config.n_fraud_rings > 0:
        fraud_rings, ad_to_rings =         generate_fraud_networks(
            rng, config.n_fraud_rings, fraud_ad_ids,
            topologies=config.ring_topologies,
            signal_keys=config.ring_signal_keys,
        )
        for ring in fraud_rings:
            if "payment_method" in ring.shared_signals:
                for ad_id in ring.member_ad_ids:
                    ring_shared_payments[ad_id] = ring.shared_signals["payment_method"]

    advertiser_profiles: Dict[str, AdvertiserProfile] = {}
    campaign_profiles: Dict[str, CampaignProfile] = {}
    landing_pages: Dict[str, LandingPageData] = {}
    investigation_data: Dict[str, Dict[str, str]] = {}

    ring_campaign_overrides: Dict[str, Dict[str, Any]] = {}
    ring_created_dates: Dict[str, str] = {}
    for ring in fraud_rings:
        shared_objective = rng.choice(["traffic", "awareness"])
        shared_bid = "lowest_cost"
        # Ring members share account creation dates within the same week
        from datetime import date, timedelta
        base_date = date(2026, 4, 6) - timedelta(days=rng.randint(5, 45))
        for ad_id in ring.member_ad_ids:
            ring_campaign_overrides[ad_id] = {
                "objective": shared_objective,
                "bid_strategy": shared_bid,
            }
            offset = timedelta(days=rng.randint(0, 6))
            ring_created_dates[ad_id] = (base_date + offset).isoformat()

    for ad in ads:
        is_fraud = ad.ground_truth_label in ("fraud", "escalate")
        stealth = config.stealth_level >= 2
        stealth_lp = config.stealth_level >= 2 or (
            config.stealth_level >= 1 and is_fraud and rng.random() < 0.5
        )

        profile = generate_advertiser_profile(
            rng, ad.ad_id, is_fraud,
            payment_method_id=ring_shared_payments.get(ad.ad_id),
            ring_created_date=ring_created_dates.get(ad.ad_id),
            stealth=stealth,
        )
        advertiser_profiles[ad.ad_id] = profile

        campaign = _generate_campaign_profile(
            rng, ad, is_fraud,
            ring_overrides=ring_campaign_overrides.get(ad.ad_id),
        )
        campaign_profiles[ad.ad_id] = campaign

        landing_page_kwargs = {}
        if ad.ad_id in ad_to_rings:
            ring = next(r for r in fraud_rings if ad.ad_id in r.member_ad_ids)
            if "domain_registrar" in ring.shared_signals:
                landing_page_kwargs["registrar_override"] = ring.shared_signals["domain_registrar"]
        elif not is_fraud and rng.random() < 0.25:
            landing_page_kwargs["registrar_override"] = rng.choice(_DECOY_REGISTRARS)

        lp = generate_landing_page(
            rng, ad.ad_id, is_fraud, ad.fraud_type,
            stealth=stealth_lp,
            **landing_page_kwargs,
        )
        landing_pages[ad.ad_id] = lp

        inv = {}
        inv["advertiser_history"] = profile.to_investigation_text()
        inv["landing_page"] = lp.to_investigation_text()
        inv["payment_method"] = _generate_payment_investigation(
            rng, profile, ad.ad_id, ad_to_rings, fraud_rings,
            lecture=(config.stealth_level < 1),
        )
        inv["targeting_overlap"] = _generate_targeting_investigation(rng, ad, ads, ad_to_rings, fraud_rings)
        inv["campaign_structure"] = _generate_campaign_investigation(
            rng, ad, campaign, profile, ad_to_rings, fraud_rings,
        )
        inv["policy_classifier"] = _generate_policy_classifier_investigation(
            ad, lp, use_ground_truth=config.stealth_level < 2,
        )
        investigation_data[ad.ad_id] = inv

    if config.stealth_level >= 3:
        fraud_lps = [
            landing_pages[a.ad_id]
            for a in ads
            if a.ground_truth_label == "fraud" and a.ad_id in landing_pages
        ]
        legit_ids = [a.ad_id for a in ads if a.ground_truth_label == "legit"]
        if fraud_lps and legit_ids:
            from dataclasses import replace as _replace
            donor = rng.choice(fraud_lps)
            for ad_id in rng.sample(legit_ids, min(2, len(legit_ids))):
                landing_pages[ad_id] = _replace(
                    landing_pages[ad_id], registrar=donor.registrar
                )
                investigation_data[ad_id]["landing_page"] = (
                    landing_pages[ad_id].to_investigation_text()
                )

    return GeneratedEpisode(
        task_config=config,
        ads=ads,
        advertiser_profiles=advertiser_profiles,
        campaign_profiles=campaign_profiles,
        landing_pages=landing_pages,
        fraud_rings=fraud_rings,
        ad_to_rings=ad_to_rings,
        investigation_data=investigation_data,
    )


def _generate_ad_queue(rng: random.Random, config: TaskConfig) -> List[Ad]:
    """Build the ad queue by sampling from templates."""
    ads: List[Ad] = []
    ad_counter = 0

    legit_templates = [t for t in LEGIT_TEMPLATES]
    fraud_templates = [
        t for t in FRAUD_TEMPLATES
        if t.difficulty in config.allowed_difficulties and t.label == "fraud"
    ]
    escalate_templates = [
        t for t in FRAUD_TEMPLATES
        if t.difficulty in config.allowed_difficulties and t.label == "escalate"
    ]

    if not escalate_templates:
        escalate_templates = [
            t for t in FRAUD_TEMPLATES if t.label == "escalate"
        ]

    for _ in range(config.n_legit):
        template = rng.choice(legit_templates)
        idx = rng.randint(0, len(template.ad_copies) - 1)
        ad_counter += 1
        ads.append(Ad(
            ad_id=f"ad_{ad_counter:03d}",
            ad_copy=template.ad_copies[idx],
            category=_surface_category(template.category, config.stealth_level),
            targeting_summary=template.targeting_hints[idx % len(template.targeting_hints)],
            initial_risk_signals=[] if config.stealth_level >= 1 else list(template.risk_signals),
            ground_truth_label=template.label,
            fraud_type=template.fraud_type,
            severity=template.severity,
            difficulty=template.difficulty,
        ))

    for _ in range(config.n_fraud):
        if fraud_templates:
            template = rng.choice(fraud_templates)
        else:
            template = rng.choice(FRAUD_TEMPLATES)
        idx = rng.randint(0, len(template.ad_copies) - 1)
        ad_counter += 1
        ads.append(Ad(
            ad_id=f"ad_{ad_counter:03d}",
            ad_copy=template.ad_copies[idx],
            category=_surface_category(template.category, config.stealth_level),
            targeting_summary=template.targeting_hints[idx % len(template.targeting_hints)],
            initial_risk_signals=[] if config.stealth_level >= 1 else list(template.risk_signals),
            ground_truth_label="fraud",
            fraud_type=template.fraud_type,
            severity=template.severity,
            difficulty=template.difficulty,
        ))

    for _ in range(config.n_escalate):
        if escalate_templates:
            template = rng.choice(escalate_templates)
            idx = rng.randint(0, len(template.ad_copies) - 1)
            ad_counter += 1
            ads.append(Ad(
                ad_id=f"ad_{ad_counter:03d}",
                ad_copy=template.ad_copies[idx],
                category=_surface_category(template.category, config.stealth_level),
                targeting_summary=template.targeting_hints[idx % len(template.targeting_hints)],
                initial_risk_signals=[] if config.stealth_level >= 1 else list(template.risk_signals),
                ground_truth_label="escalate",
                fraud_type=template.fraud_type,
                severity=template.severity,
                difficulty=template.difficulty,
            ))

    rng.shuffle(ads)

    renumbered = []
    for i, ad in enumerate(ads):
        ad.ad_id = f"ad_{i + 1:03d}"
        renumbered.append(ad)

    return renumbered


def _surface_category(category: str, stealth_level: int) -> str:
    if stealth_level < 2:
        return category
    return _SURFACE_CATEGORY.get(category, category)


def _generate_payment_investigation(
    rng: random.Random,
    profile: AdvertiserProfile,
    ad_id: str,
    ad_to_rings: Dict[str, List[str]],
    fraud_rings: List[FraudRing],
    *,
    lecture: bool = True,
) -> str:
    """Generate payment method investigation text.

    Ring signals are embedded as raw data values (shared payment IDs) without
    explicitly naming other ads. The agent must cross-reference across ads.
    """
    lines = [
        f"Payment Method Analysis for {ad_id}:",
        f"  Method type: {profile.payment_method_type}",
        f"  Payment ID: {profile.payment_method_id}",
    ]

    if lecture and profile.payment_method_type in ("prepaid_card", "crypto", "virtual_card"):
        lines.append(f"  Note: {profile.payment_method_type} payments have elevated fraud correlation in platform data.")

    if profile.previous_violations > 0:
        lines.append(f"  Chargeback/dispute history: {profile.previous_violations} incident(s) on record.")
    else:
        lines.append("  Chargeback/dispute history: Clean record.")

    velocity = rng.randint(1, 5) if ad_id not in ad_to_rings else rng.randint(3, 12)
    lines.append(f"  Payment method added to {velocity} advertiser account(s) in the last 90 days.")

    if profile.account_age_days < 30:
        lines.append(f"  First charge on this method: {profile.account_age_days} days ago.")

    return "\n".join(lines)


def _generate_targeting_investigation(
    rng: random.Random,
    ad: Ad,
    all_ads: List[Ad],
    ad_to_rings: Dict[str, List[str]],
    fraud_rings: List[FraudRing],
) -> str:
    """Generate targeting overlap investigation text.

    Ring members share an exact targeting fingerprint, presented as raw data.
    The agent must compare fingerprints across ads to detect collusion.
    """
    lines = [
        f"Targeting Analysis for {ad.ad_id}:",
        f"  Declared targeting: {ad.targeting_summary}",
    ]

    if ad.ad_id in ad_to_rings:
        ring = next(r for r in fraud_rings if ad.ad_id in r.member_ad_ids)
        if "targeting_overlap" in ring.shared_signals:
            lines.append(f"  Targeting fingerprint: {ring.shared_signals['targeting_overlap']}")
            overlap_pct = rng.randint(85, 98)
            lines.append(f"  Audience overlap with platform average for category: {overlap_pct}%")
        else:
            fingerprint = f"seg_{rng.randint(10000, 99999)}"
            lines.append(f"  Targeting fingerprint: {fingerprint}")
            overlap_pct = rng.randint(20, 55)
            lines.append(f"  Audience overlap with platform average for category: {overlap_pct}%")
    else:
        fingerprint = f"seg_{rng.randint(10000, 99999)}"
        lines.append(f"  Targeting fingerprint: {fingerprint}")
        similar = [a for a in all_ads if a.ad_id != ad.ad_id and a.category == ad.category]
        if similar:
            overlap_pct = rng.randint(30, 65)
            lines.append(f"  {len(similar)} other ad(s) in same category ({ad.category}) in queue.")
            lines.append(f"  Audience overlap with platform average for category: {overlap_pct}%")
        else:
            overlap_pct = rng.randint(10, 40)
            lines.append(f"  Audience overlap with platform average for category: {overlap_pct}%")

    geo_regions = rng.randint(1, 8) if ad.ground_truth_label != "legit" else rng.randint(1, 3)
    lines.append(f"  Geographic regions targeted: {geo_regions}")

    return "\n".join(lines)


def _generate_policy_classifier_investigation(
    ad: Ad,
    landing_page: Optional[LandingPageData] = None,
    *,
    use_ground_truth: bool = True,
) -> str:
    """Mock Llama Guard 3 / Purple Llama classification for the ad.

    Wraps ``policy_classifier_data.classify_ad``.  Deterministic per ad_id
    (seeded RNG inside the classifier), ground-truth correlated, and produces
    the same text shape the Investigator sees for every other investigation
    target.  See ``counterfeint/data/policy_classifier_data.py`` for the
    category taxonomy and marker heuristics.
    """
    from .policy_classifier_data import classify_ad

    landing_text = landing_page.content_summary if landing_page is not None else ""
    result = classify_ad(
        ad_id=ad.ad_id,
        ad_copy=ad.ad_copy,
        landing_page_text=landing_text,
        ground_truth_label=ad.ground_truth_label if use_ground_truth else None,
        fraud_type=ad.fraud_type or None if use_ground_truth else None,
    )
    return result.to_investigation_text()


_LEGIT_OBJECTIVES = ["conversions", "leads", "sales", "app_installs"]
_FRAUD_OBJECTIVES = ["traffic", "awareness", "reach", "engagement"]
_LEGIT_BID_STRATEGIES = ["cost_cap", "bid_cap", "target_cost"]
_FRAUD_BID_STRATEGIES = ["lowest_cost", "lowest_cost", "lowest_cost", "cost_cap"]

_LEGIT_PLACEMENTS = [
    ["Facebook Feed", "Instagram Feed"],
    ["Facebook Feed", "Instagram Feed", "Instagram Stories"],
    ["Facebook Feed"],
    ["Facebook Feed", "Instagram Feed", "Instagram Reels"],
]
_FRAUD_PLACEMENTS = [
    ["Audience Network", "Facebook Feed"],
    ["Audience Network", "Facebook Feed", "Instagram Stories"],
    ["Facebook Feed", "Instagram Feed", "Audience Network", "Messenger"],
    ["Audience Network"],
]


def _generate_campaign_profile(
    rng: random.Random,
    ad: Ad,
    is_fraud: bool,
    *,
    ring_overrides: Optional[Dict[str, Any]] = None,
) -> CampaignProfile:
    """Generate campaign-level metadata for an ad."""
    if is_fraud:
        objective = rng.choice(_FRAUD_OBJECTIVES)
        bid_strategy = rng.choice(_FRAUD_BID_STRATEGIES)
        daily_budget = round(rng.uniform(500, 5000), 2)
        ad_set_count = rng.randint(8, 50)
        placements = rng.choice(_FRAUD_PLACEMENTS)
    else:
        objective = rng.choice(_LEGIT_OBJECTIVES)
        bid_strategy = rng.choice(_LEGIT_BID_STRATEGIES)
        daily_budget = round(rng.uniform(20, 500), 2)
        ad_set_count = rng.randint(1, 5)
        placements = rng.choice(_LEGIT_PLACEMENTS)

    if ring_overrides:
        objective = ring_overrides.get("objective", objective)
        bid_strategy = ring_overrides.get("bid_strategy", bid_strategy)

    return CampaignProfile(
        objective=objective,
        bid_strategy=bid_strategy,
        daily_budget_usd=daily_budget,
        ad_set_count=ad_set_count,
        placements=list(placements),
    )


def _generate_campaign_investigation(
    rng: random.Random,
    ad: Ad,
    campaign: CampaignProfile,
    profile: AdvertiserProfile,
    ad_to_rings: Dict[str, List[str]],
    fraud_rings: List[FraudRing],
) -> str:
    """Generate campaign structure investigation text.

    Ring members share campaign configurations but no explicit cross-references.
    The agent must compare objective/bid/budget patterns across ads.
    """
    lines = [
        f"Campaign Structure Analysis for {ad.ad_id}:",
        campaign.to_investigation_text(profile.account_age_days),
    ]

    config_hash = f"cfg_{hash((campaign.objective, campaign.bid_strategy)) & 0xFFFF:04x}"
    lines.append(f"  Campaign configuration fingerprint: {config_hash}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fraudster-proposal extension (Round 2)
# ---------------------------------------------------------------------------


def _category_to_fraud_template(category: str) -> AdTemplate:
    """Pick the closest matching FRAUD_TEMPLATE for a Fraudster-declared category."""
    for tmpl in FRAUD_TEMPLATES:
        if tmpl.category == category:
            return tmpl
    return FRAUD_TEMPLATES[0]


def generate_proposal_data(
    *,
    rng: random.Random,
    ad_id: str,
    ad_copy: str,
    category: str,
    landing_page_blurb: Optional[str] = None,
    targeting_summary: Optional[str] = None,
    existing_ads: Optional[List[Ad]] = None,
    stealth: bool = False,
) -> Tuple[Ad, Dict[str, str], AdvertiserProfile, CampaignProfile, "LandingPageData"]:
    """
    Build a fully-formed Ad + investigation_data for a Fraudster-proposed ad.

    The Fraudster controls the *surface*: ad_copy, category, landing page blurb,
    targeting summary.  Underlying account / payment / campaign signals are
    sampled from the fraud-mode distribution so the Investigator has a real
    detection task.

    Returns
    -------
    ad
        The Ad object (ground_truth_label="fraud").
    investigation_data
        Dict[str, str] keyed by investigation target name (the 6 canonical
        targets), already rendered to text.
    profile, campaign, landing_page
        The auxiliary data structures, returned in case the caller wants to
        register them on a GeneratedEpisode.
    """
    template = _category_to_fraud_template(category)

    ad = Ad(
        ad_id=ad_id,
        ad_copy=ad_copy.strip()[:2000] if ad_copy else template.ad_copies[0],
        category=category,
        targeting_summary=(
            targeting_summary.strip()[:512]
            if targeting_summary
            else template.targeting_hints[0]
        ),
        initial_risk_signals=[] if stealth else list(template.risk_signals),
        ground_truth_label="fraud",
        fraud_type=template.fraud_type or "fraudster_proposal",
        severity=template.severity if template.severity > 0 else 0.6,
        difficulty=template.difficulty,
    )
    if stealth:
        ad.category = _surface_category(ad.category, stealth_level=2)

    profile = generate_advertiser_profile(rng, ad_id, is_fraud=True, stealth=stealth)
    campaign = _generate_campaign_profile(rng, ad, is_fraud=True)
    landing_page = generate_landing_page(
        rng, ad_id, is_fraud=True, fraud_type=ad.fraud_type, stealth=stealth,
    )

    if landing_page_blurb:
        from dataclasses import replace
        landing_page = replace(
            landing_page,
            content_summary=landing_page_blurb.strip()[:2000],
        )

    siblings = list(existing_ads or [])
    siblings.append(ad)

    investigation_data: Dict[str, str] = {
        "advertiser_history": profile.to_investigation_text(),
        "landing_page": landing_page.to_investigation_text(),
        "payment_method": _generate_payment_investigation(
            rng, profile, ad_id, ad_to_rings={}, fraud_rings=[],
            lecture=not stealth,
        ),
        "targeting_overlap": _generate_targeting_investigation(
            rng, ad, siblings, ad_to_rings={}, fraud_rings=[]
        ),
        "campaign_structure": _generate_campaign_investigation(
            rng, ad, campaign, profile, ad_to_rings={}, fraud_rings=[]
        ),
        "policy_classifier": _generate_policy_classifier_investigation(
            ad, landing_page, use_ground_truth=not stealth,
        ),
    }

    return ad, investigation_data, profile, campaign, landing_page
