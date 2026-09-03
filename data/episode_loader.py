"""
Episode loader abstraction.

Decouples the rest of the codebase from the concrete `generate_episode` call
so the Referee, tests, and (future) replay tooling can swap in alternative
sources of episodes (synthetic, file-backed, recorded human, etc.) without
touching environment code.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable
from uuid import uuid4

from .ad_generator import (
    Ad,
    GeneratedEpisode,
    generate_episode,
    generate_proposal_data,
)
from .tool_registry import InvestigationToolRegistry


@runtime_checkable
class EpisodeLoader(Protocol):
    """Pluggable episode source.

    Implementations must accept a `seed` and `task_id` keyword and return a
    fully-formed `GeneratedEpisode` (deterministic given the seed).
    """

    def load(self, *, seed: int, task_id: str) -> GeneratedEpisode: ...


class SyntheticEpisodeLoader:
    """
    Default loader: defers to `data.ad_generator.generate_episode`.

    Used by the InvestigatorEnvironment in standalone (R1) mode and by the
    Referee in Phase 1.
    """

    def __init__(self, *, default_task_id: str = "task_1") -> None:
        self.default_task_id = default_task_id

    def load(
        self, *, seed: Optional[int] = None, task_id: Optional[str] = None
    ) -> GeneratedEpisode:
        effective_seed = seed if seed is not None else hash(uuid4()) & 0xFFFFFFFF
        effective_task_id = task_id or self.default_task_id
        return generate_episode(effective_seed, effective_task_id)


class CallableEpisodeLoader:
    """
    Test/eval loader that wraps an arbitrary callable.

    Useful for golden-replay tests where you want to inject a hand-crafted
    `GeneratedEpisode` without re-running the random sampler.
    """

    def __init__(
        self, fn: Callable[..., GeneratedEpisode]
    ) -> None:
        self._fn = fn

    def load(
        self, *, seed: Optional[int] = None, task_id: Optional[str] = None
    ) -> GeneratedEpisode:
        return self._fn(seed=seed, task_id=task_id)


def extend_episode_with_proposal(
    *,
    episode: GeneratedEpisode,
    registry: InvestigationToolRegistry,
    seed: int,
    ad_copy: str,
    category: str,
    landing_page_blurb: Optional[str] = None,
    targeting_summary: Optional[str] = None,
    ad_id: Optional[str] = None,
) -> Ad:
    """
    Append a Fraudster-proposed ad to `episode` and register its
    investigation data on `registry`.

    Both the episode (canonical state) and the registry (lookup view) are
    updated so the Investigator can immediately investigate the new ad via
    the same code path it uses for synthetic ads.

    Returns the newly-created Ad.
    """
    rng = random.Random(seed)

    # Pick the next free ad_id slot (ad_001, ad_002, ...).
    if ad_id is None:
        existing_ids = {a.ad_id for a in episode.ads} | set(registry.known_ads())
        next_idx = len(episode.ads) + 1
        while True:
            candidate = f"ad_{next_idx:03d}"
            if candidate not in existing_ids:
                ad_id = candidate
                break
            next_idx += 1

    ad, investigation_data, profile, campaign, landing_page = generate_proposal_data(
        rng=rng,
        ad_id=ad_id,
        ad_copy=ad_copy,
        category=category,
        landing_page_blurb=landing_page_blurb,
        targeting_summary=targeting_summary,
        existing_ads=list(episode.ads),
        stealth=episode.task_config.stealth_level >= 2,
    )

    episode.ads.append(ad)
    episode.advertiser_profiles[ad_id] = profile
    episode.campaign_profiles[ad_id] = campaign
    episode.landing_pages[ad_id] = landing_page
    episode.investigation_data[ad_id] = dict(investigation_data)
    registry.register_ad(ad_id, investigation_data)

    return ad


__all__ = [
    "EpisodeLoader",
    "SyntheticEpisodeLoader",
    "CallableEpisodeLoader",
    "extend_episode_with_proposal",
]
