"""
Fraud network (ring) generation for Task 3 using networkx.

Generates complex fraud ring topologies, each named after a published Meta
Adversarial Threat Report (Coordinated Inauthentic Behaviour / CIB) case study:

- Clique     - Ghana DigitSol-style: small troll-farm where every account
               amplifies every other (Meta Q3 2020 Adversarial Threat Report).
- Chain      - Benin Digited-style: relay pattern where A promotes B, B promotes
               C, but A never directly touches C (Meta Q1 2021 Adversarial
               Threat Report).
- Hub-spoke  - China-Russia-style: one master account funds and controls many
               satellite accounts (Meta Q3 2022 Adversarial Threat Report).

Individual ads in a ring may look borderline; the signal is in the connections.
Each edge in the graph carries the signal type that connects the two ads.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx


@dataclass
class FraudRing:
    ring_id: str
    member_ad_ids: List[str]
    shared_signals: Dict[str, str]  # signal_type -> shared_value
    topology: str = "clique"  # clique, chain, hub_spoke
    case_name: str = ""       # e.g. "Ghana DigitSol-style"
    provenance: str = ""      # e.g. "Meta Q3 2020 Adversarial Threat Report"

    @property
    def size(self) -> int:
        return len(self.member_ad_ids)


RING_CASE_STUDIES: List[Dict[str, str]] = [
    {
        "topology": "clique",
        "case_name": "Ghana DigitSol-style",
        "provenance": "Meta Q3 2020 Adversarial Threat Report",
        "summary": (
            "Troll-farm ring where every account amplifies every other; "
            "all members share payment / creative / targeting fingerprints."
        ),
    },
    {
        "topology": "chain",
        "case_name": "Benin Digited-style",
        "provenance": "Meta Q1 2021 Adversarial Threat Report",
        "summary": (
            "Relay ring where A promotes B, B promotes C, but A never directly "
            "touches C. Transitive reasoning is required to surface the full "
            "network."
        ),
    },
    {
        "topology": "hub_spoke",
        "case_name": "China-Russia-style hub",
        "provenance": "Meta Q3 2022 Adversarial Threat Report",
        "summary": (
            "Hub-and-spoke ring: one master advertiser funds and controls many "
            "satellite accounts that share the master's payment and registrar."
        ),
    },
]

_RING_TOPOLOGIES = [cs["topology"] for cs in RING_CASE_STUDIES]

_SIGNAL_POOL_KEYS = ["payment_method", "domain_registrar", "creative_template", "targeting_overlap"]

_REGISTRAR_CHOICES = ["Njalla (privacy)", "Epik", "NameSilo", "Tucows (privacy proxy)"]
_TARGETING_CHOICES = [
    "Men 25-45, crypto+investing, US+UK+AU",
    "Adults 18-35, tech+gaming, worldwide",
    "Women 25-55, health+beauty, US+CA",
    "Adults 30-60, finance+real-estate, US+UK",
    "Adults 20-40, e-commerce+dropshipping, US+EU",
]


def _make_signal_pool(rng: random.Random, ring_index: int) -> Dict[str, str]:
    """Generate a pool of shared signal values for one ring."""
    return {
        "payment_method": f"pmt_ring_{rng.randint(10000, 99999)}",
        "domain_registrar": rng.choice(_REGISTRAR_CHOICES),
        "creative_template": f"tmpl_{rng.randint(1000, 9999)}",
        "targeting_overlap": rng.choice(_TARGETING_CHOICES),
    }


def generate_fraud_networks(
    rng: random.Random,
    n_rings: int,
    available_fraud_ad_ids: List[str],
    *,
    topologies: Optional[List[str]] = None,
    signal_keys: Optional[List[str]] = None,
) -> Tuple[List[FraudRing], Dict[str, List[str]]]:
    """
    Generate fraud ring structures with complex topologies.

    Returns:
        rings: list of FraudRing objects
        ad_to_rings: mapping from ad_id to list of ring_ids it belongs to
    """
    G = nx.Graph()
    rings: List[FraudRing] = []
    ad_to_rings: Dict[str, List[str]] = {}

    remaining = list(available_fraud_ad_ids)
    rng.shuffle(remaining)

    for i in range(n_rings):
        if len(remaining) < 3:
            break
        # Reserve 3 ads per still-to-come ring so we always fit n_rings rings,
        # which is what makes the "all three CIB topologies every episode"
        # storytelling claim true at task_3.
        remaining_rings = n_rings - i - 1
        reserved = 3 * remaining_rings
        budget = max(3, len(remaining) - reserved)
        ring_size = rng.randint(3, min(5, budget, len(remaining)))

        members = remaining[:ring_size]
        remaining = remaining[ring_size:]

        if topologies:
            topology = topologies[i % len(topologies)]
            case_study = next(
                (cs for cs in RING_CASE_STUDIES if cs["topology"] == topology),
                RING_CASE_STUDIES[i % len(RING_CASE_STUDIES)],
            )
        else:
            case_study = RING_CASE_STUDIES[i % len(RING_CASE_STUDIES)]
            topology = case_study["topology"]

        signal_pool = _make_signal_pool(rng, i)

        if signal_keys:
            shared_signals = {k: signal_pool[k] for k in signal_keys if k in signal_pool}
        else:
            signal_key_list = list(_SIGNAL_POOL_KEYS)
            rng.shuffle(signal_key_list)
            n_shared = rng.randint(2, len(signal_key_list))
            shared_signals = {k: signal_pool[k] for k in signal_key_list[:n_shared]}

        _add_edges_for_topology(G, members, shared_signals, topology, rng)

        ring_id = f"ring_{i}"
        ring = FraudRing(
            ring_id=ring_id,
            member_ad_ids=members,
            shared_signals=shared_signals,
            topology=topology,
            case_name=case_study["case_name"],
            provenance=case_study["provenance"],
        )
        rings.append(ring)

        for ad_id in members:
            ad_to_rings.setdefault(ad_id, []).append(ring_id)
            G.add_node(ad_id, ring_id=ring_id)

    # Optionally create bridge nodes between rings for extra complexity
    if len(rings) >= 2 and remaining:
        _add_bridge_ads(G, rings, remaining, ad_to_rings, rng)

    return rings, ad_to_rings


def _add_edges_for_topology(
    G: nx.Graph,
    members: List[str],
    shared_signals: Dict[str, str],
    topology: str,
    rng: random.Random,
) -> None:
    """Add edges to the graph based on the ring topology."""
    signal_types = list(shared_signals.keys())

    if topology == "clique":
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                signal = rng.choice(signal_types)
                G.add_edge(a, b, signal_type=signal, signal_value=shared_signals[signal])

    elif topology == "chain":
        for idx in range(len(members) - 1):
            signal = signal_types[idx % len(signal_types)]
            G.add_edge(
                members[idx], members[idx + 1],
                signal_type=signal, signal_value=shared_signals[signal],
            )

    elif topology == "hub_spoke":
        hub = members[0]
        for spoke in members[1:]:
            signal = rng.choice(signal_types)
            G.add_edge(hub, spoke, signal_type=signal, signal_value=shared_signals[signal])


def _add_bridge_ads(
    G: nx.Graph,
    rings: List[FraudRing],
    remaining: List[str],
    ad_to_rings: Dict[str, List[str]],
    rng: random.Random,
) -> None:
    """Optionally link two rings via a shared bridge ad from the remaining pool."""
    if len(remaining) < 1 or len(rings) < 2:
        return

    bridge_ad = remaining.pop(0)
    r1, r2 = rings[0], rings[1]

    bridge_to_r1 = rng.choice(r1.member_ad_ids)
    bridge_to_r2 = rng.choice(r2.member_ad_ids)

    r1.member_ad_ids.append(bridge_ad)
    ad_to_rings.setdefault(bridge_ad, []).extend([r1.ring_id, r2.ring_id])

    sig_key = rng.choice(list(r1.shared_signals.keys()))
    G.add_edge(bridge_ad, bridge_to_r1, signal_type=sig_key, signal_value=r1.shared_signals[sig_key])

    sig_key2 = rng.choice(list(r2.shared_signals.keys()))
    G.add_edge(bridge_ad, bridge_to_r2, signal_type=sig_key2, signal_value=r2.shared_signals[sig_key2])


def get_ring_shared_signal_text(ring: FraudRing) -> str:
    """Describe the shared signals in a ring (for grader/debug use)."""
    header_tail = f"topology={ring.topology}"
    if ring.case_name:
        header_tail = f"{ring.case_name} {ring.topology}"
    lines = [
        f"Fraud Ring {ring.ring_id} ({ring.size} members, {header_tail}):"
    ]
    if ring.provenance:
        lines.append(f"  Modelled after: {ring.provenance}")
    lines.append(f"  Members: {', '.join(ring.member_ad_ids)}")
    lines.append("  Shared signals:")
    for signal_type, value in ring.shared_signals.items():
        lines.append(f"    - {signal_type}: {value}")
    return "\n".join(lines)


def build_ground_truth_graph(rings: List[FraudRing]) -> nx.Graph:
    """Reconstruct the full ground truth network graph from rings.

    Used by graders to compute the expected set of edges.
    """
    G = nx.Graph()
    for ring in rings:
        for i, a in enumerate(ring.member_ad_ids):
            G.add_node(a, ring_id=ring.ring_id)
            for b in ring.member_ad_ids[i + 1:]:
                G.add_edge(a, b, ring_id=ring.ring_id)
    return G
