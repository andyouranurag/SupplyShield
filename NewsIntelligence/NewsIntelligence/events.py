"""Step 3: find disruption events using keyword lists (whole-word matching).
Each event type has a weight = how serious it usually is for supply chains (0 to 1).
Edit this file to add or tune keywords."""
import re

EVENT_KEYWORDS = {
    "strike_labor":     (0.7, ["workers strike", "on strike", "general strike", "port strike", "rail strike", "truckers strike",
                               "dockworkers", "walkout", "labor dispute", "labour dispute", "union protest", "work stoppage"]),
    "port_closure":     (0.9, ["port closure", "port closed", "port shut", "port congestion", "terminal closed", "canal blocked",
                               "suez canal", "ever given", "blockade", "stranded ship", "container ship"]),
    "shortage":         (0.8, ["shortage", "out of stock", "supply crunch", "chip shortage", "stockout", "scarcity", "empty shelves"]),
    "natural_disaster": (0.9, ["flood", "flooding", "earthquake", "typhoon", "hurricane", "cyclone", "wildfire", "tsunami", "drought", "storm"]),
    "geopolitical":     (0.8, ["sanction", "embargo", "tariff", "trade war", "export ban", "import ban", "border closure", "invasion"]),
    "factory_outage":   (0.8, ["factory fire", "plant shutdown", "factory shutdown", "production halt", "plant closed", "lockdown",
                               "factories closed", "factory closed"]),
    "logistics_delay":  (0.6, ["supply chain", "delay", "delayed", "backlog", "bottleneck", "late delivery", "container shortage",
                               "freight rates", "shipping costs"]),
    "demand_shock":     (0.4, ["demand surge", "panic buying", "demand collapse", "demand drop"]),
}

# whole-word patterns (so "war" never matches "warns"); optional plural "s"
_PATTERNS = {
    event: (w, [(k, re.compile(r"\b" + re.escape(k) + r"s?\b")) for k in words])
    for event, (w, words) in EVENT_KEYWORDS.items()
}


def find_events(text):
    """Return (list of event types, list of matched phrases, max event weight)."""
    events, phrases, weight = [], [], 0.0
    for event, (w, pats) in _PATTERNS.items():
        hits = [k for k, p in pats if p.search(text)]
        if hits:
            events.append(event)
            phrases.extend(hits)
            weight = max(weight, w)
    return events, phrases, weight
