"""Allocate programmatic trials and learn from matched rule changes."""
from collections import Counter
from random import Random
from statistics import median

from .research import encode

COMPONENTS = {
    "confirmation": {"confirmation": "none"},
    "signal_exit": {"policy": "bracket"},
    "entry_gate": {"entry_gate": False},
    "forecast_exit": {"forecast_exit": False},
}


def active_components(design):
    return [name for name, disabled in COMPONENTS.items()
            if any(design[key] != value for key, value in disabled.items())]


def comparison_cells(design, components):
    return [{**design, **{key: value for bit, name in enumerate(components) if not mask & (1 << bit)
                          for key, value in COMPONENTS[name].items()}}
            for mask in range(1 << len(components))]


def plan(summary, history, menu, budget, seed):
    rng = Random(seed)
    observed = {encode(item["design"]): item for item in summary}
    comparisons = [item for batch in history for item in batch.get("comparisons", [])]

    def score(item):
        previous = [row for row in comparisons if row["cells"][-1] == item["design"]]
        harmful = sum(row["verdict"] == "harmful" for row in previous)
        supported = sum(row["verdict"] == "supported" for row in previous)
        return ((item["stable_new"]/item["trials"] + .02*(supported-harmful)),
                median(item["median_window_returns"]), item["median_return"])

    ranked = sorted(summary, key=score, reverse=True)
    leaders = []
    for item in ranked:
        if item["design"]["family"] not in {d["family"] for d in leaders}:
            leaders.append(item["design"])
        if len(leaders) == 2:
            break
    exploration = [d for d in menu if encode(d) not in observed] or menu
    family_trials = Counter()
    for item in summary:
        family_trials[item["design"]["family"]] += item["trials"]
    rng.shuffle(exploration)
    exploration.sort(key=lambda d: family_trials[d["family"]])
    explore = exploration[0]
    base = leaders[0]
    nearby = [d for d in menu if d["family"] == base["family"]
              and sum(d[key] != base[key] for key in base if key != "family") <= 2
              and active_components(d)]
    tried = Counter(encode(row["cells"][-1]) for row in comparisons)
    rng.shuffle(nearby)
    target_width = 2 if len(comparisons) % 2 == 0 else 1
    nearby.sort(key=lambda d: (tried[encode(d)], abs(len(active_components(d))-target_width),
                               d != base, encode(d) in observed))
    target = nearby[0]
    components = active_components(target)
    coverage = Counter(name for row in comparisons for name in row["components"])
    components = sorted(components, key=lambda name: coverage[name])[:2]
    reason = "Isolate component effects and check whether they repeat."
    parent = base
    replications = [row for row in comparisons if row["verdict"] == "supported"
                    and tried[encode(row["cells"][-1])] == 1]
    if replications and len(comparisons) % 3 == 2:
        previous = max(replications, key=lambda row: median(row["window_lift"]))
        target, components = previous["cells"][-1], previous["components"]
        parent = previous["cells"][0]
        reason = "Retest a supported component change on fresh parameter settings."
    width = 1 << len(components)
    matched = max(width, budget//4//width*width)
    explore_budget = budget//4
    refinements = budget-matched-explore_budget
    allocations = [{"design": d, "sampling": "refine", "trials": refinements//len(leaders) + (i < refinements%len(leaders)),
                    "reason": "Refine distinct, stable discoveries across families."}
                   for i, d in enumerate(leaders)]
    allocations.append({"design": explore, "sampling": "broad", "trials": explore_budget,
                        "reason": "Explore an under-tested design in a low-coverage family."})
    allocations.append({"design": target, "sampling": "matched", "trials": matched,
                        "components": components, "parent": parent,
                        "reason": reason,
                        "expectation": "Positive median net return lift in at least two development windows."})
    return {"design": base, "allocations": allocations, "seed": seed}




def paired_evidence(cells, components, groups, activity):
    """Each group has the same underlying numerical parameters in every cell."""
    pairs = [(group[0], group[-1]) for group in groups]
    changed = sum(a["behavior"] != b["behavior"] for a, b in pairs)
    windows = [median(b["windows"][i]-a["windows"][i] for a, b in pairs) for i in range(3)]
    lift = median(b["net_return"]-a["net_return"] for a, b in pairs)
    positive = sum(x > 1e-10 for x in windows)
    negative = sum(x < -1e-10 for x in windows)
    verdict = ("supported" if changed and lift > 0 and positive >= 2 else
               "harmful" if changed and lift < 0 and negative >= 2 else "inconclusive")
    return {"cells": cells, "components": components, "matched_settings": len(groups),
            "changed_histories": changed, "window_lift": windows, "net_return_lift": lift,
            "drawdown_lift": median(b["max_drawdown"]-a["max_drawdown"] for a, b in pairs),
            "trade_delta": median(b["trades"]-a["trades"] for a, b in pairs),
            "exposure_delta": median(b["exposure"]-a["exposure"] for a, b in pairs),
            "activity_by_cell": activity, "verdict": verdict,
            "marginal_return_lift": {name: median(group[mask | (1 << bit)]["net_return"]-group[mask]["net_return"]
                                                for group in groups for mask in range(len(cells)) if not mask & (1 << bit))
                                     for bit, name in enumerate(components)},
            "interaction_return_lift": median(g[3]["net_return"]-g[2]["net_return"]-g[1]["net_return"]+g[0]["net_return"]
                                              for g in groups) if len(cells) == 4 else None}
