"""Development-only context for adaptive experiment proposals."""
from . import planner

ACTIONS = ("refinement", "simplification", "new_hypothesis")
RICH_SCHEMA = planner.obj({**planner.SCHEMA["properties"],
    "action": {"type": "string", "enum": list(ACTIONS)},
    "expected_change": planner.STRING,
})
RICH_INSTRUCTIONS = planner.INSTRUCTIONS + """
For feedback-driven research, set action to the requested action when provided.
Refinements and simplifications must name a visible parent experiment. State one
specific expected improvement and why. Keep unchanged rules as a comparison when
possible. Use weaknesses and parameter sensitivity as well as leading results.
If a family has plateaued, simplify it or explore another idea. A new hypothesis
must have a null parent and must change the indicator or rule structure rather
than just nudge a threshold. Never request final-period results.
"""


def development_score(metrics):
    """Fixed, lexicographic policy for picking a candidate at a checkpoint."""
    periods = metrics["periods"]
    positive = sum(p["return"] > 0 for p in periods) / max(1, len(periods))
    sharpe = metrics["sharpe"]
    return (sum(metrics["funnel"].values()), sharpe if sharpe is not None else -1e6,
            positive, metrics["net_return"], -metrics["max_drawdown"])


def brief(candidate):
    metrics = candidate["development"]
    return {
        "experiment": candidate["experiment"], "strategy": candidate["strategy"],
        "net_return": metrics["net_return"], "drawdown": metrics["max_drawdown"],
        "sharpe": metrics["sharpe"], "trades": metrics["trades"],
        "period_returns": [p["return"] for p in metrics["periods"]],
        "funnel": metrics["funnel"],
    }


def rich_context(experiments, candidates):
    families = {}
    for candidate in candidates:
        spec = candidate["strategy"]
        rule = spec["rule"]
        family = (spec["timeframe"], rule["indicator"], rule["direction"], spec.get("exit_policy", "signal"))
        families.setdefault(family, []).append(candidate)
    best, weak, progress = [], [], []
    for family, members in families.items():
        ordered = sorted(members, key=lambda c: development_score(c["development"]), reverse=True)
        best.append(ordered[0])
        failures = [c for c in reversed(ordered) if c["development"]["trades"] >= 3
                    and c["development"]["outcome"] != "qualified"]
        if failures:
            weak.append(failures[0])
        best_so_far = None
        last_gain = None
        for candidate in members:
            score = development_score(candidate["development"])
            if best_so_far is None or score > best_so_far:
                best_so_far, last_gain = score, candidate["experiment"]
        progress.append({"family": family, "trials": len(members), "last_gain_experiment": last_gain,
                         "qualified": sum(c["development"]["outcome"] == "qualified" for c in members)})
    weak.sort(key=lambda c: development_score(c["development"]))
    diverse_failures = []
    used_failures = set()
    for candidate in weak:
        failure = next(stage for stage, passed in candidate["development"]["funnel"].items() if not passed)
        if failure not in used_failures:
            diverse_failures.append(candidate)
            used_failures.add(failure)
    distinct = {id(c) for c in diverse_failures}
    diverse_failures.extend(c for c in weak if id(c) not in distinct)
    best.sort(key=lambda c: development_score(c["development"]), reverse=True)
    recent = []
    for exp in experiments[-10:]:
        p = exp["proposal"]
        recent.append({"experiment": exp["experiment"], "hypothesis": p["hypothesis"],
                       "parent": p.get("parent"), "action": p.get("action"),
                       "expected_change": p.get("expected_change"), "timeframe": p["timeframe"],
                       "indicator": p["indicator"], "funnel": exp["funnel"],
                       "leader": exp["top_parameters"][0] if exp["top_parameters"] else None})
    return {"last_experiments": recent, "strong_candidates": [brief(c) for c in best[:3]],
            "weak_candidates": [brief(c) for c in diverse_failures[:3]],
            "family_progress": progress}


def requested_action(number, has_history):
    return "new_hypothesis" if not has_history or number % 4 == 0 else None


def validate_action(proposal, context, action):
    selected = proposal.get("action")
    parent = proposal.get("parent")
    if selected not in ACTIONS or (action and selected != action):
        raise ValueError("Proposal did not use the requested search action.")
    visible = {exp["experiment"] for exp in context["last_experiments"]}
    visible.update(c["experiment"] for c in context["strong_candidates"] + context["weak_candidates"])
    if selected == "new_hypothesis":
        if parent is not None:
            raise ValueError("A new hypothesis cannot name a parent.")
    elif parent not in visible:
        raise ValueError("A refinement or simplification must name a visible parent.")
