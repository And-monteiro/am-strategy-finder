"""Experiment proposals. The model only receives metadata and computed summaries."""
import json
import os
import time
import urllib.request

FAMILIES = ["zscore", "return", "sma_distance", "breakout", "volatility"]


def obj(properties):
    return {"type": "object", "additionalProperties": False, "properties": properties,
            "required": list(properties)}


def array(item):
    return {"type": "array", "items": item}


STRING = {"type": "string"}
NUMBER = {"type": "number"}
INTEGER = {"type": "integer"}
SCHEMA = obj({
    "hypothesis": STRING, "parent": {"type": ["integer", "null"]},
    "timeframe": {"type": "string", "enum": ["1h", "4h", "1d"]},
    "indicator": {"type": "string", "enum": FAMILIES},
    "direction": {"type": "string", "enum": ["above", "below"]},
    "lookbacks": array(INTEGER), "entries": array(NUMBER), "exits": array(NUMBER),
    "exit_policies": array({"type": "string", "enum": ["signal", "bracket"]}),
    "holding_periods": array(INTEGER), "stops": array(NUMBER), "targets": array(NUMBER),
    "time_machine_thresholds": array(NUMBER),
    "filter_rule": {"anyOf": [{"type": "null"}, obj({
        "indicator": {"type": "string", "enum": FAMILIES}, "lookback": INTEGER,
        "entry": NUMBER, "exit": NUMBER, "direction": {"type": "string", "enum": ["above", "below"]},
    })]},
})

INSTRUCTIONS = """Design one falsifiable trading experiment using completed candles.
Choose allowed timeframe and give parameter lists for the deterministic search.
Choose exit_policies: ["signal"], ["bracket"], or both to compare exit policies.
For signal, the opposing threshold exits to cash or rotates to asset B. For bracket,
the held position exits only at the stop, take-profit, or optional holding limit.
Both stop and target must be positive. On a single asset, bracket ignores exits;
provide [] for exits if testing only bracket. In rotation, the opposing threshold
is still needed as an entry signal for asset B while flat.
Thresholds are strict comparisons: above enters A above entry and triggers the
opposing condition below exit; below reverses both comparisons. Use ordered
thresholds (entry > exit for above, entry < exit for below).
zscore is (close - mean)/population SD.
return is close / close[n] - 1; sma_distance is close / SMA - 1; breakout compares
close to the highest previous n closes as a fractional distance; volatility is
population SD of the last n one-bar returns. Lookbacks are 2–365. Optional filter_rule
gates only the primary entry on its own entry threshold. With two legs the
indicator uses the asset A / asset B price ratio.
Zero disables hold limit, stop, target, or simulation. Nonzero Time Machine requires
stop, target and max hold 1–48. Brackets and probabilities are fractions. Include
zero Time Machine threshold so its contribution can be compared. Use recent results
to propose a new family or explore a robust region. Set parent to a previous experiment
ID only when refining. You cannot change execution, qualification or data policy."""


def response(model, instruction, payload, schema, deadline, name):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY to use the OpenAI planner, or use --offline.")
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("OpenAI deadline reached.")
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        data=json.dumps({"model": model, "instructions": instruction,
                         "input": json.dumps(payload), "store": False, "max_output_tokens": 4500,
                         "text": {"format": {"type": "json_schema", "name": name,
                                             "schema": schema, "strict": True}}}).encode())
    with urllib.request.urlopen(request, timeout=min(90, remaining)) as handle:
        result = json.load(handle)
    for item in result.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and result.get("status") == "completed":
                return json.loads(content["text"]), {"id": result["id"], "model": result["model"],
                                                     "usage": result.get("usage", {})}
    raise ValueError("OpenAI did not return a completed structured response.")


def offline(index, frames):
    family, direction, entries, exits = [
        ("zscore", "below", [-2, -1.5, -1], [0, .5, 1]),
        ("return", "above", [.01, .025, .05], [-.015, -.005, 0]),
        ("sma_distance", "above", [.01, .02, .03], [-.02, -.01, 0]),
        ("breakout", "above", [0, .002, .005], [-.05, -.025, -.01]),
        ("volatility", "below", [.004, .008, .012], [.015, .025]),
    ][(index // len(frames)) % 5]
    offset = index // (len(frames)*5)
    return {"hypothesis": f"Test {family} threshold stability.", "parent": None,
            "timeframe": frames[index % len(frames)], "indicator": family,
            "direction": direction, "lookbacks": [8+offset, 16+offset, 32+offset],
            "entries": entries, "exits": exits, "holding_periods": [0, 12],
            "stops": [0, .03], "targets": [0, .06], "time_machine_thresholds": [0, .35],
            "filter_rule": None}
