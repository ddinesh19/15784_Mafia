"""Offline diagnostics for Mafia_LocalLLM_Comparison.ipynb.

Put this file beside the notebook/results and run in a new notebook cell:
    %run mafia_existing_results_diagnostics.py
Or paste this entire file into one code cell. No earlier notebook cells are needed.
Command line: python mafia_existing_results_diagnostics.py
Reads JSONL only; does not contact Ollama, rerun games, or modify any files.
"""
from pathlib import Path
from collections import Counter, defaultdict
import json
import math

# Change only these settings if your filename or condition labels differ.
RESULTS_PATH = Path("mafia_reliability_sweep_50.jsonl")
BASELINE_CONDITION = "Algorithmic"
LLM_CONDITION = "Qwen3 8B town"
MAX_TRACE_EXAMPLES = 3
TOLERANCE = 1e-6


def _table(title, records):
    print("\n" + title)
    if not records:
        print("No eligible records.")
        return
    try:
        from IPython import get_ipython
        import pandas as pd
        if get_ipython() is not None:
            from IPython.display import display
            with pd.option_context("display.max_columns", None, "display.max_rows", 100):
                display(pd.DataFrame(records).round(4))
            return
    except ImportError:
        pass
    for record in records:
        print(json.dumps(record, ensure_ascii=False))


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _wilson(wins, n):
    if not n:
        return None
    z = 1.959963984540054
    p = wins / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [round(center-half, 4), round(center+half, 4)]


def _lookup(mapping, player):
    return mapping[str(player)] if str(player) in mapping else mapping[player]


def _normalize_beliefs(values):
    if len(values) != 4 or any(type(v) not in (int, float) or not math.isfinite(v)
                               or not 0 <= v <= 1 for v in values):
        raise ValueError("Invalid saved completed-game probabilities")
    total = sum(values)
    if abs(total-1) > 0.001 + 1e-9:
        raise ValueError("Saved completed-game belief sum exceeds runner tolerance")
    return [v / total for v in values]  # Same scoring tolerance as existing runner.


def _support_mass(beliefs, target, signal):
    return max(0.0, 1-beliefs[target]) if signal == 1 else beliefs[target]


def _accepted_vote_request(row, player):
    matches = [request for request in row.get("requests", [])
               if request.get("status") == "ok"
               and request.get("observation", {}).get("phase") == "vote"
               and request["observation"].get("player_id") == player]
    return matches[-1] if matches else None


def load_diagnostic_results(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot find {path.resolve()}. Put this file beside the results, "
            "or set RESULTS_PATH to the full JSONL path.")
    records = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on line {line_number}; file may be incomplete") from error
    if not records or records[0].get("type") != "metadata":
        raise ValueError("Expected the experiment metadata as the first JSONL record")
    metadata, rows = records[0], records[1:]
    spec = metadata["experiment"]
    planned = {(label, q, spec["seed"]+i)
               for label, *_ in spec["conditions"] for q in spec["reliabilities"]
               for i in range(spec["num_games"])}
    seen = set()
    for row in rows:
        key = (row["condition"], row["reliability"], row["seed"])
        if key in seen or key not in planned:
            raise ValueError(f"Duplicate or unplanned game: {key}")
        seen.add(key)
        if row.get("type") != "game" or row.get("status") not in ("ok", "failed"):
            raise ValueError(f"Unexpected game record: {key}")
        if row["status"] == "failed" and "metrics" in row:
            raise ValueError(f"Failed game contains scored metrics: {key}")
        if row["rounds"] != spec["rounds"]:
            raise ValueError(f"Round count differs from metadata: {key}")
    return metadata, rows


def score_completed_game(row):
    truth = row["ground_truth_for_evaluation_only"]
    mafia, detective = truth["mafia"], truth["detective"]
    town = [p for p in range(4) if p != mafia]
    votes = {p: _lookup(row["votes"], p) for p in range(4)}
    if any(type(v) is not int or v not in range(4) or p == v for p, v in votes.items()):
        raise ValueError("Illegal saved vote")
    counts = Counter(votes.values())
    leaders = [p for p, count in counts.items() if count == max(counts.values())]
    town_win = int(leaders == [mafia])
    if town_win != row["metrics"]["village_win"]:
        raise ValueError(f"Winner reconstruction disagrees at seed {row['seed']}")
    player_scores = []
    for player in town:
        beliefs = _normalize_beliefs(_lookup(row["private_beliefs"], player))
        top = max(beliefs)
        tied = [p for p, value in enumerate(beliefs) if value == top]
        player_scores.append({
            "player": player,
            "role": "Detective" if player == detective else "Villager",
            "brier": sum((value-int(p == mafia))**2 for p, value in enumerate(beliefs)),
            "identification": 1/len(tied) if mafia in tied else 0,
            "voting_accuracy": int(votes[player] == mafia),
            "vote_in_top_beliefs": int(votes[player] in tied),
            "mafia_in_top_beliefs": int(mafia in tied),
        })
    result = {"town_win": town_win, "tied_vote": int(len(leaders) > 1),
              "town_votes_on_mafia": sum(votes[p] == mafia for p in town),
              "all_town_votes_agree": int(len({votes[p] for p in town}) == 1),
              "all_town_top_sets_include_mafia_but_loss": int(
                  not town_win and all(p["mafia_in_top_beliefs"] for p in player_scores)),
              "players": player_scores}
    for role in ("Villager", "Detective"):
        group = [p for p in player_scores if p["role"] == role]
        for metric in ("brier", "identification", "voting_accuracy", "vote_in_top_beliefs"):
            result[f"{role.lower()}_{metric}"] = _mean([p[metric] for p in group])
    return result


def run_existing_results_diagnostics(path=RESULTS_PATH):
    metadata, rows = load_diagnostic_results(path)
    spec = metadata["experiment"]
    print("OFFLINE EXISTING-RESULTS DIAGNOSTICS — no model calls or file changes")
    print("Source:", Path(path).resolve())
    print("Protocol:", spec["protocol"], "| rounds:", spec["rounds"],
          "| model:", metadata.get("llm_settings", {}).get("model"))
    print("Rates are proportions (0–1). Brier score sums four squared errors; lower is better.")
    groups = defaultdict(list)
    scored = {}
    for row in rows:
        groups[(row["condition"], row["reliability"])].append(row)
        if row["status"] == "ok":
            scored[(row["condition"], row["reliability"], row["seed"])] = score_completed_game(row)

    summaries, failures, quality, support, request_audit = [], [], [], [], []
    support_examples = []
    for label, *_ in spec["conditions"]:
        for q in spec["reliabilities"]:
            group = groups[(label, q)]
            completed = [r for r in group if r["status"] == "ok"]
            n, done = len(group), len(completed)
            scores = [scored[(label, q, r["seed"])] for r in completed]
            wins = sum(s["town_win"] for s in scores)
            summaries.append({"condition": label, "q": q, "planned": spec["num_games"],
                              "attempted": n, "completed": done, "failed": n-done,
                              "unrun": spec["num_games"]-n, "town_wins": wins,
                              "completion": _rate(done, n), "win_given_completion": _rate(wins, done),
                              "win_95pct_wilson": _wilson(wins, done),
                              "valid_wins_per_attempt": _rate(wins, n),
                              "failed_outcome_lower_bound": _rate(wins, n),
                              "failed_outcome_upper_bound": _rate(wins+n-done, n)})
            for (kind, category), count in Counter(
                    (r.get("failure_kind", "unclassified"), r.get("failure_category", "unspecified"))
                    for r in group if r["status"] == "failed").items():
                failures.append({"condition": label, "q": q, "kind": kind,
                                 "category": category, "failed_games": count})
            if scores:
                metrics = [m for m in scores[0] if m != "players"]
                quality.append({"condition": label, "q": q, "completed": done,
                                **{m: _mean([s[m] for s in scores]) for m in metrics}})
            # Request IDs reset for every game; never combine decisions across games.
            decisions = defaultdict(list)
            for row in group:
                for request in row.get("requests", []):
                    if "decision_id" in request and "attempt" in request:
                        decisions[(row["seed"], request["decision_id"])].append(request)
            if decisions:
                checked = []; recovered = 0; accepted = 0
                for attempts in decisions.values():
                    attempts.sort(key=lambda r: r["attempt"])
                    first = attempts[0]
                    if first["attempt"] == 0 and first.get("validation_checked"):
                        checked.append(first)
                    success = any(r.get("status") == "ok" for r in attempts)
                    accepted += success
                    recovered += first.get("status") == "invalid" and success
                request_audit.append({"condition": label, "q": q, "decisions_started": len(decisions),
                                      "first_attempts_checked": len(checked),
                                      "first_attempt_validity": _rate(sum(r["status"] == "ok" for r in checked), len(checked)),
                                      "accepted_decisions": accepted, "recovered_after_retry": recovered,
                                      "retry_requests": sum(r["attempt"] > 0 for a in decisions.values() for r in a)})
            if math.isclose(q, 1.0, abs_tol=1e-12):
                cases = []
                for row in completed:
                    truth = row["ground_truth_for_evaluation_only"]
                    target, signal, signal_q = truth["investigation"][0]
                    if signal_q != q or signal != int(target == truth["mafia"]):
                        raise ValueError("Perfect-signal truth mismatch")
                    beliefs = _normalize_beliefs(_lookup(row["private_beliefs"], truth["detective"]))
                    mass = _support_mass(beliefs, target, signal)
                    req = _accepted_vote_request(row, truth["detective"])
                    # For algorithmic policies there is no saved model observation to audit.
                    obs_match = None if req is None else (
                        req["observation"].get("role") == "Detective"
                        and req["observation"].get("investigations") == truth["investigation"]
                        and req["observation"].get("reliability") == q)
                    response_match = None
                    if req is not None:
                        response = json.loads(req["response"])
                        response_match = all(abs(a-b) <= TOLERANCE for a, b in zip(
                            _normalize_beliefs(response["beliefs"]), beliefs))
                    case = {"condition": label, "q": q, "seed": row["seed"],
                            "detective": truth["detective"], "mafia": truth["mafia"],
                            "target": target, "signal": signal, "beliefs": beliefs,
                            "forbidden_mass": mass, "violation": mass > TOLERANCE,
                            "saved_observation_matches_signal": obs_match,
                            "accepted_response_matches_saved_beliefs": response_match}
                    cases.append(case)
                    if mass > TOLERANCE:
                        support_examples.append((case, row, req))
                for signal_filter in (None, 0, 1):
                    subset = [c for c in cases if signal_filter is None or c["signal"] == signal_filter]
                    support.append({"condition": label, "signal": "all" if signal_filter is None else signal_filter,
                                    "completed_detectives": len(subset),
                                    "violations": sum(c["violation"] for c in subset),
                                    "violation_rate": _mean([c["violation"] for c in subset]),
                                    "mean_forbidden_mass": _mean([c["forbidden_mass"] for c in subset]),
                                    "observations_checked": sum(c["saved_observation_matches_signal"] is not None for c in subset),
                                    "observation_mismatches": sum(c["saved_observation_matches_signal"] is False for c in subset),
                                    "response_mismatches": sum(c["accepted_response_matches_saved_beliefs"] is False for c in subset)})

    paired = []
    for q in spec["reliabilities"]:
        by_condition = {label: {r["seed"]: r for r in groups[(label, q)] if r["status"] == "ok"}
                        for label in (BASELINE_CONDITION, LLM_CONDITION)}
        baseline, llm = by_condition[BASELINE_CONDITION], by_condition[LLM_CONDITION]
        common = sorted(baseline.keys() & llm.keys())
        differences = []; discordant_llm = discordant_baseline = 0
        for seed in common:
            if baseline[seed]["ground_truth_for_evaluation_only"] != llm[seed]["ground_truth_for_evaluation_only"]:
                raise ValueError(f"Environment mismatch for paired seed {seed}, q={q}")
            a = scored[(BASELINE_CONDITION, q, seed)]["town_win"]
            b = scored[(LLM_CONDITION, q, seed)]["town_win"]
            differences.append(b-a)
            discordant_llm += b == 1 and a == 0
            discordant_baseline += a == 1 and b == 0
        paired.append({"q": q, "matched_completed_seeds": len(common),
                       "baseline_win_rate_matched": _mean([scored[(BASELINE_CONDITION, q, s)]["town_win"] for s in common]),
                       "llm_win_rate_matched": _mean([scored[(LLM_CONDITION, q, s)]["town_win"] for s in common]),
                       "paired_difference_llm_minus_baseline": _mean(differences),
                       "llm_only_wins": discordant_llm, "baseline_only_wins": discordant_baseline})

    _table("1. Completion and town outcomes", summaries)
    _table("2. Failed-game categories", failures)
    _table("3. Completed-game beliefs, voting and coordination", quality)
    _table("4. Perfect-signal consistency (q=1 only)", support)
    _table("5. Matched completed-seed comparison", paired)
    _table("6. Model request and retry diagnostics (includes failed games)", request_audit)
    print("\nInterpretation: completed-game results and matched pairs can have completion-selection bias.")
    print("Bounds concern failed attempted games only; unrun games are excluded. No failed game is scored as a Mafia win.")
    print("Argmax disagreement need not be irrational: maximizing team victory can require coordination.")
    print("The all-town top-set metric includes ties and is not evidence of certain correct identification.")
    print("Wilson intervals are per-cell descriptive intervals, not paired significance tests.")
    print("\n7. Exact-signal contradiction traces (chronological first cases; not a representative sample)")
    for case, row, req in support_examples[:MAX_TRACE_EXAMPLES]:
        print("\nCASE", json.dumps(case))
        print("Votes:", json.dumps(row["votes"]), "| Town win:", row["metrics"]["village_win"])
        print("Public history:", json.dumps(row["public_messages"]))
        if req is not None:
            print("Exact accepted vote observation:", json.dumps(req["observation"]))
            print("Accepted raw response:", req["response"])
            print("Retry attempt:", req.get("attempt"))
    if not support_examples:
        print("No completed q=1 support violations found.")
    print("\nDone. All tables are also available in diagnostic_results.")
    return {"metadata": metadata, "summary": summaries, "failures": failures,
            "quality": quality, "perfect_signal": support, "paired": paired,
            "requests": request_audit, "support_cases": [c for c, _, _ in support_examples]}


# Works both as a pasted notebook cell and via %run / python.
if __name__ == "__main__":
    diagnostic_results = run_existing_results_diagnostics(RESULTS_PATH)
