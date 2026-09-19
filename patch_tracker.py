p = "src/results_tracker.py"
s = open(p, encoding="utf-8").read()

if "_parse_form_standings" in s:
    print("Already patched - nothing to do.")
    raise SystemExit

a = s.index("    data = resp.json()")
end_marker = "    return pd.DataFrame(rows)"
b = s.index(end_marker, a) + len(end_marker)
s = s[:a] + "    return _parse_form_standings(resp.json())" + s[b:]

helper = '''def _parse_form_standings(data: dict) -> pd.DataFrame:
    """Parses the real Sportradar structure (season_form_standings / full_time_total)."""
    blocks = data.get("season_form_standings") or data.get("season_form_standing") or []
    if isinstance(blocks, dict):
        blocks = [blocks]
    total_blocks = [b for b in blocks if "total" in str(b.get("type", ""))] or blocks

    rows = []
    for block in total_blocks:
        for group in block.get("groups", []):
            for standing in group.get("form_standings", []):
                comp = standing.get("competitor", {})
                rows.append({
                    "team": comp.get("name"),
                    "played": standing.get("played", 0),
                    "win": standing.get("win", 0),
                    "draw": standing.get("draw", 0),
                    "loss": standing.get("loss", 0),
                    "goals_for": standing.get("goals_for", 0),
                    "goals_against": standing.get("goals_against", 0),
                    "form": standing.get("form", ""),
                })
    return pd.DataFrame(rows)


'''
m = s.index("def deduce_round_results")
s = s[:m] + helper + s[m:]

open(p, "w", encoding="utf-8").write(s)
print("PATCHED OK")