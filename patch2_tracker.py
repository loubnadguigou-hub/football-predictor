p = "src/results_tracker.py"
s = open(p, encoding="utf-8").read()

if 'b.get("type") == "full_time_total"' in s:
    print("Already patched - nothing to do.")
    raise SystemExit

old1 = '    total_blocks = [b for b in blocks if "total" in str(b.get("type", ""))] or blocks\n'
new1 = (
    '    total_blocks = [b for b in blocks if b.get("type") == "full_time_total"]\n'
    '    if not total_blocks:\n'
    '        total_blocks = [b for b in blocks if "total" in str(b.get("type", ""))][:1] or blocks[:1]\n'
)
old2 = '    return pd.DataFrame(rows)\n'
new2 = (
    '    df = pd.DataFrame(rows)\n'
    '    if not df.empty:\n'
    '        df = df.drop_duplicates(subset="team", keep="first").reset_index(drop=True)\n'
    '    return df\n'
)

if old1 not in s or s.count(old2) != 1:
    print("Cannot patch: unexpected file content. Send me src/results_tracker.py")
    raise SystemExit(1)

s = s.replace(old1, new1).replace(old2, new2)
open(p, "w", encoding="utf-8").write(s)
print("PATCH 2 OK")