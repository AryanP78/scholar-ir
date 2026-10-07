"""Render report/report_template.md -> report/report_draft.md, pulling every number from results/.

Placeholders:
  {{csv:results/x.csv}}                 the CSV as a markdown table (floats to 3 decimals)
  {{csv:results/x.csv|col=val|cols=a,b}} filtered rows / selected columns
  {{json:results/x.json:key.sub}}       one value from a JSON file
  {{img:results/x.png}}                 a markdown image link (relative to report/)
Missing inputs render as a visible "(not generated yet: ...)" note, so nothing is ever made up.
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TPL = ROOT / "report" / "report_template.md"
OUT = ROOT / "report" / "report_draft.md"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else (f"{v:.3f}" if abs(v) < 1000 else f"{v:,.0f}"))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render(match: re.Match) -> str:
    kind, arg = match.group(1), match.group(2)
    if kind == "csv":
        parts = arg.split("|")
        path = ROOT / parts[0]
        if not path.exists():
            return f"*(not generated yet: `{parts[0]}`)*"
        df = pd.read_csv(path)
        for f in parts[1:]:
            k, v = f.split("=", 1)
            if k == "cols":
                df = df[[c for c in v.split(",") if c in df.columns]]
            elif k == "head":
                df = df.head(int(v))
            else:
                df = df[df[k].astype(str) == v]
        return md_table(df)
    if kind == "json":
        file, key = arg.split(":", 1)
        path = ROOT / file
        if not path.exists():
            return f"*(not generated yet: `{file}`)*"
        val = json.loads(path.read_text())
        for k in key.split("."):
            val = val[k] if isinstance(val, dict) else val[int(k)]
        return f"{val:.3f}" if isinstance(val, float) else str(val)
    if kind == "img":
        path = ROOT / arg
        return f"![{path.stem}](../{arg})" if path.exists() else f"*(figure not generated yet: `{arg}`)*"
    return match.group(0)


def main():
    text = TPL.read_text()
    out = re.sub(r"\{\{(csv|json|img):([^}]+)\}\}", render, text)
    OUT.write_text(out)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
