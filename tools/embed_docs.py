import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
docs_dir = ROOT / "docs"
output = ROOT / "website" / "assets" / "js" / "docs-inline.js"

docs_map = {}
for md_file in docs_dir.glob("*.md"):
  docs_map[md_file.stem] = md_file.read_text(encoding="utf-8")

output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
  "window.__scDocs = " + json.dumps(docs_map, ensure_ascii=False, indent=2) + ";\n",
  encoding="utf-8"
)

print(f"Embedded {len(docs_map)} docs into {output.relative_to(ROOT)}")
