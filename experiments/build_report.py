"""Assemble REPORT.md from report/REPORT.template.md by inlining results/<name>.md at {{name}}."""

import os
import re

from common import RESULTS, ROOT


def main():
    tpl = open(os.path.join(ROOT, "report", "REPORT.template.md"), encoding="utf-8").read()

    def sub(m):
        path = os.path.join(RESULTS, m.group(1) + ".md")
        return open(path, encoding="utf-8").read().strip() if os.path.exists(path) else f"*(missing {m.group(1)})*"

    out = re.sub(r"\{\{(\w+)\}\}", sub, tpl)
    with open(os.path.join(ROOT, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(out)
    print("wrote REPORT.md")


if __name__ == "__main__":
    main()
