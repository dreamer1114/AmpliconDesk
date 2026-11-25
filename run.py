"""AmpliconDesk 启动入口：python run.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amplicondesk.web.serve import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
