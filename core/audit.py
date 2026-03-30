from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import settings


class AuditLogger:
    def __init__(self, log_path: str | None = None) -> None:
        self.log_path = Path(log_path or settings.audit_log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def tail(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        rows = []
        for line in lines[-limit:]:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return list(reversed(rows))


audit_logger = AuditLogger()
