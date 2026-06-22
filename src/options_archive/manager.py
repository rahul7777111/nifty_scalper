from __future__ import annotations

from datetime import date
from typing import Iterable, List, Optional, Sequence

from .collector import OptionsArchiveCollector


class MultiIndexArchiveManager:
    """Coordinate multiple index collectors under a single archive root."""

    def __init__(self, collectors: Sequence[OptionsArchiveCollector]) -> None:
        self.collectors = list(collectors)

    def run_once_all(self, *, force: bool = False) -> List[object]:
        results: List[object] = []
        for collector in self.collectors:
            result = collector.run_once(force=force)
            if result is not None:
                results.append(result)
        return results

    def run_daily_qa(self, session_date: Optional[date] = None) -> List[object]:
        reports: List[object] = []
        for collector in self.collectors:
            report = collector.run_daily_qa(session_date=session_date)
            if report is not None:
                reports.append(report)
        return reports

    def rebuild_feature_layers(self, session_date: Optional[date] = None) -> List[object]:
        written: List[object] = []
        for collector in self.collectors:
            written.extend(collector.storage.rebuild_feature_layer(session_date=session_date, symbol_key=collector.config.symbol_key))
        return written

    def run_forever(self) -> None:
        if not self.collectors:
            return
        while True:
            for collector in self.collectors:
                try:
                    collector.run_once()
                except Exception:
                    collector.logger.exception("Multi-index collector iteration failed for %s", collector.config.symbol_key)
            poll_seconds = min(max(5, int(c.config.scheduler_poll_seconds)) for c in self.collectors)
            import time
            time.sleep(poll_seconds)
