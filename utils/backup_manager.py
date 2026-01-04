"""Utility helpers for CSV backups and retention management."""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List

import structlog

from config.config_loader import get_env


@dataclass(slots=True)
class BackupManager:
    """Handle CSV backups per pipeline step and remove outdated artifacts."""

    backup_root: Path = Path(get_env("BACKUP_DIR", "backups"))
    retention_days: int = int(get_env("BACKUP_RETENTION_DAYS", "30"))
    logger: structlog.stdlib.BoundLogger = field(
        default_factory=lambda: structlog.get_logger("backup_manager")
    )

    def backup_csv_files(self, step_name: str, run_date: str, files: Iterable[Path]) -> List[Path]:
        """Copy CSV files into backups/<run_date>/<step_name>/ with *_backup suffix."""

        backup_dir = self.backup_root / run_date / step_name
        backup_dir.mkdir(parents=True, exist_ok=True)
        created: List[Path] = []
        for source in files:
            source_path = Path(source)
            if not source_path.exists():
                raise FileNotFoundError(f"Cannot backup missing file: {source_path}")
            if source_path.suffix.lower() != ".csv":
                self.logger.warning("non_csv_backup_requested", path=source_path.as_posix())
            target = backup_dir / f"{source_path.stem}_backup{source_path.suffix}"
            shutil.copy2(source_path, target)
            created.append(target)
            self.logger.info(
                "backup_created",
                step=step_name,
                run_date=run_date,
                source=source_path.as_posix(),
                target=target.as_posix(),
            )
        return created

    def cleanup_old_backups(self) -> None:
        """Remove backup folders older than retention_days (based on folder name YYYYMMDD)."""

        if not self.backup_root.exists():
            return
        threshold = date.today() - timedelta(days=self.retention_days)
        for child in self.backup_root.iterdir():
            if not child.is_dir():
                continue
            try:
                child_date = datetime.strptime(child.name, "%Y%m%d").date()
            except ValueError:
                continue
            if child_date <= threshold:
                shutil.rmtree(child, ignore_errors=True)
                self.logger.info("backup_folder_removed", folder=child.as_posix())
