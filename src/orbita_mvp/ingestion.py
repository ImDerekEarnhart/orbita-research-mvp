from __future__ import annotations

import hashlib
import json
import mimetypes
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


class IngestionError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_profiles": [],
        "duplicates": int(df.duplicated().sum()) if len(df) else 0,
    }
    for raw_name in df.columns:
        name = str(raw_name)
        s = df[raw_name]
        missing = int(s.isna().sum())
        unique = int(s.nunique(dropna=True))
        role = "measurement"
        kind = "text"
        stats: dict[str, Any] = {}
        numeric = pd.to_numeric(s, errors="coerce")
        numeric_fraction = float(numeric.notna().mean()) if len(s) else 0.0
        if numeric_fraction >= 0.9 and unique > 0:
            kind = "numeric"
            vals = numeric.dropna().astype(float)
            if len(vals):
                stats = {
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "mean": float(vals.mean()),
                    "median": float(vals.median()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                }
        else:
            normalized_hint = name.lower().replace(" ", "_")
            parsed_dt = None
            if any(token in normalized_hint for token in ("date", "time", "timestamp")):
                parsed_dt = pd.to_datetime(s, errors="coerce", utc=True)
            if parsed_dt is not None and len(s) and float(parsed_dt.notna().mean()) >= 0.9:
                kind = "datetime"
                role = "time"
            elif unique <= max(20, int(len(s) * 0.1)):
                kind = "categorical"
                role = "group_or_category"
                counts = s.astype(str).value_counts(dropna=False).head(12)
                stats = {"top_values": {str(k): int(v) for k, v in counts.items()}}
        normalized = name.lower().replace(" ", "_")
        if unique == len(s) and len(s) > 3:
            if any(token in normalized for token in ("id", "uuid", "subject", "patient", "sample")):
                role = "identifier"
        if any(token in normalized for token in ("date", "time", "timestamp")):
            role = "time"
        if any(token in normalized for token in ("label", "class", "group", "condition", "diagnosis", "treatment")):
            role = "group_or_category"
        profile["column_profiles"].append(
            {
                "name": name,
                "kind": kind,
                "inferred_role": role,
                "missing": missing,
                "missing_fraction": float(missing / len(s)) if len(s) else 0.0,
                "unique": unique,
                "numeric_fraction": numeric_fraction,
                "stats": stats,
            }
        )
    return profile


class ArtifactIngestor:
    """Preserve uploads, extract supported content, and create deterministic profiles."""

    def __init__(self, max_unpacked_bytes: int = 250_000_000):
        self.max_unpacked_bytes = max_unpacked_bytes

    def ingest(self, source: str | Path, destination_dir: str | Path) -> dict[str, Any]:
        source = Path(source)
        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        safe_name = source.name.replace("/", "_").replace("\\", "_")
        stored = destination_dir / safe_name
        if source.resolve() != stored.resolve():
            shutil.copy2(source, stored)
        suffix = stored.suffix.lower()
        media_type = mimetypes.guess_type(stored.name)[0] or "application/octet-stream"
        base = {
            "original_name": source.name,
            "stored_path": str(stored.resolve()),
            "media_type": media_type,
            "size_bytes": stored.stat().st_size,
            "sha256": sha256_file(stored),
            "parse_status": "preserved",
            "artifact_kind": "unknown",
            "profile": {},
            "extracted_path": None,
            "error": None,
        }
        try:
            if suffix in {".csv", ".tsv"}:
                df = pd.read_csv(stored, sep="\t" if suffix == ".tsv" else ",")
                return self._table_result(base, df, destination_dir, stored.stem)
            if suffix in {".xlsx", ".xlsm", ".xls"}:
                sheets = pd.read_excel(stored, sheet_name=None)
                if not sheets:
                    raise IngestionError("Workbook has no readable sheets")
                selected_name, df = max(sheets.items(), key=lambda item: len(item[1]))
                result = self._table_result(base, df, destination_dir, stored.stem)
                result["profile"]["selected_sheet"] = str(selected_name)
                result["profile"]["sheet_names"] = [str(x) for x in sheets]
                return result
            if suffix == ".parquet":
                df = pd.read_parquet(stored)
                return self._table_result(base, df, destination_dir, stored.stem)
            if suffix == ".jsonl":
                rows = [json.loads(line) for line in stored.read_text(encoding="utf-8").splitlines() if line.strip()]
                if rows and all(isinstance(row, dict) for row in rows):
                    return self._table_result(base, pd.DataFrame(rows), destination_dir, stored.stem)
                return self._text_result(base, json.dumps(rows, indent=2), destination_dir, stored.stem)
            if suffix in {".json", ".ipynb"}:
                obj = json.loads(stored.read_text(encoding="utf-8"))
                if isinstance(obj, list) and obj and all(isinstance(row, dict) for row in obj):
                    return self._table_result(base, pd.DataFrame(obj), destination_dir, stored.stem)
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, list) and value and all(isinstance(row, dict) for row in value):
                            result = self._table_result(base, pd.DataFrame(value), destination_dir, stored.stem)
                            result["profile"]["json_record_key"] = key
                            return result
                return self._text_result(base, json.dumps(obj, indent=2), destination_dir, stored.stem)
            if suffix in {".txt", ".md", ".py", ".r", ".tex"}:
                return self._text_result(base, stored.read_text(encoding="utf-8", errors="replace"), destination_dir, stored.stem)
            if suffix == ".pdf":
                from pypdf import PdfReader

                reader = PdfReader(str(stored))
                text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
                result = self._text_result(base, text, destination_dir, stored.stem)
                result["profile"]["pages"] = len(reader.pages)
                return result
            if suffix == ".docx":
                from docx import Document

                document = Document(str(stored))
                text = "\n".join(p.text for p in document.paragraphs)
                return self._text_result(base, text, destination_dir, stored.stem)
            if suffix == ".zip":
                return self._zip_result(base, stored, destination_dir)
        except Exception as exc:
            base["parse_status"] = "partially_parsed" if base["size_bytes"] else "failed"
            base["error"] = f"{type(exc).__name__}: {exc}"
            return base
        base["parse_status"] = "unsupported"
        base["error"] = "File preserved but no safe parser is registered for this format."
        return base

    def _table_result(self, base: dict[str, Any], df: pd.DataFrame, destination_dir: Path, stem: str) -> dict[str, Any]:
        normalized = destination_dir / f"{stem}.normalized.csv"
        df.to_csv(normalized, index=False)
        base.update(
            {
                "parse_status": "parsed",
                "artifact_kind": "table",
                "extracted_path": str(normalized.resolve()),
                "profile": profile_dataframe(df),
            }
        )
        return base

    def _text_result(self, base: dict[str, Any], text: str, destination_dir: Path, stem: str) -> dict[str, Any]:
        extracted = destination_dir / f"{stem}.extracted.txt"
        extracted.write_text(text, encoding="utf-8")
        words = text.split()
        base.update(
            {
                "parse_status": "parsed" if text.strip() else "partially_parsed",
                "artifact_kind": "text",
                "extracted_path": str(extracted.resolve()),
                "profile": {
                    "characters": len(text),
                    "words": len(words),
                    "lines": len(text.splitlines()),
                    "empty": not bool(text.strip()),
                },
            }
        )
        return base

    def _zip_result(self, base: dict[str, Any], stored: Path, destination_dir: Path) -> dict[str, Any]:
        extract_dir = destination_dir / f"{stored.stem}_unpacked"
        extract_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        members: list[dict[str, Any]] = []
        with zipfile.ZipFile(stored) as archive:
            for info in archive.infolist():
                total += info.file_size
                if total > self.max_unpacked_bytes:
                    raise IngestionError("ZIP exceeds safe unpacked-size limit")
                target = (extract_dir / info.filename).resolve()
                if extract_dir.resolve() not in target.parents and target != extract_dir.resolve():
                    raise IngestionError("ZIP contains an unsafe path")
            archive.extractall(extract_dir)
        for path in sorted(extract_dir.rglob("*")):
            if path.is_file():
                members.append({"name": str(path.relative_to(extract_dir)), "size_bytes": path.stat().st_size})
        base.update(
            {
                "parse_status": "parsed",
                "artifact_kind": "archive",
                "extracted_path": str(extract_dir.resolve()),
                "profile": {"members": members[:200], "member_count": len(members), "unpacked_bytes": total},
            }
        )
        return base
