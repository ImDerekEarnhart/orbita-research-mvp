from __future__ import annotations

import re


MAX_CSV_UPLOAD_BYTES = 50 * 1024 * 1024

ALLOWED_FILE_MIME_TYPES = {
    "",
    "application/csv",
    "application/octet-stream",
    "application/vnd.ms-excel",
    "text/csv",
    "text/plain",
    "text/x-csv",
}

DANGEROUS_EXTENSIONS = {
    "bat",
    "bin",
    "cmd",
    "com",
    "dll",
    "dmg",
    "elf",
    "exe",
    "jar",
    "js",
    "jsx",
    "msi",
    "php",
    "ps1",
    "py",
    "scr",
    "sh",
    "so",
    "tar",
    "tgz",
    "ts",
    "vbs",
    "war",
    "z",
    "zip",
    "7z",
    "rar",
}

_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._() -]*$")
_FILENAME_PARAM_RE = re.compile(
    r"(?:^|;)\s*filename\*?\s*=\s*(\"(?:[^\"\\]|\\.)*\"|[^;]*)",
    re.IGNORECASE,
)
_HTML_OR_SCRIPT_RE = re.compile(r"^<(?:!doctype|html|script|svg|xml)\b", re.IGNORECASE)
_MAGIC_PREFIXES = (
    b"MZ",
    b"\x7fELF",
    b"PK\x03\x04",
    b"PK\x05\x06",
    b"PK\x07\x08",
    b"\x1f\x8b",
    b"%PDF",
    b"\x89PNG",
    b"\xff\xd8\xff",
    b"Rar!",
    b"7z\xbc\xaf",
)


class UploadSafetyError(ValueError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def filename_from_content_disposition(disposition: str | None, fallback: str | None) -> str | None:
    for match in _FILENAME_PARAM_RE.finditer(disposition or ""):
        raw = match.group(1).strip()
        if raw.startswith('"') and raw.endswith('"'):
            raw = raw[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        if raw:
            return raw
    return fallback


def validate_filename(filename: str | None) -> str:
    raw = str(filename or "")
    if not raw or len(raw) > 120:
        raise UploadSafetyError(400, "Invalid upload")
    if raw != raw.strip():
        raise UploadSafetyError(400, "Invalid upload")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in raw):
        raise UploadSafetyError(400, "Invalid upload")
    if "/" in raw or "\\" in raw or ":" in raw:
        raise UploadSafetyError(400, "Invalid upload")
    if ".." in raw or raw.startswith("."):
        raise UploadSafetyError(400, "Invalid upload")
    if not _FILENAME_RE.fullmatch(raw):
        raise UploadSafetyError(400, "Invalid upload")

    parts = raw.lower().split(".")
    if len(parts) < 2 or parts[-1] != "csv":
        raise UploadSafetyError(400, "Invalid upload")
    if any(part in DANGEROUS_EXTENSIONS for part in parts[:-1]):
        raise UploadSafetyError(400, "Invalid upload")
    return raw


def validate_mime_type(content_type: str | None) -> None:
    media_type = (content_type or "").split(";", 1)[0].strip().lower()
    if media_type not in ALLOWED_FILE_MIME_TYPES:
        raise UploadSafetyError(400, "Invalid upload")


def sniff_csv_content(content: bytes) -> None:
    if not content:
        raise UploadSafetyError(400, "Invalid upload")

    sample = content[:8192]
    if any(sample.startswith(prefix) for prefix in _MAGIC_PREFIXES):
        raise UploadSafetyError(400, "Invalid upload")
    if sample.startswith(b"#!"):
        raise UploadSafetyError(400, "Invalid upload")
    if b"\x00" in sample:
        raise UploadSafetyError(400, "Invalid upload")

    controls = sum(1 for byte in sample if byte < 32 and byte not in (9, 10, 13))
    if controls / max(len(sample), 1) > 0.01:
        raise UploadSafetyError(400, "Invalid upload")

    try:
        text = sample.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UploadSafetyError(400, "Invalid upload") from exc
    if _HTML_OR_SCRIPT_RE.match(text.lstrip()):
        raise UploadSafetyError(400, "Invalid upload")
    if not any(delimiter in text for delimiter in (",", "\t", ";")):
        raise UploadSafetyError(400, "Invalid upload")


def validate_csv_upload(filename: str | None, content_type: str | None, content: bytes) -> str:
    if len(content) > MAX_CSV_UPLOAD_BYTES:
        raise UploadSafetyError(413, "File too large")
    safe_name = validate_filename(filename)
    validate_mime_type(content_type)
    sniff_csv_content(content)
    return safe_name
