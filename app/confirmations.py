"""Remember explicit, same-name confirmations without creating path overrides."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from .config import safe_media_path
from .media_text import cached_dialogue
from .spanmatch import reference_id
from .textsim import norm


def identity(series, file, references):
    local = ((file.get("evidence") or {}).get("combined") or {}).get("local") or {}
    codes = file.get("expected") or []
    if (len(codes) != 1 or file.get("scope") != "full" or file.get("language")
            or not local.get("trusted") or not local.get("whole_file")):
        return None
    ref = local.get("reference")
    text = (references.get(ref) or {}).get("text")
    if not text or reference_id(text) != local.get("reference_id"):
        return None
    try:
        root = Path(series["path"]).resolve()
        path = safe_media_path(root / file["rel"])
        if not path.is_relative_to(root) or not path.is_file():
            return None
        st = path.stat()
        if (st.st_size, st.st_mtime_ns) != (file.get("size"), file.get("mtime_ns")):
            return None
        data = cached_dialogue(path) or {}
    except (OSError, ValueError):
        return None
    runtime = file.get("duration")
    if (data.get("scope") == "sample" or data.get("language") or not runtime
            or runtime != data.get("duration")
            or not str(data.get("source") or "").startswith(("remote ", "embedded ", "whisper "))):
        return None
    digest = sha256()
    for cue in data.get("cues") or []:
        digest.update(norm(cue[2]).strip().encode())
        digest.update(b"\n")
    if not data.get("cues"):
        return None
    return {"codes": codes, "file": [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_dev, st.st_ino],
            "duration": runtime, "dialogue": digest.hexdigest(), "reference": ref,
            "reference_id": local["reference_id"]}


def record(series, file, references):
    proof = identity(series, file, references)
    return {"proof": proof, "date": datetime.now(timezone.utc).date().isoformat()} if proof else None
