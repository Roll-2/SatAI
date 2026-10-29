"""App settings — persistence for the optional OpenRouter API key.

The key lives in ``Config/settings.local.json`` (gitignored: it is
user-specific and secret). The app treats the shipped *placeholder* key as
"no key configured" until the user pastes their real OpenRouter key
(created in 2 minutes on openrouter.ai/keys).
"""

from __future__ import annotations

import json
from pathlib import Path

from Src.utils.constants import PROJECT_ROOT

SETTINGS_PATH: Path = PROJECT_ROOT / "Config" / "settings.local.json"

_KEY = "openrouter_api_key"
_MODELS = "openrouter_models"

#: Placeholder shipped in Config/settings.local.json. The app refuses to use
#: it and reports "no key" until it is replaced with a real OpenRouter key.
PLACEHOLDER_KEY = "sk-or-v1-SOSTITUISCI-QUESTA-CHIAVE-CON-QUELLA-VERA"

#: Free vision-capable models, tried in order until one answers. Editable in
#: Config/settings.local.json under "openrouter_models" (catalog changes
#: often: openrouter.ai/models, filter free + image input).
DEFAULT_MODELS: list[str] = [
    "google/gemma-4-31b-it:free",          # primary: strongest free vision
    "inclusionai/ling-3.0-flash-vl:free",  # fallback: different provider
    "thinkingmachines/inkling:free",       # fallback: third provider
    "openrouter/free",                     # last resort: any free model
]


def _read_all() -> dict:
    """Load the whole settings file (empty dict if missing or corrupt)."""
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_all(data: dict) -> None:
    """Persist the whole settings file with user-only permissions."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:  # user-only perms — the file holds a secret
        SETTINGS_PATH.chmod(0o600)
    except OSError:
        pass


def get_api_key() -> str:
    """Return the stored OpenRouter API key ('' if missing or placeholder)."""
    key = str(_read_all().get(_KEY, "")).strip()
    if not key or key == PLACEHOLDER_KEY:
        return ""
    return key


def set_api_key(key: str) -> None:
    """Store (or clear, with empty string) the OpenRouter API key."""
    data = _read_all()
    key = (key or "").strip()
    if key:
        data[_KEY] = key
    else:
        data.pop(_KEY, None)
    _write_all(data)


def has_api_key() -> bool:
    """True when a real OpenRouter key is configured."""
    return bool(get_api_key())


def get_models() -> list[str]:
    """Return the fallback model chain (stored value or the default)."""
    stored = _read_all().get(_MODELS)
    if (isinstance(stored, list) and stored
            and all(isinstance(m, str) and m.strip() for m in stored)):
        return [m.strip() for m in stored]
    return list(DEFAULT_MODELS)
