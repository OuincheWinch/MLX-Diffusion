from fastapi import APIRouter

import civitai_service
import hf_service
from state import DATA_DIR, TokenRequest

router = APIRouter(tags=["tokens"])

CIVITAI_TOKEN_FILE = DATA_DIR / "civitai_token.txt"
HF_TOKEN_FILE = DATA_DIR / "hf_token.txt"


@router.get("/api/civitai/token")
def get_civitai_token():
    """Check if a Civitai API key is set and return a masked preview."""
    key = civitai_service.get_civitai_api_key()
    if not key:
        return {"configured": False, "masked": None}
    masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
    return {"configured": True, "masked": masked}


@router.post("/api/civitai/token")
def set_civitai_token(req: TokenRequest):
    """Save or clear Civitai API key in backend/data/civitai_token.txt."""
    tok = req.token.strip()
    if not tok:
        if CIVITAI_TOKEN_FILE.exists():
            try:
                CIVITAI_TOKEN_FILE.unlink()
            except OSError:
                pass
        return {"configured": False, "masked": None}
    CIVITAI_TOKEN_FILE.write_text(tok, "utf-8")
    masked = tok[:4] + "..." + tok[-4:] if len(tok) > 8 else "***"
    return {"configured": True, "masked": masked}


@router.get("/api/hf/token")
def get_hf_token():
    """Check if a Hugging Face token is set and return a masked preview."""
    key = hf_service.get_hf_token()
    if not key:
        return {"configured": False, "masked": None}
    masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
    return {"configured": True, "masked": masked}


@router.post("/api/hf/token")
def set_hf_token(req: TokenRequest):
    """Save or clear Hugging Face token in backend/data/hf_token.txt."""
    tok = req.token.strip()
    if not tok:
        if HF_TOKEN_FILE.exists():
            try:
                HF_TOKEN_FILE.unlink()
            except OSError:
                pass
        return {"configured": False, "masked": None}
    HF_TOKEN_FILE.write_text(tok, "utf-8")
    masked = tok[:4] + "..." + tok[-4:] if len(tok) > 8 else "***"
    return {"configured": True, "masked": masked}
