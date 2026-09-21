from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

import generator
from state import (
    _discover_local_loras,
    _init_gallery_index,
)
from routers import jobs, gallery, loras, tokens, uploads, downloads, settings as settings_router


import threading


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Discovery on startup
    _init_gallery_index()
    threading.Thread(target=_discover_local_loras, daemon=True).start()
    yield
    # Clean shutdown: terminate GPU daemons and reclaim unified memory
    try:
        generator._kill_sdxl_daemon()
        generator._drop_mflux_pipeline()
    except Exception:
        pass


app = FastAPI(title="MLX-DIFFUSION", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount domain routers
app.include_router(jobs.router)
app.include_router(gallery.router)
app.include_router(loras.router)
app.include_router(tokens.router)
app.include_router(uploads.router)
app.include_router(downloads.router)
app.include_router(settings_router.router)
