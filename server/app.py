# OpenEnv validator expects server/app.py at the root server/ path.
# This module re-exports the real FastAPI app from the package.
from pipeline_debug_env.server.app import app

__all__ = ["app"]
