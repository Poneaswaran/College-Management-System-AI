"""ASGI entrypoint re-exporting the study material service app."""

from pathlib import Path
import sys

# Ensure sibling packages (api, rag, services, study_material) resolve from any CWD.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
	sys.path.insert(0, str(APP_DIR))

from study_material.main import app