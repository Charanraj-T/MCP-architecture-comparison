"""Shared fixtures for MCP Architecture Review tests."""
import sys
from pathlib import Path

import pytest

# Ensure common/ and experiment packages are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
