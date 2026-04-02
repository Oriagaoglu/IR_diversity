"""Utilities for loading, joining, and cleaning DiverseSumm + DSGlobal data."""

import json
import os
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def load_diversesumm(path=None):
    """Load DiverseSumm dataset from JSON files."""
    if path is None:
        path = DATA_RAW / "diversesumm"
    # Will be implemented after data exploration
    raise NotImplementedError


def load_dsglobal(path=None):
    """Load DSGlobal dataset."""
    if path is None:
        path = DATA_RAW / "newscope"
    raise NotImplementedError


def load_event_mapping(path=None):
    """Load the event mapping between DSGlobal and DiverseSumm."""
    if path is None:
        path = DATA_PROCESSED / "event_mapping.json"
    with open(path) as f:
        return json.load(f)
