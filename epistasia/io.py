"""
Input / Output utilities for epistasia.

This module handles reading data from disk and converting it into
epistasia core objects (e.g. Landscape), without performing any
analysis or inference.
"""

from __future__ import annotations

import os
import pandas as pd
from typing import Iterable, Optional

from .core import Landscape


# --------------------------------------------------
# Robust table reader
# --------------------------------------------------

def read_table(path: str) -> pd.DataFrame:
    """
    Read a CSV/TSV/TXT table with automatic separator and encoding detection.

    Parameters
    ----------
    path : str
        Path to the input file (.csv, .tsv, .txt).

    Returns
    -------
    df : pandas.DataFrame
        Loaded table.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    UnicodeDecodeError
        If no supported encoding works.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    sep = "\t" if ext == ".tsv" else None

    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, sep=sep, encoding=enc)
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "read_table",
        path,
        0,
        0,
        "Could not decode file with utf-8, cp1252 or latin-1",
    )


# --------------------------------------------------
# Landscape constructors
# --------------------------------------------------

def landscape_from_file(
    path: str,
    *,
    N: Optional[int] = None,
    state_cols: Optional[Iterable[str]] = None,
    replicate_cols: Optional[Iterable[str]] = None,
    validate_binary: bool = True,
) -> Landscape:
    df = read_table(path)

    if state_cols is not None:
        state_cols = list(state_cols)
        missing = [c for c in state_cols if c not in df.columns]
        if missing:
            raise ValueError(f"state_cols not found in file: {missing}")

        # Decide replicate columns
        if replicate_cols is None:
            replicate_cols = [c for c in df.columns if c not in state_cols]
        else:
            replicate_cols = list(replicate_cols)

        # Reorder: states first, then replicates
        df = df.loc[:, state_cols + replicate_cols]

        # Force N to match
        N = len(state_cols)

    return Landscape.from_dataframe(
        df,
        N=N,
        replicate_cols=replicate_cols,
        validate_binary=validate_binary,
    )
