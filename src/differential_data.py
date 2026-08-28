"""
Condition/barcode-prefix registry for the differential-GRN task (DIFFERENTIAL_GRN_PLAN.md
Sec 4.4). Encodes, per cell type, which barcode prefix in BL--ExpressionData.csv's column
header identifies each named condition (time point / differentiation stage), and the 9
condition pairs chosen in Sec 2 of the plan (3 per cell type, each with a stated biological
rationale, not "all pairwise combinations").

Prefixes were verified directly against the column headers of
data/{cell_type}/TFs+500/BL--ExpressionData.csv (2026-08-28), not assumed from the brief.
"""

import pandas as pd

# cell_type -> {condition_name: barcode_prefix}
# Matching is via column_name.startswith(prefix).
CONDITION_REGISTRY = {
    "hESC": {
        "00h": "H9_00hb4s",
        "12h": "H9_12h",
        "24h": "H9_24h",
        "36h": "H9_36h",
        "72h": "H9_72h",
        "96h": "H9_96h",
    },
    "mESC": {
        "0h": "RamDA_mESC_00h",
        "12h": "RamDA_mESC_12h",
        "24h": "RamDA_mESC_24h",
        "48h": "RamDA_mESC_48h",
        "72h": "RamDA_mESC_72h",
    },
    "mHSC-E": {
        "LT_HSC": "LT_HSC_",
        "HSPC": "HSPC_",
        "Prog": "Prog_",
    },
}

# Expected cell counts per condition (TFs+500), verified against CSV headers -- used only
# as a sanity-check assertion in get_condition_columns, not as ground truth for anything.
EXPECTED_COUNTS = {
    "hESC": {"00h": 92, "12h": 102, "24h": 66, "36h": 172, "72h": 138, "96h": 188},
    "mESC": {"0h": 90, "12h": 68, "24h": 90, "48h": 82, "72h": 91},
    "mHSC-E": {"LT_HSC": 141, "HSPC": 384, "Prog": 546},
}

# 9 condition pairs total (3 per cell type), each with a stated rationale (plan Sec 2).
# priority: "primary" (bookend, largest contrast) or "secondary".
CONDITION_PAIRS = {
    "hESC": [
        ("00h", "96h", "primary", "pluripotent vs. committed definitive endoderm (largest contrast)"),
        ("00h", "12h", "secondary", "early transition: fast exit from pluripotency / mesendoderm onset"),
        ("36h", "72h", "secondary", "late commitment: mesendoderm/mesoderm progenitor -> definitive endoderm"),
    ],
    "mESC": [
        ("0h", "72h", "primary", "naive ESC vs. most-differentiated sampled time point"),
        ("0h", "12h", "secondary", "immediate LIF/2i-withdrawal response, naive-circuit collapse onset"),
        ("24h", "48h", "secondary", "mid-transition, formative markers (Otx2) becoming dominant"),
    ],
    "mHSC-E": [
        ("LT_HSC", "Prog", "primary", "most primitive vs. most committed state (141 vs 546 cells)"),
        ("LT_HSC", "HSPC", "secondary", "early loss-of-stemness step"),
        ("HSPC", "Prog", "secondary", "erythroid-commitment step itself (GATA2/GATA1, PU.1/GATA antagonisms)"),
    ],
}


def all_pairs():
    """Return flat list of (cell_type, condition_a, condition_b, priority, rationale)."""
    out = []
    for cell_type, pairs in CONDITION_PAIRS.items():
        for (a, b, priority, rationale) in pairs:
            out.append((cell_type, a, b, priority, rationale))
    return out


def get_condition_columns(columns, cell_type, condition):
    """
    columns: iterable of column names (e.g. data_input.columns from BL--ExpressionData.csv)
    Returns list of column names belonging to `condition` for `cell_type`.
    """
    if cell_type not in CONDITION_REGISTRY:
        raise KeyError(f"Unknown cell_type '{cell_type}' (known: {list(CONDITION_REGISTRY)})")
    if condition not in CONDITION_REGISTRY[cell_type]:
        raise KeyError(f"Unknown condition '{condition}' for cell_type '{cell_type}' "
                        f"(known: {list(CONDITION_REGISTRY[cell_type])})")
    prefix = CONDITION_REGISTRY[cell_type][condition]
    matched = [c for c in columns if str(c).startswith(prefix)]
    expected = EXPECTED_COUNTS.get(cell_type, {}).get(condition)
    if expected is not None and len(matched) != expected:
        raise AssertionError(
            f"Condition column count mismatch for {cell_type}/{condition}: "
            f"matched {len(matched)}, expected {expected} (prefix='{prefix}')"
        )
    return matched


def load_condition_subsets(data_input: pd.DataFrame, cell_type, condition_a, condition_b):
    """
    data_input: genes x cells DataFrame (BL--ExpressionData.csv, index_col=0).
    Returns (df_A, df_B): genes x cells_A / genes x cells_B sub-DataFrames, same gene row
    order as data_input (so row index i refers to the same gene in both).
    """
    cols_a = get_condition_columns(data_input.columns, cell_type, condition_a)
    cols_b = get_condition_columns(data_input.columns, cell_type, condition_b)
    return data_input[cols_a], data_input[cols_b]
