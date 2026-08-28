"""Registry of tunable experiments. To tune a different architecture/config,
add a new entry here -- everything else (manifest generation, parallel
workers, race-condition handling, result reporting) is shared and unchanged.

Each entry is (fixed_args, search_space_cls):
  - fixed_args: CLI-flag-equivalent overrides that DEFINE this experiment
    (which architecture/features are on) -- never sampled, always applied.
  - search_space_cls: one of the classes in src/optuna/search_space.py,
    reused as-is so the tuned ranges stay consistent with the project's
    original baseline search.
"""
from src.optuna.search_space import GNN_HP_search
# GAT_HP_search is available in src.optuna.search_space for a future GAT-backbone entry.

# Settings shared by every experiment below (dataset, epoch budget, eval
# cadence). Override per-experiment via fixed_args if a specific experiment
# needs something different.
COMMON_ARGS = {
    "cell_type": "hESC",
    "num_TF": "500",
    "gnn_type": "GCN",
    "llm_type": "Geneformer",
    "gnn_epochs": 300,
    "gnn_eval_interval": 5,
    "type": "MLP",
    "flag": True,
    "loop": False,
    "random_seed": 42,
}

EXPERIMENTS = {
    "baseline": {
        "fixed_args": {},
        "search_space_cls": GNN_HP_search,
    },
    "panda_exp_b": {
        # PANDA per-gene profile replaces scFM entirely (no GRN change).
        "fixed_args": {"node_feature_source": "panda"},
        "search_space_cls": GNN_HP_search,
    },
    "panda_exp_c": {
        # Full scFM+GNN pipeline + PANDA (TF,target) edge score as an extra
        # MLP feature -- the only experiment that beat baseline untuned.
        "fixed_args": {"use_panda_edge": True},
        "search_space_cls": GNN_HP_search,
    },
    "panda_exp_d": {
        # scFM + PANDA edge feature, GNN encoding skipped entirely.
        "fixed_args": {"ablation": "no_grn", "use_panda_edge": True},
        "search_space_cls": GNN_HP_search,
    },
    "ppi_method1": {
        # Dual GNN encoder (GRN + PPI branches), late attention fusion.
        "fixed_args": {"use_ppi": True, "ppi_fusion_type": "attention"},
        "search_space_cls": GNN_HP_search,
    },
    "ppi_method2": {
        # Multi-view: GRN/PPI branches interleaved layer-by-layer.
        "fixed_args": {"use_ppi": True, "multi_view": True,
                        "ppi_fusion_type": "attention", "view_guidance": "symmetric"},
        "search_space_cls": GNN_HP_search,
    },
    "scgpt": {
        "fixed_args": {"llm_type": "scGPT"},
        "search_space_cls": GNN_HP_search,
    },
    "uce": {
        "fixed_args": {"llm_type": "UCE"},
        "search_space_cls": GNN_HP_search,
    },
}


def get_experiment(name):
    if name not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment '{name}'. Registered: {sorted(EXPERIMENTS)}")
    return EXPERIMENTS[name]
