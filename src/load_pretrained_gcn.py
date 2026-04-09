import os
from collections import OrderedDict

import torch


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ["state_dict", "model_state_dict", "gcn_state_dict", "model"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        return ckpt_obj
    return ckpt_obj


def inspect_model_keys(model, max_keys=80):
    keys = list(model.state_dict().keys())
    print(f"[inspect_model_keys] total keys: {len(keys)}")
    for index, key in enumerate(keys[:max_keys]):
        print(f"  {index:03d}: {key}")
    if len(keys) > max_keys:
        print(f"  ... ({len(keys) - max_keys} more)")


def _candidate_key_maps(source_key):
    candidates = [source_key]
    prefixes = [
        "module.",
        "model.",
        "backbone.",
        "gcn.",
        "encoder.",
        "encoder.gcn.",
    ]

    for prefix in prefixes:
        if source_key.startswith(prefix):
            stripped = source_key[len(prefix):]
            candidates.append(stripped)
            if not stripped.startswith("convs.") and stripped[0].isdigit():
                candidates.append(f"convs.{stripped}")

    if source_key.startswith("convs."):
        candidates.append(source_key[len("convs."):])

    if source_key and source_key[0].isdigit():
        candidates.append(f"convs.{source_key}")

    return candidates


def load_pretrained_gcn(model, ckpt_path, map_location=None, verbose=True):
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Pretrained GCN checkpoint not found: {ckpt_path}")

    if map_location is None:
        map_location = "cpu"

    load_kwargs = {"map_location": map_location}
    if "weights_only" in torch.load.__code__.co_varnames:
        load_kwargs["weights_only"] = False

    raw_ckpt = torch.load(ckpt_path, **load_kwargs)
    src_state = _extract_state_dict(raw_ckpt)
    dst_state = model.state_dict()

    mapped = OrderedDict()
    for src_key, value in src_state.items():
        for cand in _candidate_key_maps(src_key):
            if cand in dst_state and cand.startswith("convs."):
                if tuple(dst_state[cand].shape) == tuple(value.shape):
                    mapped[cand] = value
                    break

    if len(mapped) == 0:
        raise RuntimeError(
            "No compatible GCN parameters were found in checkpoint. "
            "Use inspect_model_keys(model) to inspect model key names."
        )

    load_result = model.load_state_dict(mapped, strict=False)

    if verbose:
        missing = [k for k in load_result.missing_keys if k.startswith("convs.")]
        print(f"[load_pretrained_gcn] loaded {len(mapped)} GCN tensors from {ckpt_path}")
        if missing:
            print(f"[load_pretrained_gcn] missing GCN keys: {missing[:10]}")
            if len(missing) > 10:
                print(f"[load_pretrained_gcn] ... and {len(missing) - 10} more")

    return mapped
