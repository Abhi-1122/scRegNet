"""
Aggregates the plan's Sec 5.3 named biological checks against the `full` model's
full_candidate_predictions.csv output (written by src/train_differential.py) for every
condition pair a cell type was trained on. Prints a pass/fail table; does not train
anything.
"""
import os
import json
import pandas as pd

OUT_ROOT = "out/differential"


def load_preds(cell_type, pair, ablation="full"):
    p = os.path.join(OUT_ROOT, cell_type, pair, ablation, "full_candidate_predictions.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def gene_summary(df, tf, target=None):
    if target is not None:
        sub = df[(df.TF == tf) & (df.Target == target)]
    else:
        sub = df[df.TF == tf]
    if len(sub) == 0:
        return None
    return sub['pred_label'].value_counts().to_dict(), len(sub)


def pct(counts, total, label):
    return 100.0 * counts.get(label, 0) / total if total else 0.0


def check_direction_skew(df, genes, expect_label, min_out_degree=1):
    """For each gene (as TF), report fraction of its out-edges predicted `expect_label`."""
    rows = []
    for g in genes:
        res = gene_summary(df, g)
        if res is None:
            rows.append((g, None, None, "gene not TF / no out-edges in candidate set"))
            continue
        counts, total = res
        frac = pct(counts, total, expect_label)
        rows.append((g, total, frac, counts))
    return rows


def main():
    print("=" * 90)
    print("hESC: pluripotency-to-definitive-endoderm handoff check")
    print("=" * 90)
    hesc_00_96 = load_preds("hESC", "00h_vs_96h")
    hesc_00_12 = load_preds("hESC", "00h_vs_12h")
    hesc_36_72 = load_preds("hESC", "36h_vs_72h")

    print("\n-- POU5F1/NANOG/SOX2 expected A-specific (00h-specific) in 00h_vs_96h --")
    if hesc_00_96 is not None:
        for g, total, frac, counts in check_direction_skew(hesc_00_96, ["POU5F1", "NANOG", "SOX2"], "A-specific"):
            print(f"  {g}: n_out_edges={total}, %A-specific={frac}, breakdown={counts}")
    else:
        print("  [no output yet]")

    print("\n-- SOX17/GATA4/GATA6 expected B-specific (96h-specific) in 00h_vs_96h --")
    if hesc_00_96 is not None:
        for g, total, frac, counts in check_direction_skew(hesc_00_96, ["SOX17", "GATA4", "GATA6"], "B-specific"):
            print(f"  {g}: n_out_edges={total}, %B-specific={frac}, breakdown={counts}")

    print("\n-- SOX17/GATA4/GATA6 expected B-specific (72h-specific) in 36h_vs_72h --")
    if hesc_36_72 is not None:
        for g, total, frac, counts in check_direction_skew(hesc_36_72, ["SOX17", "GATA4", "GATA6"], "B-specific"):
            print(f"  {g}: n_out_edges={total}, %B-specific={frac}, breakdown={counts}")

    print("\n-- EOMES expected B-specific (12h-specific, transient) in 00h_vs_12h --")
    if hesc_00_12 is not None:
        for g, total, frac, counts in check_direction_skew(hesc_00_12, ["EOMES"], "B-specific"):
            print(f"  {g}: n_out_edges={total}, %B-specific={frac}, breakdown={counts}")
    else:
        print("  [no output yet]")

    print("\n" + "=" * 90)
    print("mESC: naive-pluripotency-circuit collapse check")
    print("=" * 90)
    mesc_0_72 = load_preds("mESC", "0h_vs_72h")
    mesc_0_12 = load_preds("mESC", "0h_vs_12h")
    mesc_24_48 = load_preds("mESC", "24h_vs_48h")

    naive_genes = ["NANOG", "POU5F1", "SOX2", "KLF4", "KLF2", "ESRRB", "TBX3", "ZFP42", "PRDM14"]
    print("\n-- naive-circuit genes expected A-specific (0h-specific) in 0h_vs_72h --")
    if mesc_0_72 is not None:
        for g, total, frac, counts in check_direction_skew(mesc_0_72, naive_genes, "A-specific"):
            print(f"  {g}: n_out_edges={total}, %A-specific={frac}, breakdown={counts}")
    else:
        print("  [no output yet]")

    print("\n-- naive-circuit genes expected A-specific (0h-specific) in 0h_vs_12h (fast onset) --")
    if mesc_0_12 is not None:
        for g, total, frac, counts in check_direction_skew(mesc_0_12, naive_genes, "A-specific"):
            print(f"  {g}: n_out_edges={total}, %A-specific={frac}, breakdown={counts}")

    print("\n-- OTX2 expected B-specific (72h-specific, formative marker) in 0h_vs_72h --")
    if mesc_0_72 is not None:
        for g, total, frac, counts in check_direction_skew(mesc_0_72, ["OTX2"], "B-specific"):
            print(f"  {g}: n_out_edges={total}, %B-specific={frac}, breakdown={counts}")

    print("\n" + "=" * 90)
    print("mHSC-E: erythroid commitment checks")
    print("=" * 90)
    mhsc_lt_prog = load_preds("mHSC-E", "LT_HSC_vs_Prog")
    mhsc_lt_hspc = load_preds("mHSC-E", "LT_HSC_vs_HSPC")
    mhsc_hspc_prog = load_preds("mHSC-E", "HSPC_vs_Prog")

    print("\n-- FLAGSHIP: GATA2 -> GATA1 edge, expect differential (not neither/conserved) --")
    for name, df in [("LT_HSC_vs_Prog", mhsc_lt_prog), ("HSPC_vs_Prog", mhsc_hspc_prog)]:
        if df is None:
            print(f"  [{name}: no output yet]")
            continue
        row = df[(df.TF == "GATA2") & (df.Target == "GATA1")]
        if len(row):
            r = row.iloc[0]
            print(f"  {name}: pred_label={r['pred_label']}  probs: conserved={r['prob_conserved']:.3f} "
                  f"A-spec={r['prob_A_specific']:.3f} B-spec={r['prob_B_specific']:.3f} neither={r['prob_neither']:.3f}")
        else:
            print(f"  {name}: GATA2->GATA1 edge not found in candidate predictions (unexpected)")

    print("\n-- GATA2 other out-edges: LT_HSC/HSPC-specific + conserved enrichment expected --")
    for name, df, expect_labels in [("LT_HSC_vs_Prog", mhsc_lt_prog, ["A-specific", "conserved"]),
                                     ("HSPC_vs_Prog", mhsc_hspc_prog, ["A-specific", "conserved"])]:
        if df is None:
            continue
        res = gene_summary(df, "GATA2")
        if res:
            counts, total = res
            frac = sum(pct(counts, total, l) for l in expect_labels)
            print(f"  {name}: n_out_edges={total}, %in{expect_labels}={frac:.1f}, breakdown={counts}")

    print("\n-- SPI1 (PU.1) out-edges: LT_HSC/HSPC-skewed expected --")
    for name, df in [("LT_HSC_vs_Prog", mhsc_lt_prog), ("HSPC_vs_Prog", mhsc_hspc_prog)]:
        if df is None:
            continue
        res = gene_summary(df, "SPI1")
        if res:
            counts, total = res
            frac = pct(counts, total, "A-specific")
            print(f"  {name}: n_out_edges={total}, %A-specific={frac:.1f}, breakdown={counts}")

    print("\n-- GFI1 vs GFI1B out-edges: internal-consistency check (should trend SAME direction as each other's antagonist pairing, i.e. GFI1->LT_HSC/HSPC-specific, GFI1B->Prog-specific/conserved) --")
    for name, df in [("LT_HSC_vs_Prog", mhsc_lt_prog), ("HSPC_vs_Prog", mhsc_hspc_prog)]:
        if df is None:
            continue
        for g in ["GFI1", "GFI1B"]:
            res = gene_summary(df, g)
            if res:
                counts, total = res
                print(f"  {name} {g}: n_out_edges={total}, breakdown={counts}")

    print("\n-- EPOR in-edges (12 candidates) convergence check --")
    epor_regulators = ["CBFB", "ETS1", "GATA2", "GFI1", "GFI1B", "HCFC1", "HDAC2", "IKZF1", "MYC", "SPI1", "STAT3"]
    for name, df in [("LT_HSC_vs_Prog", mhsc_lt_prog), ("HSPC_vs_Prog", mhsc_hspc_prog)]:
        if df is None:
            continue
        sub = df[(df.Target == "EPOR") & (df.TF.isin(epor_regulators))]
        print(f"  {name}: {len(sub)} EPOR in-edges found")
        for _, r in sub.iterrows():
            print(f"    {r['TF']} -> EPOR: {r['pred_label']}")


if __name__ == "__main__":
    main()
