#!/bin/bash
# Phase 2 reduced core matrix (DIFFERENTIAL_GRN_PLAN.md Sec 6, reduced per work.md's scoping
# entry). Fixed hyperparameters throughout -- the same HP set that reproduces this
# codebase's existing hESC/TFs+500 static baseline (0.8874/0.6225 AUROC/AUPRC), found by
# the pre-existing Optuna search on `summer26`, reused as-is per this pass's "no new HP
# search" instruction.
set -e
cd /home/abhishekg/Documents/Research/scRegNet

PY=python3.12
HP="--gnn_type GCN --llm_type Geneformer --type MLP \
  --gnn_hidden_dims 227 29 34 --mlp_hidden_dims 198 \
  --gnn_num_layers 3 --mlp_num_layers 1 \
  --dropout 0.17477888900593389 --gnn_lr 0.0014411386699467767 --gnn_weight_decay 1.0449337997892844e-05 \
  --gnn_dim_hidden 256 --batch_size 193 \
  --gnn_epochs 300 --gnn_eval_interval 5 --single_gpu 0 --random_seed 0"

run_full () {
  ct=$1; ca=$2; cb=$3; ablation=$4
  pairdir="${ca}_vs_${cb}"
  outdir="out/differential/${ct}/${pairdir}/${ablation}"
  if [ -f "${outdir}/metrics.json" ]; then
    echo "SKIP (already done): ${ct} ${pairdir} ${ablation}"
    return
  fi
  mkdir -p "${outdir}/ckpt"
  echo "=== RUN full/${ablation}: ${ct} ${ca} vs ${cb} ==="
  $PY -m src.train_differential --cell_type $ct --num_TF 500 --condition_a $ca --condition_b $cb \
    $HP --diff_ablation $ablation \
    --ckpt_dir "${outdir}/ckpt" --output_dir "${outdir}" \
    > "${outdir}/log.txt" 2>&1 || echo "FAILED: ${ct} ${pairdir} ${ablation}"
}

run_posthoc () {
  ct=$1; ca=$2; cb=$3
  pairdir="${ca}_vs_${cb}"
  outdir="out/differential/${ct}/${pairdir}/posthoc_baseline"
  if [ -f "${outdir}/metrics.json" ]; then
    echo "SKIP (already done): posthoc ${ct} ${pairdir}"
    return
  fi
  mkdir -p "${outdir}"
  echo "=== RUN posthoc_baseline: ${ct} ${ca} vs ${cb} ==="
  $PY -m src.posthoc_baseline --cell_type $ct --num_TF 500 --condition_a $ca --condition_b $cb \
    $HP --output_dir "${outdir}" \
    > "${outdir}/log.txt" 2>&1 || echo "FAILED: posthoc ${ct} ${pairdir}"
}

run_naive () {
  ct=$1; ca=$2; cb=$3
  pairdir="${ca}_vs_${cb}"
  outdir="out/differential/${ct}/${pairdir}/naive_expression_baseline"
  if [ -f "${outdir}/metrics.json" ]; then
    echo "SKIP (already done): naive ${ct} ${pairdir}"
    return
  fi
  mkdir -p "${outdir}"
  echo "=== RUN naive_expression_baseline: ${ct} ${ca} vs ${cb} ==="
  $PY -m src.naive_baseline --cell_type $ct --num_TF 500 --condition_a $ca --condition_b $cb \
    --output_dir "${outdir}" > "${outdir}/log.txt" 2>&1 || echo "FAILED: naive ${ct} ${pairdir}"
}

PAIRS="hESC:00h:96h hESC:00h:12h hESC:36h:72h mESC:0h:72h mESC:0h:12h mESC:24h:48h mHSC-E:LT_HSC:Prog mHSC-E:LT_HSC:HSPC mHSC-E:HSPC:Prog"
PRIMARY_PAIRS="hESC:00h:96h mESC:0h:72h mHSC-E:LT_HSC:Prog"

STAGE=${1:-all}

if [ "$STAGE" == "full" ] || [ "$STAGE" == "all" ]; then
  for p in $PAIRS; do
    IFS=':' read -r ct ca cb <<< "$p"
    run_full $ct $ca $cb full
  done
fi

if [ "$STAGE" == "posthoc" ] || [ "$STAGE" == "all" ]; then
  for p in $PAIRS; do
    IFS=':' read -r ct ca cb <<< "$p"
    run_posthoc $ct $ca $cb
  done
fi

if [ "$STAGE" == "naive" ] || [ "$STAGE" == "all" ]; then
  for p in $PAIRS; do
    IFS=':' read -r ct ca cb <<< "$p"
    run_naive $ct $ca $cb
  done
fi

if [ "$STAGE" == "ablations" ] || [ "$STAGE" == "all" ]; then
  for p in $PRIMARY_PAIRS; do
    IFS=':' read -r ct ca cb <<< "$p"
    for ab in no_scfm no_grn no_contrastive; do
      run_full $ct $ca $cb $ab
    done
  done
fi

echo "=== Phase 2 stage '${STAGE}' complete ==="
