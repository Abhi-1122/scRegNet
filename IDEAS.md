# Graph idea backlog

Ideas for changing the GRN graph or using graphs in new ways. Status: `testing` (in progress, see
PROGRESS.md under 7 Nov), `open` (not tried), `done` (result in PROGRESS.md, section noted).
Preferred: methods proven in another domain (recommender systems, knowledge graphs, molecules,
citation networks) that have not been applied to single-cell GRN inference.

## Tested (7 Nov), results in PROGRESS.md sections 25-31

| # | Idea | Origin | Result (within-TF vs hero) | Section |
|---|---|---|---|---|
| 1 | TF summary compared with the candidate gene | Item-based CF; DIN (Zhou et al. 2018) | hHEP +0.011, mHSC-E +0.011, new genes +0.018; uniform mean does as well as attention. **Keep (uniform version)** | 25 |
| 2 | Path-based scoring | NBFNet (Zhu et al., NeurIPS 2021) | −0.052 | 26 |
| 3 | Subgraph classification | SEAL (Zhang & Chen, NeurIPS 2018) | −0.042 | 27 |
| 4 | Correct & Smooth on a graph of pairs | C&S (Huang et al., ICLR 2021) | −0.014 | 28 |
| 6 | Hub caps and inverse-degree weights | Topping et al. 2022; recsys popularity bias | ±0.004, mESC cap −0.009 | 29 |
| 9 | Positional encodings | Dwivedi et al. 2020/2022; PEG | −0.002 to −0.019 | 30 |
| S | Degree-corrected block model, learned blocks (no neural network) | Karrer & Newman 2011 | Ties best GNN on hHEP (+0.009) and mESC (+0.005); new genes +0.029. **Keep for the per-cell pipeline** | 31 |

## Open (from the 7 Nov brainstorm)

Last column added 12 Nov from the literature scan below. "Not found" means no hit in the arXiv and web
searches run that day (not exhaustive).

| # | Idea | Origin | Why it might help here | Already done for GRNs? (12 Nov) |
|---|---|---|---|---|
| 5 | Known negatives as a second, signed edge type | Signed GCN (Derr et al., ICDM 2018, arXiv 1808.06354) | Training non-edges are labels the graph throws away | Not found. "Signed" in GRN papers means activation/inhibition (scSGL; MSGRNLink, arXiv 2512.11927), not negatives as edges |
| 7 | DropEdge-rate tuning | Rong et al., ICLR 2020 | Train graph is sparser than test graph; gives a tuning knob | Partly. GCLink (Bioinformatics 2025) drops 20% of edges as a contrastive view; BRIDGE (arXiv 2606.14734) criticises random edge perturbation. No rate sweep reported |
| 8 | Co-targeting graphs (genes sharing regulators, TFs sharing targets) | Bipartite projection / common neighbours | Neighbourhood by regulator set, not by noisy co-expression | Not found as a GNN relation. Closest: NetREX-CF (Commun. Biol. 2022, PMC9684490) uses CF on the TF-gene matrix. See L3, L4, L5 below for stronger variants |
| 10 | GENIE3 / GRNBoost2 network as a relation or decoder scalar | Huynh-Thu et al. 2010; Moerman et al. 2019 | Label-free, stronger than correlation (near chance for us) | Partly. GT-GRN (arXiv 2504.16961) encodes networks from unsupervised inference as graph input; DeepRIG (PLoS Genet. 2023) uses a co-expression prior graph |
| 11 | Pseudotime-lagged directed graph | LEAP, SCODE (BEELINE methods) | Direction and timing, closer to causal | Partly. Temporal GNN on pseudotime-binned SCENIC networks: Dileo and Sottoriva, arXiv 2605.28659 (preprint). Unsupervised: PseudoGRN (PMC12756061) |
| 12 | Smooth expression over a cell kNN graph before features | MAGIC (van Dijk et al. 2018) | Attacks dropout noise directly | Negative evidence: a BEELINE imputation study (bioRxiv 2021.04.13.439623, title not retrieved) reports imputation does not generally improve network reconstruction |
| 13 | Gene kNN graph from Geneformer embeddings | kNN graph construction | Geneformer is the strongest ingredient | Not found as a message graph. scRegNet / scTransNet use FM embeddings as node features only |
| 14 | Joint cell-gene graph | DeepMAPS, scGNN | Learns gene embeddings from cells directly (heavy) | Yes. GP-DHT (arXiv 2601.10995, gene-cell heterogeneous graph), BRIDGE (arXiv 2606.14734, gene and cell spaces), HyperG-VAE (Cell Rep. Methods 2025, cell hyperedges) |
| 15 | Knowledge graph: GO / Reactome gene-gene, JASPAR family TF-TF, RGCN | RGCN (Schlichtkrull et al. 2018) | External biology without ATAC | GO/Reactome + RGCN not found. Text-derived priors exist: GRNITE (bioRxiv 2025.11.25.690454), GRNPT (NCBI descriptions, unseen regulators), InfoSEM (arXiv 2503.04483) |
| 16 | Cross-cell-type transfer (hESC<->hHEP, mESC<->mHSC-E) as a prior relation | Transfer learning | Needs a ChIP-overlap leak check first | Yes, as pretrain-then-fine-tune: GCLink (mESC source, 5% of target edges for tuning), BRIDGE few-shot transfer. As a prior relation: not found |
| 17 | Self-supervised pretraining on co-expression + PPI graph | GraphCL, GRACE, Hu et al. 2020 | Aimed at unseen genes | Partly. Contrastive objectives in GCLink, GP-DHT, BRIDGE (all edge-drop or expression views, TF-target graph only). SupGCL (arXiv 2505.17786, ICML 2026) targets patient GRNs, not BEELINE. No PPI-graph pretraining found |

## Literature scan (cross-domain methods not yet used for GRNs)

Scan of 12 Nov 2026: arXiv search plus web search; every paper below was seen in a search result (arXiv id,
DOI or URL given). Gains are those stated in the abstracts; where an abstract gives no number the cell
says so. "Unused" = no GRN/scRNA-seq hit found, which is weaker than a proof of absence. Cost: S = under a
day in `src/pair_gnn.py`, M = a few days, L = new module.

### Candidates

| ID | Method | Home domain | Paper | Evidence there | GRN status | Why it fits here | Cost |
|---|---|---|---|---|---|---|---|
| L1 | DropoutNet: drop the graph/preference input per node during training | Recsys cold start | Volkovs, Yu, Poutanen, NeurIPS 2017 (papers.nips.cc/paper/7081) | Comparable to SOTA warm, "significantly" better cold (abstract, no figures retrieved) | Unused | Unseen TFs and genes have no neighbour signal; trains the decoder to rely on features alone | S |
| L2 | Popularity deconfounding and logit adjustment (PDA, MACR, logit adjustment) | Recsys; long-tail classification | Zhang et al., SIGIR 2021, arXiv 2105.06067; Wei et al., KDD 2021, arXiv 2010.15363; Menon et al., ICLR 2021, arXiv 2007.07314 | PDA: 3 datasets (latent factor model); MACR: 5 datasets (MF, LightGCN); abstracts give no figures | Partly. The bias is diagnosed (Stock et al., bioRxiv 2025.12.19.695616: degree heuristics match GNNs under standard negative sampling; InfoSEM, arXiv 2503.04483) and degree-aware sampling proposed; no model-side debiasing found | Popularity alone gives 0.724 on unseen TFs; PDA says keep it at inference with a tuned weight, train on the residual | S |
| L3 | Degree-normalised length-3 paths (L3, NormalizedL3) | PPI | Kovács et al., Nat. Commun. 10:1240, 2019; Yuen and Jansson, BMC Bioinf. 24:59, 2023 | Authors report L3 beats all compared link predictors on PPI (no figure retrieved) | Unused. Only a patent (US11657311B2) lists GRNs | In a bipartite TF-gene graph common neighbours do not exist; odd paths do. Normalisation damps hubs | S |
| L4 | ProbS / HeatS hybrid (network-based inference) | Recsys; drug-target | Zhou et al., PNAS 107:4511, 2010, arXiv 0808.2670; Cheng et al., PLoS Comput. Biol. 8:e1002503, 2012 | Hybrid improves accuracy and diversity together; NBI best on 4 DTI sets | Unused | One parameter (lambda) moves between popularity-driven and niche scores; parameter-free graph signal | S |
| L5 | EASE: closed-form item-item linear autoencoder | Recsys | Steck, WWW 2019, arXiv 1905.03375 | Beats deep non-linear CF on most datasets | Partly. CF on the TF-gene matrix inside NetREX-CF (PMC9684490); closed-form item-item not found | Gives a learned gene-gene "co-targeting" matrix usable as a score or as a second relation | S |
| L6 | SimGCL: noise-perturbed contrastive view | Recsys | Yu et al., SIGIR 2022, arXiv 2112.08679 | Shows contrastive gain comes from uniformity that mitigates popularity bias; beats augmentation-based CL on 3 datasets | Partly. Edge-drop contrastive in GCLink, GP-DHT, BRIDGE; noise-based view not found | No edge perturbation (BRIDGE and SupGCL argue random edge drops distort GRNs); aims at popularity | S |
| L7 | Tail-GNN: neighbourhood translation from head to tail nodes | Graph learning, long tail | Liu, Nguyen, Fang, KDD 2021 | Beats SOTA on 5 datasets (abstract) | Unused | Low-degree and unseen genes are where messages are thin | M |
| L8 | Graph structure learning (IDGL, SLAPS) | Graph learning | Chen et al., NeurIPS 2020, arXiv 2006.13009; Fatemi et al., NeurIPS 2021, arXiv 2102.05034 | Better than other structure-learning models (abstracts) | Not found for supervised TF-target link prediction | Replaces hand-picked co-expression/STRING graphs with a learned one | M |
| L9 | Graph diffusion convolution (PPR / heat kernel, sparsified) | Citation networks | Gasteiger et al., NeurIPS 2019, arXiv 1911.05485 | Gains "across a wide range of models and datasets" | Unused | Rewires noisy edges into multi-hop weights without extra layers; overlaps item 6 | S |
| L10 | Hyperbolic GCN | Hierarchical / scale-free graphs | Chami et al., NeurIPS 2019, arXiv 1910.12933 | Up to 63.1% ROC-AUC error reduction for link prediction | Not found | Master regulators vs leaf targets look hierarchical; weak evidence it matters at 1 layer | M |
| L11 | Hypergraph NN / hypergraph contrastive CF | Recsys; vision | Feng et al., AAAI 2019, arXiv 1809.09401; Xia et al., SIGIR 2022, arXiv 2204.12200 | HCCF beats SOTA on 3 datasets | Partly. HyperG-VAE (cells as hyperedges), MHHGRN, Motif-GRN, scHyperLink; regulon-as-hyperedge not found | Regulon = hyperedge; but mean-pooling a hyperedge equals the SAGE aggregation we have | M |
| L12 | Signed GCN with negatives as edges | Social networks | Derr et al., ICDM 2018, arXiv 1808.06354 | Not in the abstract | Partly (different meaning of "signed"; see item 5) | Uses training non-edges as a graph relation | M |
| L13 | Temporal graph networks | Dynamic graphs | Rossi et al., arXiv 2006.10637 | Not in the abstract | Partly. Dileo and Sottoriva, arXiv 2605.28659 (preprint, two mouse datasets) | Pseudotime-binned graphs; weak fit with static BEELINE labels | L |
| L14 | ULTRA: zero-shot inductive KG link prediction | Knowledge graphs | Galkin et al., ICLR 2024, arXiv 2310.04562 | Zero-shot on 57 KGs on par or better than trained baselines | Unused | Transfer across cell types with different gene sets; built on NBFNet, so overlaps item 2 | L |
| L15 | DrugBAN: bilinear attention + conditional domain adversarial learning | Drug-target | Bai et al., arXiv 2208.02194 (venue unverified) | Best overall vs 5 baselines on 3 benchmarks, in-domain and cross-domain | Unused for GRN link prediction (domain adaptation exists for TF binding: Cochran et al., Genome Res. 2022) | Cross-cell-type and unseen-TF shift handled explicitly | M |
| L16 | Text-embedding gene features (GenePT) | LLM + genes | Chen and Zou, bioRxiv 2023.10.16.562533 | Comparable or better than expression-pretrained models on gene-property tasks | Partly. GRNPT (unseen regulators), InfoSEM, GRNITE use NCBI/LLM text | A second, non-expression view to fuse with Geneformer; late fusion of diverse models is the robust gain (week 3) | S-M |
| L17 | Neo-GNN: learned neighbourhood-overlap features | Link prediction | Yun et al., arXiv 2206.04216 | Beats overlap heuristics on standard benchmarks (not GRN data) | Unused on bipartite TF-gene graph | Vehicle for feeding L3, L4, L5 scores into the decoder | M |

### Existing GNN methods on BEELINE (no longer novel)

Metrics are as reported by the authors or by a citing paper, under their own splits (random pair splits,
standard negative sampling); they are not comparable with the within-TF AUROC used here. Stock et al.
(bioRxiv 2025.12.19.695616) argue such splits reward node degree.

| Method | Year, venue | Core graph idea | Reported metric |
|---|---|---|---|
| GENELink (Chen, Liu) | 2022, Bioinformatics 38:4522 | GAT on known TF-gene graph with expression features | Comparable or better than 8 methods on 7 datasets; AUROC 0.9057 mESC-500, 0.8618 hESC-500 (as quoted in GT-GRN) |
| GNNLink (Mao et al.) | 2023, Brief. Bioinform. 24(6) | GCN encoder on known GRN, matrix-completion decoder | AUROC 0.8768 mESC-500, 0.8251 hESC-500 (as quoted in GT-GRN) |
| DeepRIG (Wang, Chen, Zou) | 2023, PLoS Genet. 19(9) | Graph autoencoder on a co-expression prior graph | Best on BEELINE in-silico and real sets (abstract) |
| GMFGRN (Li et al.) | 2024, Brief. Bioinform. (DOI bbad529, unverified) | GNN matrix factorisation of gene and cell embeddings | Unverified |
| scMGATGRN (Yuan et al.) | 2024, Brief. Bioinform. bbae526 | Multiview graph attention | Compared with 10 methods on 7 datasets |
| IGEGRNS (Gan et al.) | 2024, Bioinformatics 40(5) btae291 | GraphSAGE embeddings + top-k pooling + CNN on pairs (close to the present model) | TFs+1000 AUROC: hESC 0.808, hHEP 0.863, mESC 0.841 |
| scTransNet (Kommu et al.) | 2024, arXiv 2407.18181 | scBERT embeddings + GNN over known GRN | hESC AUROC 0.880, AUPRC 0.595 (from a secondary snippet) |
| scRegNet (Kommu et al.) | 2025, Bioinformatics | scFM embeddings (Geneformer, scGPT, scBERT) + GNN | SOTA vs 9 baselines on 7 datasets (project baseline) |
| GCLink (Yu et al.) | 2025, Bioinformatics btaf074 | 2-layer GAT, edge-drop second view, contrastive loss, source-to-target fine-tuning | Best in most cases vs GENELink, CNNC, GNE; values only in figures |
| GRLGRN (Wang et al.) | 2025, BMC Bioinf. 26:108 | Graph transformer extracting implicit links from the prior GRN | Best AUROC on 78.6%, AUPRC on 80.9% of datasets |
| GAEDGRN (Wei et al.) | 2025, Brief. Bioinform. 26(3) bbaf232 | Gravity-inspired directed graph autoencoder, PageRank*-weighted features, random-walk regulariser | Best AUROC on 38 of 42 datasets |
| GT-GRN (Teji, Roy) | 2025, arXiv 2504.16961 | BERT on random walks of inferred networks + graph transformer | AUROC 0.9402 mESC-500, 0.8793 hESC-500 |
| GRNFormer (Hegde, Cheng) | 2025, bioRxiv 2025.01.26.634966 | Transformer encoder + variational graph autoencoder with pairwise attention | Not retrieved |
| MSGRNLink (Xi et al.) | 2025, arXiv 2512.11927 | Signed directed graph, magnetic signed Laplacian convolution | Best AUROC vs baselines (abstract) |
| GP-DHT (Yan et al.) | 2026, arXiv 2601.10995 | Gene-cell heterogeneous graph, multi-relational attention, supervised contrastive | +5 to 7% AUROC/AUPRC on 7 BEELINE sets |
| BRIDGE (Dong et al.) | 2026, arXiv 2606.14734 | Gene and cell contrastive learning on co-expression-refined vs original graph, gated encoding | +5% AUPRC over GCLink on Specific networks; better few-shot transfer |

Earlier but relevant: GRGNN (Wang et al. 2020, subgraph classification over TF-gene pairs, SEAL-like) and
SIRENE (Mordelet, Vert 2008, arXiv 0802.3959, supervised per-TF inference). GTAT-GRN (2025, PMC12540167) and Motif-GRN (TNNLS 2026) were seen only in search listings; details
unverified.

### Ranking

Score = fit to findings x novelty in GRN x cost to try (1 = best). Top six:

| Rank | ID | Reason |
|---|---|---|
| 1 | L1 DropoutNet | Targets the unseen-TF/gene gap directly, unused, one dropout mask |
| 2 | L2 popularity split / logit adjustment | Attacks the dominant signal (0.724) instead of ignoring it; diagnosed but not fixed in GRN literature |
| 3 | L3 normalised L3 paths | Unused, no training, answers item 8 with a stronger tool |
| 4 | L5 EASE | Closed form, gives a learned gene-gene relation; partly used only as CF framing |
| 5 | L4 ProbS/HeatS hybrid | One parameter, unused, popularity dial |
| 6 | L6 SimGCL-style noise contrast | Popularity-oriented, avoids edge perturbation that GRN papers criticise |

Next: L16 (text embeddings, cheap and fits late fusion but partly used), L9 (GDC), L7 (Tail-GNN).

### Cheapest first test for the top six

1. **L1.** In training, for each node in the batch draw a Bernoulli(p) mask and replace its aggregated neighbour message by zeros (self features kept), p in {0.25, 0.5}. The existing removal of batch edges is edge-level; this is node-level. Evaluate on the held-out-TF split against 0.724.
2. **L2.** (a) Post hoc: score = GNN logit + lambda * z(log(1 + train in-degree of gene)), lambda picked on validation. (b) Training: add the same term as a fixed offset to the logit, train on offset + GNN, drop the offset at test time (logit adjustment) or keep it at a tuned weight (PDA). Report within-TF AUROC for seen and unseen TFs.
3. **L3.** With train TF x gene matrix A, S = A Dg^(-1/2) A^T Dt^(-1/2) A (paths t-g1-t1-g, divided by sqrt of the two intermediate degrees; Kovács et al. normalisation, check against the paper). Exclude the scored pair's own edge when scoring training pairs. Report within-TF AUROC of S alone on seen TFs, then rank-average with the GNN score (late fusion). Zero for TFs with no training edges.
4. **L5.** B = I - P diag(1/diag P), P = (A^T A + lambda I)^(-1), diagonal set to 0, lambda in {10, 100, 1000}. Score = A B. Test alone, then fused, then top-k positive entries of B as a second gene-gene relation in the SAGE layer (items 5 and 8 overlap).
5. **L4.** W[a,b] = (1 / (k_a^(1-lambda) k_b^lambda)) * sum over TFs of A[t,a] A[t,b] / k_t (formula from memory of Zhou et al.; verify before coding), lambda in {0, 0.25, 0.5, 0.75, 1}. Score = A W. Same protocol as L3; share the code.
6. **L6.** Second forward pass with uniform noise (normalised random vector scaled by eps in {0.05, 0.1, 0.2}) added to the SAGE output; InfoNCE (temperature 0.2) between the two views over genes in the batch, weight in {0.05, 0.2}, added to the pair loss. Check seen/unseen split separately.
