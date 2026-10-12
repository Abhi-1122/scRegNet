# Link prediction ideas for TF-to-target inference on hESC: a literature review

Written 2026-10-12. Scope: which ideas from general link prediction, recommender systems and biomedical pair-input prediction we should apply to our hESC setting (34 labelled TFs, 910 genes, about 818 labelled candidates per TF, 3,030 training positives, 1,061 test positives, 14.9% positive rate).

How this was produced: papers were checked with the arXiv tools (dates, abstracts, comments fields) and with web search or page fetches (venues, volumes, pages). I did not run any code, touch the GPU, or read the repository. The five findings are taken from the task description as given. Anything I could not confirm is marked "unverified". A list of all unverified items is at the end.

How the "likelihood of improving within-TF AUROC" lines should be read: they are my own judgement from the structure of the problem, not numbers from any paper. No paper in this review was run on our data.

---

## 0. The short version

- Our problem is matrix completion on a 34 x 910 binary matrix. Recommender systems have worked on exactly this shape of problem for twenty years, and the standard answer is biased matrix factorisation: score = bias_TF + bias_gene + dot(u_TF, v_gene). The two bias terms soak up popularity. The dot product supplies the TF-gene interaction that scRegNet's concatenation plus linear layer cannot represent.
- Preferential attachment (score = TF degree times gene degree) is the oldest link prediction heuristic. Our "two-count model" is that heuristic. A 2024 paper (Aiyappa et al.) shows that on 90 standard graphs this degree-only heuristic reaches a mean AUROC of 0.83, and drops to 0.54 once the benchmark is corrected for degree. Our 0.872 vs 0.887 gap is therefore a known property of random-edge link prediction benchmarks, not a quirk of our data.
- The message-passing vs supervision edge issue (finding 4) is a documented pitfall (Zhu et al., WSDM 2024) and has standard fixes (remove the target edge from the graph while scoring it). It matters even more if we switch to an interacting decoder, because an interacting decoder can exploit the leak and the current linear decoder cannot (section 9, point 2).
- Pair-aware GNNs (SEAL, NBFNet, NCN, BUDDY) all inject information about the specific pair, such as paths between the TF and the gene. On our small graph all of them are cheap. Because our graph is bipartite (TFs on one side, genes on the other) and gene degrees are tiny (3,030 edges over 910 genes is about 3.3 per gene), the information they use boils down to "do TFs that share targets with this TF also regulate this gene?". Finding 5 says that signal is real but modest, so I would not expect large gains from the heaviest models.
- The published GRN link-prediction papers (GENELink, GNNLink, GCLink, scTransNet, scRegNet) all split over known gene pairs. None reports a degree-only baseline and none reports a held-out-TF or held-out-gene test in the main text that I could read. The dot-product ones (GENELink by default, GNNLink, GCLink) can express interactions, so the additive ceiling is specific to the scRegNet family, but the popularity problem applies to all of them.

---

## 1. Plain-language vocabulary used below

- **Link prediction**: given a graph with some edges hidden, rank the hidden edges above non-edges.
- **Bipartite graph**: nodes come in two groups and edges only go between groups. Our TF-to-gene labels form a bipartite graph (TFs are also genes, so there are a few TF-to-TF edges, but almost all edges are TF to non-TF gene).
- **Message-passing edges**: edges the GNN is allowed to read while computing node embeddings.
- **Supervision edges**: edges used as training labels in the loss. Finding 4 is that in scRegNet these two sets are identical during training and different at test time.
- **Decoder**: the last piece of the model that turns the two node embeddings of a candidate pair into a score.
- **Degree**: number of edges at a node. Popularity bias means a model scores well just by favouring high-degree nodes.
- **Within-TF AUROC**: AUROC computed inside each TF's candidate list and averaged over TFs. It is blind to how popular a TF is and only measures whether the model ranks a TF's own targets above its non-targets.

---

## 2. Classic heuristics

All items in this section are scoring rules that use only the observed graph. They need no training. They matter to us for two reasons: they are strong baselines (HeaRT, section 6, reports that Katz and resource allocation rank second and third on ogbl-collab, ahead of most GNNs), and they are cheap features to feed into a learned model.

### 2.1 The formulas, written for a TF x gene matrix

Let A be the 34 x 910 training matrix (A[t,g] = 1 if g is a known target of t). Let d_t be a TF's number of known targets and d_g a gene's number of known regulators.

| Heuristic | General graph definition | Bipartite analogue for a pair (t, g) | Within-TF ranking depends on |
|---|---|---|---|
| Preferential attachment | deg(u) x deg(v) | d_t x d_g | d_g only (d_t is a constant inside one TF) |
| Common neighbours | number of shared neighbours | There are no shared neighbours in a bipartite graph, so use paths of length 3: t - g' - t' - g. Count = (A A^T A)[t,g] | gene popularity and TF-TF overlap |
| Adamic-Adar | sum over shared neighbours of 1 / log(degree) | same 3-paths, each weighted by 1/log(d_g') or 1/log(d_t') of its middle nodes | mostly TF-TF overlap |
| Resource allocation | sum over shared neighbours of 1 / degree | same 3-paths, weighted by 1/d of the middle nodes | mostly TF-TF overlap |
| Katz | weighted count of all paths, shorter paths weigh more | beta x (I - beta^2 A A^T)^(-1) A, odd path lengths only, needs beta below 1 / (largest singular value of A) | mix |

Two readings of the 3-path count are worth stating in recommender language, because both are old ideas there:

- **TF-based (user-based) collaborative filtering**: score(t,g) = sum over other TFs t' of similarity(t,t') x A[t',g], where similarity is how many targets t and t' share. This is exactly finding 5 (interacting TFs share more targets) turned into a predictor.
- **Gene-based (item-based) collaborative filtering**: score(t,g) = average over t's known targets g' of similarity(g,g'), where similarity is how many regulators g and g' share. Sarwar et al. (section 4.2) report this variant is usually better than the TF-based one.

A practical warning that is a mini version of finding 4. For a training-positive pair with A[t,g] = 1, the count (A A^T A)[t,g] includes degenerate walks that reuse the pair's own edge. Walking t to g, then back to some t' that regulates g, then to g again contributes d_g. Walking t to some g', back to t, then to g contributes d_t. So training positives get an automatic boost of roughly d_g + d_t - 1 that test pairs never get. Any heuristic used as a feature must be computed with the pair's own edge removed (leave-one-edge-out, or K-fold masking of training positives). This is the same disease as finding 4, and it applies equally to the PANDA-style co-expression feature of finding 3: when computing "mean expression profile of the TF's known targets" for a training positive (t, g), gene g must be excluded from that mean.

### 2.2 Citations

1. **Liben-Nowell, D. and Kleinberg, J. "The link-prediction problem for social networks." Journal of the American Society for Information Science and Technology (JASIST), 58(7):1019-1031, 2007.** Verified (volume, issue, pages from a bibliographic index; an earlier short version was at CIKM 2003).
   - Idea: Defines link prediction as scoring unconnected pairs by network proximity, and compares common neighbours, Adamic-Adar, Katz, preferential attachment and others. Simple proximity scores beat random guessing by a wide margin on collaboration networks.
   - Connection: finding 1. Preferential attachment is on their list; our two-count model is that rule split into TF and gene factors.
   - Cost: none, a few matrix products.
   - Likelihood: as a model, low for within-TF AUROC (the popularity ones are blind to the TF). As a source of features, see 2.1.

2. **Adamic, L. A. and Adar, E. "Friends and neighbors on the Web." Social Networks, 25(3):211-230, 2003.** Venue and pages taken from a secondary reading-list citation, so treat the page numbers as semi-verified.
   - Idea: Rare shared neighbours say more than common ones, so weight each shared neighbour by 1/log(its degree).
   - Connection: findings 1 and 5. Down-weighting hubs is a direct countermeasure to popularity.
   - Cost: trivial.
   - Likelihood: low to moderate as a stand-alone score, moderate as a feature.

3. **Zhou, T., Lü, L., and Zhang, Y.-C. "Predicting missing links via local information." European Physical Journal B, 71(4):623-630, 2009.** Verified (journal page, arXiv 0901.0553).
   - Idea: Introduces the resource allocation index (each shared neighbour passes on a unit of resource split among its neighbours, so it is weighted 1/degree) and compares nine local scores on six real networks. Common neighbours was best overall among the older scores and resource allocation beat it.
   - Connection: finding 5, as above. Resource allocation is the heuristic that HeaRT later uses to build hard negatives.
   - Cost: trivial.
   - Likelihood: moderate as a feature (see 2.1).

4. **Katz, L. "A new status index derived from sociometric analysis." Psychometrika, 18(1):39-43, 1953.** Verified. (Careful: a different Katz paper from the same year and volume exists, Katz and Powell on pages 249-256. The Katz index paper is pages 39-43.)
   - Idea: Count all paths between two nodes and discount long ones geometrically, so many short paths mean a strong tie.
   - Connection: findings 3 and 5. Katz on the TF x gene matrix is a smooth blend of "shares targets" (3-paths) and longer chains (5-paths and up). With only 34 TFs the inverse is 34 x 34, which is instant.
   - Cost: trivial. One tuning knob (beta); choose it on a validation split.
   - Likelihood: moderate.

5. **Barabási, A.-L. and Albert, R. "Emergence of scaling in random networks." Science, 286(5439):509-512, 1999.** Verified.
   - Idea: Real networks grow by preferential attachment: new edges prefer already popular nodes, which produces a few hubs and many low-degree nodes.
   - Connection: finding 1. This is the origin of the rule that our two-count model implements. In GRNs it also reflects biology: some TFs (and some promiscuously bound genes) really do have many edges.
   - Cost: none.
   - Likelihood: this is the baseline to beat, not an improvement. It gives the null model that every new method should be compared against.

6. **Lü, L. and Zhou, T. "Link prediction in complex networks: A survey." Physica A, 390(6):1150-1170, 2011.** Verified (arXiv 1010.0725 preprint plus Physica A citation).
   - Idea: Survey that organises similarity-based, random-walk-based and maximum-likelihood link predictors and lists AUROC-style evaluation conventions.
   - Connection: all findings; useful as a single reference for reviewers.
   - Cost: none.
   - Likelihood: not a method.

---

## 3. Autoencoder decoders and why concatenation plus a linear layer cannot interact

### 3.1 Why scRegNet's decoder is additive (finding 2, in one paragraph)

If the score is w^T [z_TF ; z_gene], with w = [w1 ; w2], then the score equals w1^T z_TF + w2^T z_gene. The first term depends only on the TF and the second only on the gene. Inside one TF the first term is a constant, so the gene order is set entirely by w2^T z_gene, which is the same for every TF. That is the "same gene order for every TF" result with additive-fit R^2 = 1.0. No amount of better embeddings, better GNNs or better foundation-model features can fix this, because the limit is in the last layer. The published scRegNet description says each gene's joint embedding passes through an MLP, the two outputs are concatenated, then a fully connected layer and a softmax (Kommu et al.; see the table in section 8). If there is a nonlinearity after the concatenation, the model could in principle interact. Our own check (R^2 = 1.0) shows that in practice it does not, so I take finding 2 as established.

What kinds of decoder can express interactions:

| Decoder | Formula for pair (t, g) | Expresses a TF-gene interaction? | Notes |
|---|---|---|---|
| Concatenate + linear | w1.u_t + w2.v_g | No | Our current model |
| Concatenate + MLP with hidden nonlinearity | f([u_t ; v_g]) | In theory yes | In practice hard to learn and worse than a dot product (Rendle et al.) |
| Dot product (GAE) | u_t . v_g | Yes, rank d | Needs the embedding to also encode popularity |
| Biased dot product (matrix factorisation) | b_t + b_g + u_t . v_g | Yes, rank d | Popularity goes into the biases, the dot term models what is specific to the pair |
| DistMult | sum_k r_k u_tk v_gk | Yes | Dot product with learned per-dimension weights; with a single relation it adds almost nothing over the dot product |
| Bilinear / RESCAL | u_t^T W v_g | Yes | Full d x d matrix W; more parameters, more risk with only 34 TFs |
| Hadamard + MLP (NCN-style) | MLP(u_t * v_g) | Yes | Richer than the dot product, still symmetric in the two sides unless you add side-specific parts |

### 3.2 Citations

1. **Kipf, T. N. and Welling, M. "Variational Graph Auto-Encoders." NIPS 2016 Workshop on Bayesian Deep Learning; arXiv 1611.07308 (2016-11-21).** Verified (arXiv comment and repository listing; this is a workshop paper, not a main-conference paper).
   - Idea: Encode every node with a graph convolutional network, then score a pair with the inner product of the two node embeddings, followed by a sigmoid (GAE), with a variational version (VGAE). Trained to reconstruct the observed adjacency matrix.
   - Connection: finding 2 (the dot-product decoder is the simplest fix) and finding 4 (the original GAE reconstructs the same adjacency it reads as input, i.e. message-passing edges equal supervision edges, the same issue we have).
   - Cost: very low. Replace the final linear layer by a dot product of the TF and gene vectors.
   - Likelihood of helping within-TF AUROC: moderate. It gives the model access to interactions, but without bias terms the embeddings must also carry popularity, which the biased version avoids. And it is only as good as the TF embedding (see 9.2).

2. **Yang, B., Yih, W., He, X., Gao, J. and Deng, L. "Embedding Entities and Relations for Learning and Inference in Knowledge Bases." ICLR 2015 (arXiv 1412.6575, 2014-12-20).** Verified (proceedings listings, arXiv).
   - Idea: Scores a triple (head, relation, tail) with a bilinear form restricted to a diagonal matrix, which is now called DistMult. The paper reports this simple bilinear model beat TransE and other fancier scorers on link prediction in knowledge bases (top-10 accuracy 73.2% vs 54.7% on Freebase according to the abstract summary I could read).
   - Connection: finding 2. Our TF-gene pair is a single-relation triple, so DistMult is just a weighted dot product. Also a model for evaluation: they rank each true triple among candidates with known true triples filtered out, which is close to our within-TF ranking.
   - Cost: very low.
   - Likelihood: same as the dot product. Not a separate bet.

3. **Nickel, M., Tresp, V. and Kriegel, H.-P. "A Three-Way Model for Collective Learning on Multi-Relational Data." ICML 2011, pages 809-816.** Verified (proceedings listing).
   - Idea: RESCAL. Factorise a three-way tensor (entity x entity x relation) so that each relation has a full d x d interaction matrix W and a pair is scored as u^T W v.
   - Connection: finding 2. This is the "bilinear" row above.
   - Cost: low, but with only 34 TFs a full W is likely to overfit.
   - Likelihood: low to moderate; try only after the plain biased dot product works.

4. **Rendle, S., Krichene, W., Zhang, L. and Anderson, J. "Neural Collaborative Filtering vs. Matrix Factorization Revisited." RecSys 2020 (arXiv 2005.09683, 2020-05-19).** Verified (abstract read; venue from search results, one listing omitted the first author).
   - Idea: With proper hyperparameter tuning, a plain dot product beats an MLP that is fed the two embeddings. An MLP can in theory represent a dot product but the paper shows it is non-trivial to learn one.
   - Connection: finding 2. The tempting patch is to put an MLP on the concatenation. This paper says do not: use the dot product, and treat the MLP as a harder-to-train alternative.
   - Cost: none (it tells us which option to try first).
   - Likelihood: high that a tuned dot product is at least as good as an MLP head here. It also lowers the risk that a reviewer will say we did not try an MLP; we can cite this.

---

## 4. The recommender-system view

Our setting is a ratings matrix. TFs play the role of users, genes play the role of items, a known regulation is a "liked" item. The labelled non-targets are like explicit "disliked" entries (with a caution: they are probably "unobserved" rather than truly negative, since the ground truth comes from experiments such as ChIP-seq; see section 6).

1. **Koren, Y., Bell, R. and Volinsky, C. "Matrix Factorization Techniques for Recommender Systems." IEEE Computer, 42(8):30-37, August 2009.** Verified (IEEE Xplore listing).
   - Idea: Model each rating as a baseline for the user plus a baseline for the item plus the dot product of a user factor vector and an item factor vector. The baselines absorb "this user rates high" and "this item is liked by everyone", so the factors only have to explain the leftovers.
   - Connection: findings 1 and 2. The two baselines are exactly our popularity model (TF positive rate, gene degree), and the dot product is the interaction scRegNet lacks. I only read the abstract and index listing this session, so the exact form of the baseline terms is from my knowledge of the paper and counts as unverified in detail.
   - Cost: very low. Add two scalar biases per TF and per gene and a dot product of low-dimensional vectors (rank 2 to 8 is enough with 34 TFs). Training is seconds to minutes. Gene vectors can be produced from expression plus the GNN (so the model is inductive for genes), or be free parameters (so it is purely transductive).
   - Likelihood of helping within-TF AUROC: moderate to high, conditional on the leak fix (section 9). It is the most direct fix for finding 2, and it is the decoder recommender systems have found to be a strong default for two decades.

2. **Koren, Y. "Factorization meets the neighborhood: a multifaceted collaborative filtering model." KDD 2008, pages 426-434, doi 10.1145/1401890.1401944.** Verified (ACM listing via a secondary index).
   - Idea: Combine latent-factor models (global structure) with neighbourhood models (very local structure) in one score, because the two capture different things.
   - Connection: findings 3 and 5. The latent factor part is the dot product; the neighbourhood part is the 3-path collaborative filtering of section 2. Together they are the recsys version of "use the GNN plus the heuristics".
   - Cost: low. A neighbourhood term can be added as one extra feature.
   - Likelihood: moderate.

3. **Sarwar, B., Karypis, G., Konstan, J. and Riedl, J. "Item-based collaborative filtering recommendation algorithms." WWW 2001 (WWW10, Hong Kong), pages 285-295, doi 10.1145/371920.372071.** Verified (conference listing).
   - Idea: Predict a user's interest in an item from the items the user already likes that are similar to it. Item-item similarity is computed from who likes both. Reported better than user-user neighbourhoods.
   - Connection: finding 3. "Gene g is similar to the TF's known targets" is item-based collaborative filtering. Finding 3's co-expression z-score is the same idea with similarity computed from expression instead of from shared regulators. The two similarities can be combined.
   - Cost: trivial (see section 2.1).
   - Likelihood: moderate. Strong chance that it improves over 0.72 within-TF by itself; whether it adds to PANDA-like co-expression is unclear, since the two partly overlap.

4. **van den Berg, R., Kipf, T. N. and Welling, M. "Graph Convolutional Matrix Completion." arXiv 1706.02263 (2017-06-07).** arXiv date verified; the venue (I recall a KDD 2018 workshop) is unverified.
   - Idea: Treat rating prediction as link prediction on a user-item bipartite graph. A graph-convolution encoder produces user and item embeddings and a bilinear decoder predicts the rating class.
   - Connection: findings 2 and 4. It is the closest published template to our setup (bipartite graph, GCN encoder, bilinear decoder), though the original does not address leakage either.
   - Cost: low.
   - Likelihood: same as the dot-product decoder.

5. **He, X., Deng, K., Wang, X., Li, Y., Zhang, Y. and Wang, M. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." SIGIR 2020, pages 639-648 (arXiv 2002.02126).** Verified.
   - Idea: For collaborative filtering, drop the feature transformation and nonlinearity from GCN and keep only neighbourhood averaging on the user-item graph. The final embedding is a weighted sum over layers and the score is a dot product.
   - Connection: finding 3. Mean aggregation over the training graph followed by a dot product is literally what finding 3 says a GNN could do. LightGCN is a published, tuned version of it. It uses no node features, which is a difference: we have expression and Geneformer embeddings.
   - Cost: very low (no weights beyond embeddings). If we use free embeddings it is purely transductive.
   - Likelihood: moderate. A good check of how much of the performance is graph structure alone. The same leak caveat applies: LightGCN trains on edges that are also in its input graph.

6. **Ferrari Dacrema, M., Cremonesi, P. and Jannach, D. "Are we really making much progress? A worrying analysis of recent neural recommendation approaches." RecSys 2019 (arXiv 1907.06902).** Verified (venue from search results; award name varies by source, so not cited here).
   - Idea: Of 18 neural recommendation methods published between 2015 and 2018, only 7 could be reproduced, and per the secondary source I read, most of those did not beat well-tuned simple baselines. Simple nearest-neighbour methods were often competitive.
   - Connection: finding 1 and a reviewer-readiness point. The lesson is to report tuned simple baselines next to every complex model.
   - Cost: none.
   - Likelihood: not a method. It is a reason to put item-based collaborative filtering and the biased dot product in the main results table.

---

## 5. Pair-aware GNNs

Common idea: a node-level GNN embeds the TF and the gene separately, so the pair's score is a function of two independent summaries. Pair-aware methods let the computation depend on both nodes at once: by looking at the neighbourhood around the pair (SEAL, IGMC), the paths between them (NBFNet), or their shared neighbours (NCN, Neo-GNN, BUDDY). Our graph has 910 nodes, about 3,000 training edges, and about 27,800 labelled candidate pairs (34 x 818), so everything below is affordable on an 8 GB GPU. The limit is not compute but information: all of these use only the training adjacency (plus node features), and our adjacency is sparse and bipartite.

1. **Zhang, M. and Chen, Y. "Link Prediction Based on Graph Neural Networks." NeurIPS 2018 (arXiv 1802.09691, 2018-02-27).** Verified (proceedings page).
   - Idea (SEAL): For each candidate pair, cut out the h-hop enclosing subgraph around the two nodes, label every node by its distances to the pair (double-radius node labelling), and run a GNN on that subgraph to classify "link or no link". They prove that many popular heuristics (Katz, PageRank, SimRank) can be approximated from a small h-hop subgraph.
   - What extra information about the pair it uses: the local wiring around both nodes, expressed relative to the pair.
   - Connection: finding 2 (the score depends on both nodes jointly), finding 5 (shared structure). Whether SEAL removes the target link from the subgraph is not stated in the text I could search; I believe common implementations do, but that is unverified.
   - Cost on our graph: with h = 1, the subgraph of a pair (t, g) is the TF, the gene, t's roughly 90 known targets, g's few known regulators, and the edges among them. That is about 100 nodes. For 27,800 pairs that is under 3 million node rows per epoch, so minutes per epoch even with a naive implementation. With h = 2 the subgraph of any TF covers most of the graph, so stay at h = 1.
   - Likelihood: moderate. On a bipartite graph the 1-hop subgraph contains exactly the t - g' - t' - g paths of section 2, so SEAL is a learned replacement for hand-built 3-path counts. I expect it to find the same signal as the heuristics and cost more engineering.

2. **Zhang, M. and Chen, Y. "Inductive Matrix Completion Based on Graph Neural Networks" (IGMC). ICLR 2020 (arXiv 1904.12058).** Verified (conference page).
   - Idea: SEAL for ratings. Take the 1-hop subgraph around a (user, item) pair in the rating matrix, label user and item, run a GNN, and read the rating from the two target nodes. It needs no side information and generalises to unseen users and items.
   - What extra information it uses: the neighbouring users of the item and the neighbouring items of the user.
   - Connection: it is the exact recommender analogue of "TF-gene pair from the bipartite training matrix". Also the best published precedent for an inductive (held-out-TF or held-out-gene) evaluation of this kind of model.
   - Cost: low, as for SEAL with h = 1.
   - Likelihood: moderate, with the same caveat as SEAL.

3. **Zhang, M., Li, P., Xia, Y., Wang, K. and Jin, L. "Labeling Trick: A Theory of Using Graph Neural Networks for Multi-Node Representation Learning." NeurIPS 2021 (arXiv 2010.16103, first posted 2020-10-30).** Verified (arXiv comment and proceedings page).
   - Idea: A GNN gives one vector per node, and simply combining two node vectors loses the dependence between the nodes. If you first mark the target nodes in the graph (give them a special label) and then run the GNN, the node vectors become relative to the pair, and in principle you can represent anything about the pair.
   - Connection: finding 2. Scoring from two separately computed node embeddings is exactly the "aggregate independent single-node embeddings" setup that this theory shows to be fundamentally limited. Note the scope difference: their limit is about structural expressiveness of the GNN, ours is stronger because our last layer is linear. A pair-aware GNN with a linear decoder would still be additive per pair-specific vector, but the vector would now depend on the pair.
   - Cost: a cheap variant for us is to mark only the TF. Because we have just 34 TFs, run the GNN 34 times per epoch, once with TF t marked, and score all 910 genes from that single run. That is 34 forward passes over a 910-node graph per epoch. (Marking only one side deviates from the full theory, which marks both targets; that is my adaptation. NBFNet below does the same single-source trick.)
   - Likelihood: moderate to high as an idea, since it gives the model a TF-specific view of every gene.

4. **Zhu, Z., Zhang, Z., Xhonneux, L.-P. and Tang, J. "Neural Bellman-Ford Networks: A General Graph Neural Network Framework for Link Prediction." NeurIPS 2021 (arXiv 2106.06935, 2021-06-13).** Verified (arXiv comment "NeurIPS 2021", repository citation).
   - Idea (NBFNet): The representation of a pair is a learned "sum over all paths of the product of edge representations". It is computed with a Bellman-Ford style propagation that starts from the source node only (that node gets a learned query vector, all other nodes start at zero) and spreads over the graph for a few steps. One run for a source computes the representation of every possible target. The score is a small MLP on the target node's final vector. Per the paper, edges that directly connect the query pair are dropped during training "to encourage the model to capture longer paths and prevent overfitting". That is a built-in fix for finding 4.
   - What extra information it uses: all the paths from the TF to the gene, learned.
   - Connection: findings 2, 3, 4, 5. It is TF-conditioned by construction, it can use paths such as TF to shared target to other TF to gene (finding 5), and it already handles the edge leakage.
   - Cost: the paper states one run costs O(|E| d + |V| d^2), where d is the hidden size. On 910 nodes and about 3,000 edges, with d = 32 to 64, one source takes milliseconds. With 34 sources per epoch this trains in minutes. Gene expression can be given as node or edge features.
   - Likelihood: moderate to high. It is the pair-aware model that fits our graph best (few sources, tiny graph). Risk: with 34 sources and about 3,000 edges, the model can overfit; use edge dropout and early stopping on a validation split.

5. **Yun, S., Kim, S., Lee, J., Kang, J. and Kim, H. J. "Neo-GNNs: Neighborhood Overlap-aware Graph Neural Networks for Link Prediction." NeurIPS 2021 (arXiv 2206.04216).** Verified (arXiv journal-reference field, conference listing).
   - Idea: Learn a function of the adjacency matrix that generalises neighbourhood-overlap heuristics (common neighbours, Adamic-Adar, resource allocation) and extends them to multi-hop overlaps, then add it to a GNN score.
   - What extra information it uses: overlap of neighbourhoods and degrees of the pair.
   - Connection: findings 1 and 5. It is a learned version of section 2.
   - Cost: low, but designed for a symmetric, non-bipartite graph; the bipartite adaptation (3-paths) needs a small change that I would have to design myself.
   - Likelihood: low to moderate. If hand-built heuristic features already help, Neo-GNN is not needed; if they do not, learned versions of the same thing are unlikely to.

6. **Wang, X., Yang, H. and Zhang, M. "Neural Common Neighbor with Completion for Link Prediction." ICLR 2024 (arXiv 2302.00890; the arXiv comment says ICLR 2024).** Verified.
   - Idea (NCN): Run a standard GNN once. For a pair (i, j), build the pair vector from the elementwise product of the two endpoint embeddings, concatenated with the sum of the embeddings of their common neighbours, and feed it to an MLP. During training, the target links are removed from the adjacency used by the GNN and the predictor (the paper does not spell out the reason; the natural reading is to keep the training graph looking like the test graph). NCNC adds a "completion" step that fills in likely missing common neighbours.
   - What extra information it uses: the identity and features of shared neighbours.
   - Connection: findings 2 and 4. The elementwise product is an interaction term, and the target-link removal is exactly the fix for finding 4.
   - Cost: low. The paper states NCN's overhead is similar to a plain GAE.
   - Caveat: in a purely bipartite graph a TF and a gene have no common neighbours, so NCN's second term would be empty. It becomes useful only if we add TF-TF edges, or gene-gene edges (STRING, co-expression; finding 5 says these add little). So for us the portable parts are the Hadamard product head and the target-link removal.
   - Likelihood: the head and the removal, high; the full model, low.

7. **Chamberlain, B. P., Shirobokov, S., Rossi, E., Frasca, F., Markovich, T., Hammerla, N., Bronstein, M. M. and Hansmire, M. "Graph Neural Networks for Link Prediction with Subgraph Sketching" (ELPH and BUDDY). ICLR 2023, oral (arXiv 2209.15486).** Verified (arXiv journal-reference field).
   - Idea: GNNs often lose to simple heuristics because they cannot count triangles and cannot tell apart nodes in identical structural roles. ELPH and BUDDY feed the model approximate counts of how many nodes lie at each pair of distances from the two endpoints (these counts include the common neighbour counts), estimated with hashing so no subgraph needs to be built.
   - What extra information it uses: the distance-count profile around the pair.
   - Connection: findings 1 and 5. These counts include the degree information and the shared-neighbour information in one vector.
   - Cost: the hashing is meant for huge graphs. On 910 nodes we can compute the counts exactly with matrix powers, so the "sketching" part is not needed at all.
   - Likelihood: moderate, and the cheapest version is a feature table (degrees, 3-path counts, Katz) fed to a small gradient-boosted model or logistic regression. That version is my adaptation, not the paper's method.

---

## 6. Evaluation pitfalls and leakage in link prediction

1. **Zhu, J., Zhou, Y., Ioannidis, V. N., Qian, S., Ai, W., Song, X. and Koutra, D. "Pitfalls in Link Prediction with Graph Neural Networks: Understanding the Impact of Target-link Inclusion and Better Practices." WSDM 2024, pages 994-1002, doi 10.1145/3616855.3635786 (arXiv 2306.00899).** Verified (DOI and arXiv record; pages from a repository entry).
   - Idea: Keeping the edges being predicted inside the message-passing graph, during training or testing, causes three problems: overfitting, distribution shift between training and test, and (at test) leakage that inflates the score. The effect is largest for low-degree nodes. Their fix, SpotTarget, removes each training edge from the graph while it is being scored if it touches a low-degree node, and removes all test edges to be predicted at test time.
   - Connection: this is finding 4, stated and studied. Our genes have mean degree about 3, so most of our genes are exactly the low-degree case where the paper says the effect is strongest.
   - Cost: low. Per batch, delete the batch's supervision edges from the adjacency used by the GNN, run the GNN, score, and put them back. The graph is small enough to rebuild the normalised adjacency each step.
   - Likelihood of improving within-TF AUROC: moderate. The main gain is honesty and generalisation (training stops rewarding "is my own edge in my neighbourhood"). It may lower the training fit and raise test performance.

2. **Li, J., Shomer, H., Mao, H., Zeng, S., Ma, Y., Shah, N., Tang, J. and Yin, D. "Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking" (HeaRT). NeurIPS 2023, Datasets and Benchmarks Track (arXiv 2306.10453, 2023-06-18).** Verified (proceedings page and arXiv).
   - Idea: Existing link prediction benchmarks have three problems: baselines are under-tuned, splits and metrics are inconsistent, and negatives are random and therefore easy. After fair tuning, simple models like GAE or GraphSAGE are competitive. They propose HeaRT, which picks hard negatives by combining resource allocation, personalised PageRank and feature similarity. Under HeaRT simple models rank in the top three more often, and the authors conclude that recent methods may have benefited from easy negatives. On ogbl-collab, Katz and resource allocation rank second and third.
   - Connection: findings 1 and 3. Random negatives let popularity win. Hard negatives chosen by heuristic similarity are a stricter test. Our within-TF AUROC already removes TF popularity from the comparison, but not gene popularity.
   - Cost: low. Hard negatives for us are the labelled non-targets that score high on a heuristic (for example high in-degree genes or genes with high co-expression with the TF's targets).
   - Likelihood: not a model change; an evaluation upgrade. It also tempers expectations: tuned simple models are hard to beat.

3. **Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., Catasta, M. and Leskovec, J. "Open Graph Benchmark: Datasets for Machine Learning on Graphs." NeurIPS 2020 (arXiv 2005.00687).** Verified (conference page; dataset conventions from the OGB website).
   - Idea: OGB standardises link prediction datasets with fixed splits, fixed evaluation negatives and ranking metrics. From the OGB website: ogbl-ppa uses a biological throughput split with Hits@100 against 3,000,000 sampled negatives; ogbl-collab splits by time (train to 2017, validation 2018, test 2019) with Hits@50; ogbl-ddi splits by which protein a drug targets with Hits@20; ogbl-citation2 and ogbl-wikikg2 use time splits and MRR; ogbl-biokg uses a random split. The ppa description says the graph is built from training edges only. HeaRT notes that for ogbl-collab, validation edges are added to the training graph at test time.
   - Connection: finding 4 and the evaluation list. Most OGB link tasks avoid pure random edge splits and use a split that mimics deployment. The nearest equivalent for us is a held-out-TF or held-out-gene split. All edges that the model reads at test time are training edges (plus validation edges when stated).
   - Cost: none to adopt the conventions.
   - Likelihood: not a model change.

4. **Mao, H., Li, J., Shomer, H., Li, B., Fan, W., Ma, Y., Zhao, T., Shah, N. and Tang, J. "Revisiting Link Prediction: A Data Perspective." ICLR 2024 (arXiv 2310.00793).** Verified (proceedings page).
   - Idea: Three data properties decide which link predictor works: local structural proximity, global structural proximity and feature proximity. Global structure helps only when local structure is weak, and feature and structure signals can conflict, which makes GNNs underperform when feature proximity dominates.
   - Connection: findings 3 and 5. Co-expression is a feature-proximity signal, shared targets is a structural-proximity signal. In our case structure is weak (low degree), so the paper predicts feature proximity (finding 3) is where gains are. It also warns that combining the two can hurt on some edges.
   - Cost: none (analysis).
   - Likelihood: not a model change; supports prioritising the co-expression-style interaction over exotic structure models.

5. **Aiyappa, R., Wang, X., Kim, M., Seckin, O. C., Yoon, J., Ahn, Y.-Y. and Kojaku, S. "Implicit degree bias in the link prediction task." arXiv 2405.14985 (2024-05-23). Published venue: unverified (I found only the arXiv record).**
   - Idea: Sampling test edges uniformly at random and drawing negatives uniformly makes high-degree nodes dominate the test set, so a degree-only method (preferential attachment) is "nearly optimal": the paper reports a mean AUROC of 0.83 across 90 graphs, dropping to 0.54 on a degree-corrected benchmark where negatives are drawn with the same degree bias. They also report that the degree-corrected setting trains better models by reducing overfitting to degree.
   - Connection: finding 1, directly. Our 0.872 vs 0.887 fits the pattern.
   - Cost: low. Draw negatives with probability proportional to gene degree (and TF degree), or equivalently stratify.
   - Likelihood: not a model change. It gives us a citation and a recipe for the degree-corrected control.

---

## 7. Biomedical pair-input prediction pitfalls

1. **Park, Y. and Marcotte, E. M. "Flaws in evaluation schemes for pair-input computational predictions." Nature Methods, 9:1134-1136, 2012, doi 10.1038/nmeth.2259.** Verified. (The author manuscript on PubMed Central is titled "A flaw in the typical evaluation scheme for pair-input computational predictions".)
   - Idea: In pair-input tasks (protein-protein interaction, drug-target, and so on) every test pair can be classed by how many of its two members appear in the training set. C1: both members appear in training pairs. C2: exactly one appears. C3: neither appears. Methods score much higher on C1 than on C2 and C3, across seven PPI predictors from SVMs to heuristics. The paper reports that ordinary random cross-validation test sets were more than 99% C1, and in the human protein-pair population the shares are about 19.2% C1, 49.2% C2 and 31.6% C3 (figures from the PubMed Central manuscript). They recommend reporting performance separately per class.
   - Connection: findings 1 and 3, strongly. Our random within-TF split is a pure C1 test: all 34 TFs are in training, and almost every gene has at least one training edge. A C1 score rewards learning properties of individual nodes (this TF is promiscuous, this gene is a hub), which is what the two-count model does. A model that has really learned something about regulation should also work on C2 (new gene) and C3-like (new TF and new gene) pairs.
   - Cost: low. Hold out a fraction of genes completely (all their edges go to test), and separately hold out whole TFs (leave-one-TF-out, 34 folds).
   - Likelihood: not a model change. This is the evaluation addition a reviewer familiar with the biology-ML literature is most likely to ask for (see section 10b).
   - Honest scope note: the Park & Marcotte text I read does not discuss node degree directly; the link to popularity is my reading, not theirs.

2. **Hamp, T. and Rost, B. "More challenges for machine-learning protein interactions." Bioinformatics, 31(10):1521-1525, 2015, doi 10.1093/bioinformatics/btu857.** The bibliographic record is verified (issue listing, dblp). I did not read the content, so what it argues is unverified. It is a Discovery Note on pitfalls in machine-learning prediction of protein interactions and is worth reading before citing.

3. **Zietz, M., Himmelstein, D. S., Kloster, K., Williams, C., Nagle, M. W. and Greene, C. S. "The probability of edge existence due to node degree: a baseline for network-based predictions." GigaScience, 13:giae001, 2024, doi 10.1093/gigascience/giae001.** Verified.
   - Idea: In biomedical networks (gene-disease, drug repurposing, gene function) node degree varies hugely. They build features from degree-preserving permutations of the network, so one can split a model's performance into the part explained by degree alone and the part due to the specific connections. They report that the performance not attributable to degree is often a small part of the total, and recommend the permutation baseline.
   - Connection: finding 1 and finding 5. For us the equivalent is: keep each TF's number of targets and each gene's number of regulators fixed, shuffle which pairs are edges, recompute the heuristic features, and see how much of each model's score survives. It also fits the finding that the gene-gene graphs add almost nothing beyond popularity.
   - Cost: low. The authors released a Python package (xswap).
   - Likelihood: not a model change; a standard control.

4. **Li, P., Shao, B., Zhao, G. and Liu, Z.-P. "Negative sampling strategies impact the prediction of scale-free biomolecular network interactions with machine learning." BMC Biology, 23(1), 2025, doi 10.1186/s12915-025-02231-w.** Title, authors, journal and year verified from a literature database record; I could not load the full text, so details beyond the abstract are unverified.
   - Idea: Biomolecular interaction networks are scale-free (a few nodes with many edges, most with few). Across tasks, datasets and models, the authors report that the network topology distorts how performance is measured, and they propose a degree distribution balanced (DDB) negative sampling strategy. They also say some models struggle to learn the interactions from molecular features alone. A search snippet attributed to the paper the claim that well-trained models predict pairs largely from node degree; I only saw that in a snippet, so treat it as unverified.
   - Connection: findings 1 and 2.
   - Cost: low. The recipe is to pick negatives so each node has about as many negatives as positives.
   - Likelihood: not a model change.

5. **Yilmaz, S., Yorgancioglu, K. and Koyuturk, M. "Bias-aware training and evaluation of link prediction algorithms in network biology." Proceedings of the National Academy of Sciences (PNAS), June 2025, doi 10.1073/pnas.2416646122.** Verified (journal index record; first-name initials as listed).
   - Idea: Uniformly sampled test edges favour "rich" (high-degree) nodes, and that bias persists even when evaluating on later snapshots of the network. They propose a weighted validation that emphasises low-degree nodes and training and evaluation strategies they call AWARE.
   - Connection: finding 1. It is the biology-specific version of Aiyappa et al.
   - Cost: low (weights in the validation metric).
   - Likelihood: not a model change. Also suggests choosing early stopping with a degree-weighted validation score.

6. **Bonner, S. et al. "Implications of topological imbalance for representation learning on biomedical knowledge graphs." Briefings in Bioinformatics, 2022, doi 10.1093/bib/bbac279 (arXiv 2112.06567).** Verified (indexed record; only the first author's name was checked).
   - Idea: In drug-discovery knowledge graphs, densely connected entities rank high regardless of context, across datasets, models and tasks. Perturbation experiments suggest the embedding models respond more to how often an entity appears than to the biological meaning of its relations.
   - Connection: finding 1 and finding 5. Same story in another biomedical domain.
   - Cost: none.
   - Likelihood: not a model change; a citation that this problem is documented in biomedical link prediction.

---

## 8. GNN-based GRN inference that already treats the problem as link prediction

Table of what I could verify. "Decoder" is how a TF-gene pair is turned into a score. "Split" is how test pairs are chosen.

| Method | Venue | Encoder | Decoder | Split and negatives | Held-out TF or gene? | Degree-only baseline? |
|---|---|---|---|---|---|---|
| GENELink (Chen and Liu) | Bioinformatics 38(19):4522-4529, 2022, doi 10.1093/bioinformatics/btac559 | Graph attention network on the known TF-gene graph with expression features | The repository README names a dot-product scorer (the default) and an MLP option; the README does not describe how the MLP scores | Not described on the repository page; unverified | Not found | Not found |
| GNNLink (Mao et al.) | Briefings in Bioinformatics 24(6):bbad414, 2023, doi 10.1093/bib/bbad414 | Two-layer GCN on the known GRN, gene features are expression vectors | Dot product of the two gene vectors, then an activation | Known gene pairs split randomly 3/5 train, 1/5 validation, 1/5 test. Each positive is paired with a negative that shares one of its genes, drawn at random. Split is over edges, not per TF | None reported | None reported |
| GCLink (Yu et al.) | Bioinformatics 41(3):btaf074, 2025 | Graph attention encoder plus contrastive learning | MLP per gene, then dot product and sigmoid | Split over gene pairs (2/3 train for the cell-type-specific networks, with a validation share). Only training positives build the adjacency matrix. Training and validation negatives drawn uniformly at 1:1. Test negative ratio is in a supplement I could not read | None described in the main text | None found |
| scTransNet (Kommu et al.) | ICML 2024 AI for Science workshop, arXiv 2407.18181 | Pretrained scBERT embeddings plus GNN over the known GRN | Not read | Not read | Not found | Not found |
| scRegNet (Kommu et al.) | Bioinformatics 41(Suppl 1):i619-i627, 2025, doi 10.1093/bioinformatics/btaf217 | Frozen single-cell foundation model embeddings joined with GNN embeddings | Each gene's joint embedding passes through an MLP; the two are concatenated; a fully connected layer and softmax give the probabilities. Additive when nothing nonlinear follows the concatenation | Per TF: for each TF, positive and negative targets are split about 67% train and 33% test; 10% of the training part is held out for validation; the GNN is built from training interactions only. The authors state that the same gene does not appear in both training and test for the same TF, which is narrower than pair-level independence (a gene can be in training for one TF and test for another) | None reported | None reported; the paper notes TFs are few and high-degree but does not analyse it |
| InfoSEM (Cui et al.) | ICML 2025 (arXiv 2503.04483, comment field) | Unsupervised generative model with text-embedding priors | Not applicable | Proposes a new benchmark framework to expose bias in supervised GRN methods | Their benchmark is designed to test generalisation; details unverified | Their abstract says supervised models risk learning gene-specific biases such as class imbalance |
| BEELINE (Pratapa et al.) | Nature Methods 17(2):147-154, 2020, doi 10.1038/s41592-019-0690-6 | The benchmark that produces the hESC data and ground-truth networks (cell-type-specific ChIP-seq, non-specific ChIP-seq, STRING) all these papers use | Not applicable | Reports moderate AUPRC for unsupervised methods and notes that methods that do not need pseudotime tend to be more accurate | Not applicable | Not applicable |

Notes on the table, with sources:

- GNNLink and GCLink facts come from fetched full texts; the scRegNet facts come from the open-access full text of the Bioinformatics version; the GENELink facts come from the repository README plus a database record for the title, volume and pages (the paper itself is not open access, so the split details are unverified).
- The BEELINE paper is the source of the data benchmark and ground truths; the "moderate AUPRC" finding is from its abstract summary.

What this means for us:

- **Do they have the additive problem?** The dot-product decoders (GENELink's default, GNNLink, GCLink) can in principle express TF-gene interactions, so the hard additive ceiling of finding 2 applies to scRegNet and its relatives (scTransNet is presumably the same, but I did not read its decoder), not to the whole field. But a pure dot product has no bias terms, so popularity has to be stored inside the embeddings, and none of these papers tests whether that is what is happening.
- **Do they have the popularity problem?** Almost certainly, because none reports a degree-only baseline and all split over known pairs. In scRegNet's case the authors state TFs are few and high-degree and move on. GNNLink's choice to pair each positive with a negative sharing one of its genes partly neutralises popularity on that side, but it is not a within-TF evaluation and the paper does not discuss it as such.
- **Do they have the leak?** GCLink and scRegNet both say the adjacency is built only from training positives, which prevents test leakage. That is not the same as the supervision leak of finding 4, which none of the three papers mentions.
- **A caution about the published rankings.** scRegNet is reported to beat GENELink and GNNLink on most datasets (the one exception I could see in the text is the mDC AUPRC, where GNNLink's 0.25 beats scRegNet's 0.12 to 0.17). Those baselines can express interactions and still scored lower in scRegNet's table. If the table is comparable (I did not check that all methods used identical splits), it suggests that on the random-pair benchmark the additive model is not at a disadvantage. That is consistent with finding 1 (the benchmark rewards popularity) and is a reason to expect that fixing the decoder shows up in within-TF AUROC rather than in pooled AUROC.

---

## 9. Putting the literature next to our five findings

| Finding | What the literature says | Suggested action |
|---|---|---|
| 1. Two counts reach 0.872 vs 0.887 | Preferential attachment is a known strong baseline on random-edge benchmarks (Aiyappa), also in biology (Zietz; Yilmaz; Li et al.; Bonner) | Keep the two-count model and a degree-permutation control in every table; keep within-TF AUROC as primary |
| 2. Concat + linear is additive | Labeling-trick theory (independent node embeddings), Rendle (dot product is the right default), Koren (biases plus dot product) | Replace the head with b_t + b_g + dot(u_t, v_g), low rank |
| 3. PANDA's useful part is co-expression of g with the TF's known targets | This is item-based collaborative filtering (Sarwar) computed from expression; LightGCN is mean aggregation plus a dot product | Give the model this quantity explicitly as the interaction term, with the gene itself excluded from the TF centroid during training |
| 4. Message-passing edges equal supervision edges | Zhu et al. (WSDM 2024), NBFNet (edge dropout), NCN (target-link removal) | Per-batch removal of supervision edges, or a fixed split of training positives into message-passing and supervision sets |
| 5. Gene-gene graphs add little, TF-TF cooperation is modest | Mao et al. says global structure helps only when local structure is weak; heuristics based on shared targets capture TF-TF cooperation | Use the TF-TF 3-path signal as a feature, not a new graph |

Two points that deserve to be said plainly because they change the order of work:

1. **The decoder fix and the leak fix are coupled.** Today the linear decoder cannot exploit the leak much, because it cannot combine TF and gene information. An interacting decoder can. Consider a gene with 3 training regulators. After mean aggregation over its training neighbours, its embedding is about one third made of the TF's embedding for every training positive. A dot product between that TF's vector and that gene's vector is then large exactly because the edge is in the graph, and at test time (edge absent) that boost vanishes. The model will happily learn "dot product high means edge is in my neighbourhood". So the decoder should be swapped together with target-edge removal, not before it. I expect that, if only the decoder is changed, training loss will look great and test within-TF AUROC may fall.
2. **Excluding the gene from the TF's centroid** (finding 3's feature) is the same leak at the feature level. Check that PANDA's TF-specific number was never computed from training targets that include the test-time-hidden pair, and, for a trained version, that it is leave-one-out on training positives.

---

## 10. Recommendations

### (a) Ranked top 5: what to implement first

1. **Bipartite collaborative filtering and Katz baselines with leave-one-edge-out (CPU only, no GPU needed).**
   Compute item-based CF (cosine on shared regulators), TF-based CF (cosine on shared targets), resource allocation, Adamic-Adar on 3-paths and bipartite Katz, all with the scored pair's own edge removed. Evaluate with within-TF AUROC next to the two-count model and PANDA. Justification: it takes an hour, the GPU is busy, and it answers the key question before any model work: how much of the TF-specific signal is just "TFs that share targets". It also gives reviewers the tuned-simple-baseline row that HeaRT and Dacrema et al. say is missing from most papers. My estimate: well-tuned versions land somewhere between the 0.72 within-TF baseline and the 0.843 additive ceiling; whether they pass the ceiling is uncertain.

2. **Leak-free training: remove the supervision edges from the message-passing graph.**
   Per batch (SpotTarget / NCN style), or with a fixed random split of training positives into a message-passing part and a supervision part that is re-drawn each epoch (at test time, use all training positives). Justification: it fixes finding 4, is a prerequisite for item 3, and is cheap. One trade-off to report: with a fixed split the training graph is sparser than the test graph (distribution shift), which per-batch removal minimises.

3. **Biased interacting decoder on top of the existing scRegNet embeddings: b_t + b_g + dot(u_t, v_g).**
   Keep Geneformer and the GNN as encoders. Use low rank (2 to 16) with weight decay, because 34 TFs gives each TF vector only about 89 positives and a few hundred labelled pairs to learn from. Try three TF vectors: free parameters, a projection of the TF's own embedding, and the mean embedding of its known targets (finding 3 made explicit). Justification: smallest change that removes the additive cap, and it is the recommender-system default (Koren, Rendle). After training, rerun the additive-fit R^2 check: it should drop below 1.0, which verifies the model uses interactions.

4. **A TF-conditioned propagation model in the NBFNet style (34 sources).**
   One source TF is marked, messages flow over the training graph (with the target edge dropout NBFNet already uses), and all 910 genes are scored from the one run. Expression can be a node feature. Justification: pair-aware by construction, leak-aware by construction, fits our graph size, and uses TF-TF cooperation (finding 5) through paths. It is the one deep model in this review I would bet on, though still moderately.

5. **A small stacker over pair features (BUDDY-style exact counts, but computed with matrices).**
   Features: d_t, d_g, 3-path counts, RA, Katz, the co-expression z-score of finding 3, and the scRegNet score (all computed leave-one-edge-out). Model: logistic regression or a small gradient-boosted tree. Justification: if a stacker over cheap features matches the deep models, the story in the paper becomes "interaction features matter", which is a cleaner message; if it does not, we know the deep models are adding something. Training takes seconds. SEAL or IGMC with h = 1 is the next step if we want a learned version of the same information.

Not in the top 5 on purpose: full RESCAL bilinear (too many parameters for 34 TFs), Neo-GNN (redundant with item 5), full NCNC (needs common neighbours that a bipartite graph lacks).

### (b) Evaluation changes to adopt so a reviewer does not object

1. **Held-out-gene test (Park and Marcotte C2).** Choose about 15 to 20% of genes, remove all their edges from training and from the message-passing graph, evaluate only on their pairs. Run several random draws. Popularity cannot help here, since the gene's degree is zero; a model that scores well must use expression or foundation-model information.
2. **Held-out-TF test (Park and Marcotte C3-like).** Leave one TF out at a time (34 folds), or five folds of about 7 TFs. The TF's training edges are hidden. The model must score its targets from expression or pretrained information alone. This is a strict test of whether a TF-specific term was really learned; the two-count model cannot run here because the TF's positive rate is unknown. Expect noisy numbers and report them per TF.
3. **Validation-based model selection.** All choices (epochs, rank, learning rate, heuristic parameters such as the Katz beta) are picked on a validation split carved from training, never on the test AUROC. I am not claiming the current runs use test numbers; the repository was not inspected. scRegNet's paper holds out 10% of training for validation, so this should be easy to keep. Ideally the validation metric is the within-TF AUROC, possibly degree-weighted (Yilmaz et al.).
4. **Report within-TF AUROC, within-TF AUPRC, and a per-TF table, with uncertainty.** Use bootstrap confidence intervals over TFs and over test pairs, at least 5 seeds, and a paired test across the 34 TFs (for example a Wilcoxon signed-rank test) when comparing two models. With 34 TFs, differences of 0.01 within-TF AUROC are unlikely to be significant.
5. **Gene-degree-stratified and degree-matched results.** Report AUROC by bins of gene training degree (0, 1, 2 to 3, 4 or more), as in Zhu et al., and also an AUROC where negatives are drawn with matching degree (Aiyappa; Li et al., BMC Biology). This isolates how much the model does beyond popularity.
6. **Null and control rows in every results table.** Two-count model (preferential attachment), additive-only model, degree-preserving permutation baseline (Zietz et al.), a shuffled-labels check, and a version of any new feature computed on the permuted graph.
7. **Leak audit.** State explicitly which edges are in the message-passing graph during training, validation and test; report results with and without the supervision edges in the graph (Zhu et al.). Check that no feature (including PANDA and co-expression centroids) was computed with the scored pair's label.
8. **Rank-based and top-k metrics per TF.** Hits@k or precision at k within each TF's candidate list, and mean reciprocal rank, as in the knowledge-graph literature (DistMult and OGB use them). A biologist uses a method by looking at a TF's top candidates.
9. **Say what the negatives are.** Non-targets in these datasets come from the absence of a ChIP-seq peak, not from proof of no regulation, so some negatives are probably false negatives. This is not studied in the papers above, so I do not cite a remedy; just state the limitation. If we want a remedy, positive-unlabeled learning is the general area to read.
10. **More than one dataset and ground truth.** Repeat on at least the other BEELINE human line and on the other ground-truth networks. STRING-based truth is partly protein association, which is popularity-heavy, so conclusions may differ.

### (c) Anything in the literature that contradicts or undercuts our findings

- **"Popularity" is not always a flaw.** Park and Marcotte are clear that C1-heavy test sets are a problem when the real use case is C2 or C3. If the real use case is completing a partly known network for well-studied TFs, then C1 performance is legitimate and a popularity-driven model is fine. Our claim that "the benchmark rewards popularity" is true but whether it matters depends on the application. Within-TF AUROC is the right metric only if the application is "for a given TF, rank its candidate targets". If the application is a single global ranking of all pairs, the TF-only term (the TF's positive rate) is a legitimate and useful part of the score, and the additive model's 0.887 is the relevant number.
- **The 0.843 ceiling is a statement about additive models only.** Rendle et al. do not say an MLP on the concatenation cannot interact; they say it is hard to learn and not better than a dot product. If someone adds a nonlinear hidden layer after the concatenation, the cap no longer applies strictly. Our check (additive-fit R^2 = 1.0) shows the current model does not interact in practice, but a reviewer may read the cap as covering more than it does.
- **Tuned simple models may match fancy ones.** HeaRT and Dacrema et al. both report that tuned simple baselines match or beat complicated models. If the interacting decoder and heuristics reach the same within-TF AUROC as NBFNet or SEAL, that is the expected outcome, not a failure. It also means the headline claim "scRegNet's advantage over popularity is small" should be backed by tuned baselines, not default ones.
- **Gains from pair-aware structure may be small on our graph.** Mao et al. (data perspective) say global and pair structure help mainly when local structure is weak and feature proximity does not conflict. Our gene degrees are very low (about 3), and the useful TF-specific signal (finding 3) is feature-based. That does not contradict finding 3, but it suggests the structural models in section 5 may add little on top of it.
- **The claim in finding 3 that a GNN "could compute it itself" is a hypothesis, not a result.** Rendle et al. show that learned networks have difficulty even recovering a dot product from data. So the explicit dot-product decoder may succeed where the current GNN, in practice, does not; this supports the proposal but means we should not assume the capability is already present.
- **Co-expression with known targets can be a shortcut of its own kind.** A gene that looks like the TF's other known targets may simply belong to the same co-regulated module (regulated by other TFs too), which is similarity to training positives of the same TF rather than TF-specific binding. In Park and Marcotte's framing this is another C1-friendly feature. The held-out-TF test above is what separates real TF-specific signal from module membership. This might lower the apparent value of finding 3 once evaluated that way.
- **Foundation-model gains need an ablation.** Ahlmann-Eltze, Huber and Anders (Nature Methods 22(8):1657-1661, 2025, doi 10.1038/s41592-025-02772-6) report that deep-learning and foundation-model methods, including scGPT and scFoundation, did not beat deliberately simple baselines at predicting perturbation effects. It is a different task (perturbation prediction, not GRN link prediction), so it does not contradict our findings, but it is a precedent a reviewer may use to ask whether the Geneformer branch adds anything beyond expression and graph degree. A cheap answer is an ablation that swaps Geneformer embeddings for random vectors of the same size or for a simple degree feature.
- **Nothing I found contradicts finding 4 or finding 5.** Finding 4 is supported by Zhu et al., and finding 5's modest TF-TF effect agrees with Mao et al.'s view of when structure helps. I did not find a GRN paper that tests TF-TF cooperation, so that part has no literature check either way.

---

## Appendix: verification log and unverified items

Verified in this session with arXiv records plus a venue source (proceedings or journal page, repository citation, or database record): VGAE (workshop), SEAL, IGMC, labeling trick, NBFNet, Neo-GNN, NCN, ELPH and BUDDY, Rendle et al., HeaRT, OGB, Mao et al., Zhu et al., DistMult, LightGCN (venue), Dacrema et al., Zhou et al. (EPJB), Lü and Zhou, Liben-Nowell and Kleinberg, Katz, Barabási and Albert, Koren (IEEE Computer, KDD 2008), Sarwar et al., RESCAL, Park and Marcotte, Zietz et al., Yilmaz et al., Bonner et al., Li et al. (BMC Biology; metadata only), GENELink (metadata only), GNNLink, GCLink, scRegNet, scTransNet (arXiv), InfoSEM (arXiv comment), BEELINE, Ahlmann-Eltze et al.

Unverified or only partly verified:

- Adamic and Adar (2003): volume and pages from a secondary reading list.
- GC-MC: arXiv date verified, venue not verified.
- Aiyappa et al. (2024): only an arXiv preprint record found; published venue unverified.
- Hamp and Rost (2015): bibliographic record only; content not read.
- Li et al. (BMC Biology 2025): metadata verified, full text not read; the "models predict from degree alone" statement came from a search snippet.
- GENELink: decoder options from the repository README only; the data split and negatives are unverified.
- scTransNet: decoder and splits not read.
- scRegNet: whether any nonlinearity follows the concatenation is not stated in what I read; the "MLP, concatenate, fully connected layer, softmax" description is from the open-access text.
- SEAL: whether the target link is removed from the enclosing subgraph is not stated in the text I searched; implementations may differ.
- Koren et al. (2009): the exact form of the bias terms was not re-read this session.
- Park and Marcotte: class shares (19.2%, 49.2%, 31.6%) and the ">99% C1" figure are from the PubMed Central author manuscript; the Nature Methods version was paywalled.
- The "likelihood" estimates and the adaptations marked "my adaptation" (one-sided labeling, exact counts instead of sketches, a pair-feature stacker, bipartite 3-path heuristics, the leak arithmetic in section 2.1 and section 9) are my own reasoning and have not been tested on our data.
