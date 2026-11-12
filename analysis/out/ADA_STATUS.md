# ADA status (appended by the ada worker)
Setup 2026-11-12: ~/venv312 has torch 2.5.1+cu121, torch_geometric 2.7.0, sklearn 1.5.1, scipy 1.14.1, pandas 2.2.2, numpy 2.2.6; cuda True on 2080 Ti.
Smoke ada_smoke_hero_s42 (P=1): refit AUROC 0.8867 / AUPRC 0.6223 / within-TF 0.7454, 0.86 min, MaxRSS 0.88 GB. Local w3_hero_s42: 0.8890 / 0.6264 / 0.7499, 0.27 min.
Defaults: pair_gnn P=6 (-c 12, 36G), nbf/seal P=2. Inbox files of the same module are merged and split over free GPUs; _s45/_s46 runs are dropped.
## Errors
## Log
- 2026-11-12 18:42:01 SUBMIT w12_nbf_sweep_c1: 2 runs, mod=src.nbf, P=2, merged files=1, s45/s46 dropped=0; Submitted batch job 74353
- 2026-11-12 18:42:02 SUBMIT w12_nbf_sweep_c2: 2 runs, mod=src.nbf, P=2, merged files=1, s45/s46 dropped=0; Submitted batch job 74355
- 2026-11-12 18:42:03 SUBMIT w12_nbf_sweep_c3: 2 runs, mod=src.nbf, P=2, merged files=1, s45/s46 dropped=0; Submitted batch job 74356
- 2026-11-12 18:42:05 SUBMIT w12_nbf_sweep_c4: 2 runs, mod=src.nbf, P=2, merged files=1, s45/s46 dropped=0; Submitted batch job 74357
- worker B (seal): each run extracts subgraphs with 2 CPU workers (--workers 2), GPU use small (<1.5 GB, P=2 fine; h2 runs may use ~3 GB).
- 2026-11-12 18:44:19 SUBMIT w12_seal_sweep: 6 runs, mod=src.seal, P=2, merged files=1, s45/s46 dropped=0; Submitted batch job 74371
- 2026-11-12 18:44:20 jobs: w12_seal_sweep:PENDING w12_nbf_sweep_c4:RUNNING w12_nbf_sweep_c3:RUNNING w12_nbf_sweep_c1:RUNNING  | inbox: empty | new tracebacks: none
    w12_nbf_sweep_c1.txt: 0/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 0/2
    w12_nbf_sweep_c4.txt: 0/2
    w12_seal_sweep.txt: 0/6
- 2026-11-12 18:49:23 SUBMIT w12_nbf_sweep2_c1: 5 runs, mod=src.nbf, P=2, merged files=1, s45/s46 dropped=0; Submitted batch job 74384
- 2026-11-12 18:49:24 SUBMIT w12_nbf_sweep2_c2: 5 runs, mod=src.nbf, P=2, merged files=1, s45/s46 dropped=0; Submitted batch job 74385
- 2026-11-12 18:49:25 jobs: w12_nbf_sweep2_c2:PENDING w12_nbf_sweep2_c1:PENDING w12_seal_sweep:RUNNING w12_nbf_sweep_c1:RUNNING  | inbox: empty | new tracebacks: none
    w12_nbf_sweep2_c1.txt: 0/5
    w12_nbf_sweep2_c2.txt: 0/5
    w12_nbf_sweep_c1.txt: 0/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_seal_sweep.txt: 4/6
- 2026-11-12 18:54:55 SUBMIT w12_4_cs_c1: 22 runs, mod=src.pair_gnn, P=6, merged files=2, s45/s46 dropped=0; Submitted batch job 74395
- 2026-11-12 18:54:55 SUBMIT w12_4_cs_c2: 21 runs, mod=src.pair_gnn, P=6, merged files=2, s45/s46 dropped=0; Submitted batch job 74396
- 2026-11-12 18:54:57 jobs: w12_4_cs_c2:PENDING w12_4_cs_c1:PENDING w12_nbf_sweep2_c2:RUNNING w12_nbf_sweep_c1:RUNNING  | inbox: w12_seal_confirm.txt  | new tracebacks: none
    w12_4_cs_c1.txt: 0/22
    w12_4_cs_c2.txt: 0/21
- 2026-11-12 18:57:56 SUBMIT w12_seal_confirm: 11 runs, mod=src.seal, P=2, merged files=1, s45/s46 dropped=0; Submitted batch job 74401

### ERR analysis/out/pair/w12_cs_herogf_s43.out (18:57:58)
```
        ^^^^^^^^^^^^^^^
  File "/home2/garimella.abhishek/scRegNet/src/pair_gnn.py", line 233, in prepare
    from src.train import Trainer
  File "/home2/garimella.abhishek/scRegNet/src/train.py", line 13, in <module>
    from src.utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation, build_ppi_adjacency
  File "/home2/garimella.abhishek/scRegNet/src/utils.py", line 9, in <module>
    import colorlog
ModuleNotFoundError: No module named 'colorlog'
```
- 2026-11-12 18:57:58 jobs: w12_seal_confirm:PENDING w12_4_cs_c2:RUNNING w12_4_cs_c1:RUNNING w12_nbf_sweep_c1:RUNNING  | inbox: empty | new tracebacks: analysis/out/pair/w12_cs_herogf_s43.out
    w12_4_cs_c1.txt: 7/22
    w12_4_cs_c2.txt: 0/21
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 0/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_seal_confirm.txt: 0/11
    w12_seal_sweep.txt: 6/6
- NOTE 18:59 FIXED: runs w12_cs_herogf_s42/43/44 (scfm 1) failed with ModuleNotFoundError colorlog (src.train imports it). colorlog 6.12.0 was added to ~/venv312 by wheel unzip. The 3 runs were re-queued as w12_rerun_herogf (ada_inbox). Other scfm 1 jobs submitted from now on are OK.
- 2026-11-12 19:01:19 SUBMIT w12_nbf_confirm: 28 runs, mod=src.nbf, P=2, merged files=2, s45/s46 dropped=0; Submitted batch job 74406

### ERR analysis/out/pair/w12_cs_herogf_s42.out (19:01:22)
```
        ^^^^^^^^^^^^^^^
  File "/home2/garimella.abhishek/scRegNet/src/pair_gnn.py", line 233, in prepare
    from src.train import Trainer
  File "/home2/garimella.abhishek/scRegNet/src/train.py", line 13, in <module>
    from src.utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation, build_ppi_adjacency
  File "/home2/garimella.abhishek/scRegNet/src/utils.py", line 9, in <module>
    import colorlog
ModuleNotFoundError: No module named 'colorlog'
```

### ERR analysis/out/pair/w12_cs_herogf_s44.out (19:01:22)
```
        ^^^^^^^^^^^^^^^
  File "/home2/garimella.abhishek/scRegNet/src/pair_gnn.py", line 233, in prepare
    from src.train import Trainer
  File "/home2/garimella.abhishek/scRegNet/src/train.py", line 13, in <module>
    from src.utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation, build_ppi_adjacency
  File "/home2/garimella.abhishek/scRegNet/src/utils.py", line 9, in <module>
    import colorlog
ModuleNotFoundError: No module named 'colorlog'
```
- 2026-11-12 19:01:22 jobs: w12_nbf_confirm:PENDING w12_seal_confirm:RUNNING w12_4_cs_c2:RUNNING w12_nbf_sweep_c1:RUNNING  | inbox: w12_rerun_herogf.txt w12_seal_ho.txt  | new tracebacks: analysis/out/pair/w12_cs_herogf_s42.out analysis/out/pair/w12_cs_herogf_s44.out
    w12_4_cs_c1.txt: 21/22
    w12_4_cs_c2.txt: 14/21
    w12_nbf_confirm.txt: 0/28
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 1/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_seal_confirm.txt: 0/11
    w12_seal_sweep.txt: 6/6
- 2026-11-12 19:06:24 SUBMIT w12_seal_ho: 20 runs, mod=src.seal, P=2, merged files=1, s45/s46 dropped=0; Submitted batch job 74414
- 2026-11-12 19:06:26 jobs: w12_seal_ho:PENDING w12_nbf_confirm:RUNNING w12_seal_confirm:RUNNING w12_nbf_sweep_c1:RUNNING  | inbox: w12_rerun_herogf.txt  | new tracebacks: none
    w12_4_cs_c1.txt: 21/22
    w12_4_cs_c2.txt: 19/21
    w12_nbf_confirm.txt: 4/28
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 1/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_seal_confirm.txt: 6/11
    w12_seal_ho.txt: 0/20
    w12_seal_sweep.txt: 6/6
- 2026-11-12 19:12:32 SUBMIT w12_rerun_herogf: 33 runs, mod=src.pair_gnn, P=6, merged files=2, s45/s46 dropped=0; Submitted batch job 74423
- 2026-11-12 19:12:34 jobs: w12_rerun_herogf:PENDING w12_seal_ho:RUNNING w12_nbf_confirm:RUNNING w12_nbf_sweep_c1:RUNNING  | inbox: w12_nbf_deg.txt  | new tracebacks: none
    w12_4_cs_c1.txt: 21/22
    w12_4_cs_c2.txt: 19/21
    w12_nbf_confirm.txt: 8/28
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 1/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 0/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 6/20
    w12_seal_sweep.txt: 6/6
- 2026-11-12 19:18:37 SUBMIT w12_nbf_deg: 9 runs, mod=src.nbf, P=2, merged files=2, s45/s46 dropped=0; Submitted batch job 74433
- 2026-11-12 19:18:41 jobs: w12_nbf_deg:PENDING w12_rerun_herogf:RUNNING w12_seal_ho:RUNNING w12_nbf_confirm:RUNNING  | inbox: w12_seal_pca.txt  | new tracebacks: none
    w12_4_cs_c1.txt: 21/22
    w12_4_cs_c2.txt: 19/21
    w12_nbf_confirm.txt: 12/28
    w12_nbf_deg.txt: 0/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 0/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 13/20
    w12_seal_sweep.txt: 6/6
- 2026-11-12 19:21:52 jobs: w12_nbf_deg:PENDING w12_rerun_herogf:RUNNING w12_seal_ho:RUNNING w12_nbf_confirm:RUNNING  | inbox: w12_seal_pca.txt  | new tracebacks: none
    w12_4_cs_c1.txt: 21/22
    w12_4_cs_c2.txt: 19/21
    w12_nbf_confirm.txt: 16/28
    w12_nbf_deg.txt: 0/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 0/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 17/20
    w12_seal_sweep.txt: 6/6
- 2026-11-12 19:24:29 SUBMIT w12_seal_pca: 6 runs, mod=src.seal, P=2, merged files=1, s45/s46 dropped=0; Submitted batch job 74438
- 2026-11-12 19:24:30 jobs: w12_seal_pca:PENDING w12_nbf_deg:RUNNING w12_rerun_herogf:RUNNING w12_nbf_confirm:RUNNING  | inbox: empty | new tracebacks: none
    w12_4_cs_c1.txt: 21/22
    w12_4_cs_c2.txt: 19/21
    w12_nbf_confirm.txt: 16/28
    w12_nbf_deg.txt: 0/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 0/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 20/20
    w12_seal_pca.txt: 0/6
    w12_seal_sweep.txt: 6/6
- 2026-11-12 19:27:16 jobs: w12_seal_pca:PENDING w12_nbf_deg:RUNNING w12_rerun_herogf:RUNNING w12_nbf_confirm:RUNNING  | inbox: empty | new tracebacks: none
    w12_4_cs_c1.txt: 21/22
    w12_4_cs_c2.txt: 19/21
    w12_nbf_confirm.txt: 19/28
    w12_nbf_deg.txt: 5/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 0/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 20/20
    w12_seal_pca.txt: 0/6
    w12_seal_sweep.txt: 6/6
- 2026-11-12 19:30:08 jobs: w12_seal_pca:RUNNING w12_rerun_herogf:RUNNING w12_nbf_confirm:RUNNING  | inbox: empty | new tracebacks: none
    w12_4_cs_c1.txt: 21/22
    w12_4_cs_c2.txt: 19/21
    w12_nbf_confirm.txt: 21/28
    w12_nbf_deg.txt: 9/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 0/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 20/20
    w12_seal_pca.txt: 1/6
    w12_seal_sweep.txt: 6/6
- 2026-11-12 19:33:06 SUBMIT w12_ho_a: 60 runs, mod=src.pair_gnn, P=6, merged files=1, s45/s46 dropped=0; Submitted batch job 74452
- 2026-11-12 19:33:07 jobs: w12_ho_a:PENDING w12_seal_pca:RUNNING w12_rerun_herogf:RUNNING w12_nbf_confirm:RUNNING  | inbox: empty | new tracebacks: none
    w12_4_cs_c1.txt: 21/22
    w12_4_cs_c2.txt: 19/21
    w12_ho_a.txt: 0/60
    w12_nbf_confirm.txt: 23/28
    w12_nbf_deg.txt: 9/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 0/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 20/20
    w12_seal_pca.txt: 4/6
    w12_seal_sweep.txt: 6/6
- 2026-11-12 19:36:07 jobs: w12_ho_a:RUNNING w12_rerun_herogf:RUNNING w12_nbf_confirm:RUNNING  | inbox: empty | new tracebacks: none
    w12_4_cs_c1.txt: 21/22
    w12_4_cs_c2.txt: 19/21
    w12_ho_a.txt: 6/60
    w12_nbf_confirm.txt: 26/28
    w12_nbf_deg.txt: 9/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 0/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 20/20
    w12_seal_pca.txt: 6/6
    w12_seal_sweep.txt: 6/6
- 2026-11-12 19:38:59 jobs: w12_ho_a:RUNNING w12_rerun_herogf:RUNNING  | inbox: empty | new tracebacks: none
    w12_4_cs_c1.txt: 21/22
    w12_4_cs_c2.txt: 19/21
    w12_ho_a.txt: 33/60
    w12_nbf_confirm.txt: 28/28
    w12_nbf_deg.txt: 9/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 0/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 20/20
    w12_seal_pca.txt: 6/6
    w12_seal_sweep.txt: 6/6
- 2026-11-12 19:42:00 jobs: w12_ho_a:RUNNING w12_rerun_herogf:RUNNING  | inbox: empty | new tracebacks: none
    w12_4_cs_c1.txt: 21/22
    w12_4_cs_c2.txt: 19/21
    w12_ho_a.txt: 45/60
    w12_nbf_confirm.txt: 28/28
    w12_nbf_deg.txt: 9/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 0/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 20/20
    w12_seal_pca.txt: 6/6
    w12_seal_sweep.txt: 6/6
- 2026-11-12 19:44:59 jobs: w12_ho_a:RUNNING w12_rerun_herogf:RUNNING  | inbox: empty | new tracebacks: none
    w12_4_cs_c1.txt: 21/22
    w12_4_cs_c2.txt: 19/21
    w12_ho_a.txt: 56/60
    w12_nbf_confirm.txt: 28/28
    w12_nbf_deg.txt: 9/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 0/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 20/20
    w12_seal_pca.txt: 6/6
    w12_seal_sweep.txt: 6/6
- 2026-11-12 19:47:55 jobs: w12_rerun_herogf:RUNNING  | inbox: empty | new tracebacks: none
    w12_4_cs_c1.txt: 21/22
    w12_4_cs_c2.txt: 19/21
    w12_ho_a.txt: 60/60
    w12_nbf_confirm.txt: 28/28
    w12_nbf_deg.txt: 9/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 0/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 20/20
    w12_seal_pca.txt: 6/6
    w12_seal_sweep.txt: 6/6
- 2026-11-12 19:50:52 jobs: w12_rerun_herogf:RUNNING  | inbox: empty | new tracebacks: none
    w12_4_cs_c1.txt: 21/22
    w12_4_cs_c2.txt: 19/21
    w12_ho_a.txt: 60/60
    w12_nbf_confirm.txt: 28/28
    w12_nbf_deg.txt: 9/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 0/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 20/20
    w12_seal_pca.txt: 6/6
    w12_seal_sweep.txt: 6/6
- 2026-11-12 19:53:52 SUBMIT w12_1_drop_c1: 8 runs, mod=src.pair_gnn, P=6, merged files=1, s45/s46 dropped=0; Submitted batch job 74522
- 2026-11-12 19:53:53 SUBMIT w12_1_drop_c2: 8 runs, mod=src.pair_gnn, P=6, merged files=1, s45/s46 dropped=0; Submitted batch job 74523
- 2026-11-12 19:53:54 SUBMIT w12_1_drop_c3: 7 runs, mod=src.pair_gnn, P=6, merged files=1, s45/s46 dropped=0; Submitted batch job 74524
- 2026-11-12 19:53:55 jobs: w12_1_drop_c2:RUNNING w12_1_drop_c3:RUNNING w12_1_drop_c1:RUNNING w12_rerun_herogf:RUNNING  | inbox: empty | new tracebacks: none
    w12_1_drop_c1.txt: 0/8
    w12_1_drop_c2.txt: 0/8
    w12_1_drop_c3.txt: 0/7
    w12_4_cs_c1.txt: 21/22
    w12_4_cs_c2.txt: 19/21
    w12_ho_a.txt: 60/60
    w12_nbf_confirm.txt: 28/28
    w12_nbf_deg.txt: 9/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 8/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 20/20
    w12_seal_pca.txt: 6/6
    w12_seal_sweep.txt: 6/6
- 2026-11-12 19:56:53 jobs: w12_1_drop_c2:RUNNING w12_rerun_herogf:RUNNING  | inbox: empty | new tracebacks: none
    w12_1_drop_c1.txt: 8/8
    w12_1_drop_c2.txt: 7/8
    w12_1_drop_c3.txt: 7/7
    w12_4_cs_c1.txt: 21/22
    w12_4_cs_c2.txt: 19/21
    w12_ho_a.txt: 60/60
    w12_nbf_confirm.txt: 28/28
    w12_nbf_deg.txt: 9/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 29/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 20/20
    w12_seal_pca.txt: 6/6
    w12_seal_sweep.txt: 6/6
- 2026-11-12 19:59:49 SUBMIT w12_1_uni_c1: 9 runs, mod=src.pair_gnn, P=6, merged files=2, s45/s46 dropped=0; Submitted batch job 74529
- 2026-11-12 19:59:50 SUBMIT w12_1_uni_c2: 8 runs, mod=src.pair_gnn, P=6, merged files=2, s45/s46 dropped=0; Submitted batch job 74530
- 2026-11-12 19:59:51 SUBMIT w12_1_uni_c3: 8 runs, mod=src.pair_gnn, P=6, merged files=2, s45/s46 dropped=0; Submitted batch job 74531
- 2026-11-12 19:59:52 jobs: w12_1_uni_c3:PENDING w12_1_uni_c2:PENDING w12_1_uni_c1:PENDING w12_rerun_herogf:RUNNING  | inbox: empty | new tracebacks: none
    w12_1_drop_c1.txt: 8/8
    w12_1_drop_c2.txt: 8/8
    w12_1_drop_c3.txt: 7/7
    w12_1_uni_c1.txt: 0/9
    w12_1_uni_c2.txt: 0/8
    w12_1_uni_c3.txt: 0/8
    w12_4_cs_c1.txt: 22/22
    w12_4_cs_c2.txt: 21/21
    w12_ho_a.txt: 60/60
    w12_nbf_confirm.txt: 28/28
    w12_nbf_deg.txt: 9/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 33/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 20/20
    w12_seal_pca.txt: 6/6
    w12_seal_sweep.txt: 6/6
- 2026-11-12 20:02:49 jobs: w12_1_uni_c1:RUNNING w12_1_uni_c2:RUNNING w12_1_uni_c3:RUNNING w12_rerun_herogf:RUNNING  | inbox: w12_6_cells.txt  | new tracebacks: none
    w12_1_drop_c1.txt: 8/8
    w12_1_drop_c2.txt: 8/8
    w12_1_drop_c3.txt: 7/7
    w12_1_uni_c1.txt: 8/9
    w12_1_uni_c2.txt: 7/8
    w12_1_uni_c3.txt: 7/8
    w12_4_cs_c1.txt: 22/22
    w12_4_cs_c2.txt: 21/21
    w12_ho_a.txt: 60/60
    w12_nbf_confirm.txt: 28/28
    w12_nbf_deg.txt: 9/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 33/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 20/20
    w12_seal_pca.txt: 6/6
    w12_seal_sweep.txt: 6/6
- 2026-11-12 20:05:45 jobs: w12_1_uni_c1:RUNNING w12_1_uni_c2:RUNNING w12_1_uni_c3:RUNNING w12_rerun_herogf:RUNNING  | inbox: w12_6_cells.txt  | new tracebacks: none
    w12_1_drop_c1.txt: 8/8
    w12_1_drop_c2.txt: 8/8
    w12_1_drop_c3.txt: 7/7
    w12_1_uni_c1.txt: 8/9
    w12_1_uni_c2.txt: 7/8
    w12_1_uni_c3.txt: 7/8
    w12_4_cs_c1.txt: 22/22
    w12_4_cs_c2.txt: 21/21
    w12_ho_a.txt: 60/60
    w12_nbf_confirm.txt: 28/28
    w12_nbf_deg.txt: 9/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 33/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 20/20
    w12_seal_pca.txt: 6/6
    w12_seal_sweep.txt: 6/6
- 2026-11-12 20:08:42 SUBMIT w12_6_cells_c1: 18 runs, mod=src.pair_gnn, P=6, merged files=1, s45/s46 dropped=0; Submitted batch job 74535
- 2026-11-12 20:08:43 SUBMIT w12_6_cells_c2: 18 runs, mod=src.pair_gnn, P=6, merged files=1, s45/s46 dropped=0; Submitted batch job 74536
- 2026-11-12 20:08:44 jobs: w12_6_cells_c2:PENDING w12_6_cells_c1:PENDING w12_1_uni_c2:RUNNING w12_rerun_herogf:RUNNING  | inbox: empty | new tracebacks: none
    w12_1_drop_c1.txt: 8/8
    w12_1_drop_c2.txt: 8/8
    w12_1_drop_c3.txt: 7/7
    w12_1_uni_c1.txt: 9/9
    w12_1_uni_c2.txt: 7/8
    w12_1_uni_c3.txt: 8/8
    w12_4_cs_c1.txt: 22/22
    w12_4_cs_c2.txt: 21/21
    w12_6_cells_c1.txt: 0/18
    w12_6_cells_c2.txt: 0/18
    w12_ho_a.txt: 60/60
    w12_nbf_confirm.txt: 28/28
    w12_nbf_deg.txt: 9/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 33/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 20/20
    w12_seal_pca.txt: 6/6
    w12_seal_sweep.txt: 6/6
- 2026-11-12 20:11:46 jobs: w12_6_cells_c1:RUNNING w12_6_cells_c2:RUNNING w12_rerun_herogf:RUNNING  | inbox: empty | new tracebacks: none
    w12_1_drop_c1.txt: 8/8
    w12_1_drop_c2.txt: 8/8
    w12_1_drop_c3.txt: 7/7
    w12_1_uni_c1.txt: 9/9
    w12_1_uni_c2.txt: 8/8
    w12_1_uni_c3.txt: 8/8
    w12_4_cs_c1.txt: 22/22
    w12_4_cs_c2.txt: 21/21
    w12_6_cells_c1.txt: 0/18
    w12_6_cells_c2.txt: 0/18
    w12_ho_a.txt: 60/60
    w12_nbf_confirm.txt: 28/28
    w12_nbf_deg.txt: 9/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 33/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 20/20
    w12_seal_pca.txt: 6/6
    w12_seal_sweep.txt: 6/6
- NOTE 20:13: ada skips a run only if its json exists ON ADA; local jsons (laptop runs) are now rsynced to ada before each poll/submission so duplicates are skipped. Job w12_rerun_herogf (queued 19:13) contained 32 runs the laptop finished meanwhile; they will be skipped from now on.
- 2026-11-12 20:14:58 SUBMIT w12_1_uni_cells: 9 runs, mod=src.pair_gnn, P=6, merged files=1, s45/s46 dropped=0; Submitted batch job 74539
- 2026-11-12 20:15:00 jobs: w12_1_uni_cells:PENDING w12_6_cells_c1:RUNNING w12_6_cells_c2:RUNNING w12_rerun_herogf:RUNNING  | inbox: empty | new tracebacks: none
    w12_1_drop_c1.txt: 8/8
    w12_1_drop_c2.txt: 8/8
    w12_1_drop_c3.txt: 7/7
    w12_1_uni_c1.txt: 9/9
    w12_1_uni_c2.txt: 8/8
    w12_1_uni_c3.txt: 8/8
    w12_1_uni_cells.txt: 0/9
    w12_4_cs_c1.txt: 22/22
    w12_4_cs_c2.txt: 21/21
    w12_6_cells_c1.txt: 0/18
    w12_6_cells_c2.txt: 0/18
    w12_ho_a.txt: 60/60
    w12_nbf_confirm.txt: 28/28
    w12_nbf_deg.txt: 9/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 33/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 20/20
    w12_seal_pca.txt: 6/6
    w12_seal_sweep.txt: 6/6
- 2026-11-12 20:17:55 jobs: w12_1_uni_cells:RUNNING w12_6_cells_c1:RUNNING w12_6_cells_c2:RUNNING w12_rerun_herogf:RUNNING  | inbox: empty | new tracebacks: none
    w12_1_drop_c1.txt: 8/8
    w12_1_drop_c2.txt: 8/8
    w12_1_drop_c3.txt: 7/7
    w12_1_uni_c1.txt: 9/9
    w12_1_uni_c2.txt: 8/8
    w12_1_uni_c3.txt: 8/8
    w12_1_uni_cells.txt: 6/9
    w12_4_cs_c1.txt: 22/22
    w12_4_cs_c2.txt: 21/21
    w12_6_cells_c1.txt: 1/18
    w12_6_cells_c2.txt: 1/18
    w12_ho_a.txt: 60/60
    w12_nbf_confirm.txt: 28/28
    w12_nbf_deg.txt: 9/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 33/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 20/20
    w12_seal_pca.txt: 6/6
    w12_seal_sweep.txt: 6/6
- 2026-11-12 20:20:53 jobs: w12_1_uni_cells:RUNNING w12_6_cells_c1:RUNNING w12_6_cells_c2:RUNNING w12_rerun_herogf:RUNNING  | inbox: empty | new tracebacks: none
    w12_1_drop_c1.txt: 8/8
    w12_1_drop_c2.txt: 8/8
    w12_1_drop_c3.txt: 7/7
    w12_1_uni_c1.txt: 9/9
    w12_1_uni_c2.txt: 8/8
    w12_1_uni_c3.txt: 8/8
    w12_1_uni_cells.txt: 6/9
    w12_4_cs_c1.txt: 22/22
    w12_4_cs_c2.txt: 21/21
    w12_6_cells_c1.txt: 6/18
    w12_6_cells_c2.txt: 5/18
    w12_ho_a.txt: 60/60
    w12_nbf_confirm.txt: 28/28
    w12_nbf_deg.txt: 9/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 33/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 20/20
    w12_seal_pca.txt: 6/6
    w12_seal_sweep.txt: 6/6
- 2026-11-12 20:23:52 jobs: w12_1_uni_cells:RUNNING w12_6_cells_c1:RUNNING w12_6_cells_c2:RUNNING w12_rerun_herogf:RUNNING  | inbox: empty | new tracebacks: none
    w12_1_drop_c1.txt: 8/8
    w12_1_drop_c2.txt: 8/8
    w12_1_drop_c3.txt: 7/7
    w12_1_uni_c1.txt: 9/9
    w12_1_uni_c2.txt: 8/8
    w12_1_uni_c3.txt: 8/8
    w12_1_uni_cells.txt: 6/9
    w12_4_cs_c1.txt: 22/22
    w12_4_cs_c2.txt: 21/21
    w12_6_cells_c1.txt: 13/18
    w12_6_cells_c2.txt: 13/18
    w12_ho_a.txt: 60/60
    w12_nbf_confirm.txt: 28/28
    w12_nbf_deg.txt: 9/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 33/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 20/20
    w12_seal_pca.txt: 6/6
    w12_seal_sweep.txt: 6/6
- 2026-11-12 20:26:53 jobs: w12_1_uni_cells:RUNNING w12_rerun_herogf:RUNNING  | inbox: empty | new tracebacks: none
    w12_1_drop_c1.txt: 8/8
    w12_1_drop_c2.txt: 8/8
    w12_1_drop_c3.txt: 7/7
    w12_1_uni_c1.txt: 9/9
    w12_1_uni_c2.txt: 8/8
    w12_1_uni_c3.txt: 8/8
    w12_1_uni_cells.txt: 8/9
    w12_4_cs_c1.txt: 22/22
    w12_4_cs_c2.txt: 21/21
    w12_6_cells_c1.txt: 18/18
    w12_6_cells_c2.txt: 18/18
    w12_ho_a.txt: 60/60
    w12_nbf_confirm.txt: 28/28
    w12_nbf_deg.txt: 9/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 33/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 20/20
    w12_seal_pca.txt: 6/6
    w12_seal_sweep.txt: 6/6
- 2026-11-12 20:29:51 jobs: w12_rerun_herogf:RUNNING  | inbox: empty | new tracebacks: none
    w12_1_drop_c1.txt: 8/8
    w12_1_drop_c2.txt: 8/8
    w12_1_drop_c3.txt: 7/7
    w12_1_uni_c1.txt: 9/9
    w12_1_uni_c2.txt: 8/8
    w12_1_uni_c3.txt: 8/8
    w12_1_uni_cells.txt: 9/9
    w12_4_cs_c1.txt: 22/22
    w12_4_cs_c2.txt: 21/21
    w12_6_cells_c1.txt: 18/18
    w12_6_cells_c2.txt: 18/18
    w12_ho_a.txt: 60/60
    w12_nbf_confirm.txt: 28/28
    w12_nbf_deg.txt: 9/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 33/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 20/20
    w12_seal_pca.txt: 6/6
    w12_seal_sweep.txt: 6/6
- 2026-11-12 20:32:53 jobs: w12_rerun_herogf:RUNNING  | inbox: empty | new tracebacks: none
    w12_1_drop_c1.txt: 8/8
    w12_1_drop_c2.txt: 8/8
    w12_1_drop_c3.txt: 7/7
    w12_1_uni_c1.txt: 9/9
    w12_1_uni_c2.txt: 8/8
    w12_1_uni_c3.txt: 8/8
    w12_1_uni_cells.txt: 9/9
    w12_4_cs_c1.txt: 22/22
    w12_4_cs_c2.txt: 21/21
    w12_6_cells_c1.txt: 18/18
    w12_6_cells_c2.txt: 18/18
    w12_ho_a.txt: 60/60
    w12_nbf_confirm.txt: 28/28
    w12_nbf_deg.txt: 9/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 33/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 20/20
    w12_seal_pca.txt: 6/6
    w12_seal_sweep.txt: 6/6
- 2026-11-12 20:36:00 jobs: none | inbox: STOP  | new tracebacks: none
    w12_1_drop_c1.txt: 8/8
    w12_1_drop_c2.txt: 8/8
    w12_1_drop_c3.txt: 7/7
    w12_1_uni_c1.txt: 9/9
    w12_1_uni_c2.txt: 8/8
    w12_1_uni_c3.txt: 8/8
    w12_1_uni_cells.txt: 9/9
    w12_4_cs_c1.txt: 22/22
    w12_4_cs_c2.txt: 21/21
    w12_6_cells_c1.txt: 18/18
    w12_6_cells_c2.txt: 18/18
    w12_ho_a.txt: 60/60
    w12_nbf_confirm.txt: 28/28
    w12_nbf_deg.txt: 9/9
    w12_nbf_sweep2_c1.txt: 5/5
    w12_nbf_sweep2_c2.txt: 5/5
    w12_nbf_sweep_c1.txt: 2/2
    w12_nbf_sweep_c2.txt: 2/2
    w12_nbf_sweep_c3.txt: 2/2
    w12_nbf_sweep_c4.txt: 2/2
    w12_rerun_herogf.txt: 33/33
    w12_seal_confirm.txt: 11/11
    w12_seal_ho.txt: 20/20
    w12_seal_pca.txt: 6/6
    w12_seal_sweep.txt: 6/6
- 2026-11-12 20:38:16 STOP seen, no jobs left, final pull done.
