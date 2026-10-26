"""Motif-only TF -> gene prior for hESC (TFs+500). No ChIP-seq-derived data is used anywhere.
JASPAR 2024 CORE vertebrates non-redundant PFMs (data/external/jaspar2024) mapped to TFs by gene symbol; promoter = TSS -1000/+200 of the Ensembl
canonical transcript of each of the 910 genes (hg38, strand-aware), fetched per gene through Ensembl REST and cached raw in data/external/promoters_hg38/.
Writes data/hESC/TFs+500/motif/motif_max.npy (best log2-odds over the promoter, both strands, max over the TF's motifs) and motif_hits.npy
(hits at p < 1e-4, summed over the TF's motifs); (910, 910) float32, row = TF gene index, col = target gene index, NaN = no motif / no promoter.
Run from the repo root:  python3.12 -P analysis/w9_motif_build.py   (-P: nothing from the cwd is importable; the data files are only read as text/JSON)"""
import json, os, re, time
import numpy as np, pandas as pd, requests

P = 'data/hESC/TFs+500/'
CACHE = 'data/external/promoters_hg38/'
UP, DOWN, PV, BATCH, LBATCH = 1000, 200, 1e-4, 50, 100
os.makedirs(CACHE, exist_ok=True)
os.makedirs(P + 'motif', exist_ok=True)
sess = requests.Session()
sess.headers.update({'Content-Type': 'application/json', 'Accept': 'application/json', 'User-Agent': 'scRegNet-motif-prior (research; garimella.abhishek@research.iiit.ac.in)'})


def post(path, body, fn):
    if os.path.exists(CACHE + fn):
        return json.load(open(CACHE + fn))
    for _ in range(12):
        time.sleep(0.5)  # polite: ~2 requests/s, far below the 15/s limit
        try:
            r = sess.post('https://rest.ensembl.org' + path, data=json.dumps(body), timeout=300)
        except requests.exceptions.RequestException:  # timeout / dropped connection
            time.sleep(10 * (_ + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:  # rate limit or a busy server: wait and retry
            time.sleep(float(r.headers.get('Retry-After', 10 * (_ + 1))))
            continue
        r.raise_for_status()
        json.dump(r.json(), open(CACHE + fn, 'w'))
        return r.json()
    raise RuntimeError(path)


genes = pd.read_csv(P + 'Target.csv', index_col=0).Gene.tolist()
n = len(genes)
# 1. canonical transcript TSS per symbol: gene lookup (gives the canonical transcript id), then transcript lookup (gives coordinates)
info, canon = {}, {}
for k in range(0, n, LBATCH):
    print('lookup', k, flush=True)
    res = post('/lookup/symbol/homo_sapiens', {'symbols': genes[k:k + LBATCH]}, f'sym_{k:04d}.json')
    for s, g in res.items():
        if g and g.get('canonical_transcript'): canon[s.upper()] = g['canonical_transcript'].split('.')[0]
ids = sorted(set(canon.values()))
trs = {}
for k in range(0, len(ids), LBATCH):
    print('transcripts', k, flush=True)
    trs.update(post('/lookup/id', {'ids': ids[k:k + LBATCH]}, f'tr_{k:04d}.json'))
for s, i in canon.items():
    t = trs.get(i)
    if t and t['seq_region_name'] in [str(c) for c in range(1, 23)] + ['X', 'Y']:
        info[s] = (t['seq_region_name'], t['start'], t['end'], t['strand'])
# 2. promoter sequence (gene orientation)
reg = {}
for s, (c, a, b, st) in info.items():
    lo, hi = (a - UP, a + DOWN) if st == 1 else (b - DOWN, b + UP)
    reg[s] = f'{c}:{max(lo, 1)}..{hi}:{st}'
syms = sorted(reg)
seq = {}
for k in range(0, len(syms), BATCH):
    chunk = syms[k:k + BATCH]
    res = post('/sequence/region/human', {'regions': [reg[s] for s in chunk]}, f'seq_{k:04d}.json')
    assert len(res) == len(chunk)
    for s, r in zip(chunk, res):  # the response keeps the request order
        assert r['query'].split(':')[0] == reg[s].split(':')[0]
        seq[s] = r['seq'].upper()
T = UP + DOWN + 1
X = np.full((n, T), 4, dtype=np.int8)  # 0-3 = ACGT, 4 = N / padding
has = np.zeros(n, bool)
for i, g in enumerate(genes):
    if g.upper() in seq:
        v = np.frombuffer(seq[g.upper()].encode(), dtype=np.uint8)
        c = np.full(len(v), 4, np.int8)
        for j, ch in enumerate(b'ACGT'): c[v == ch] = j
        X[i, :len(c)] = c[:T]
        has[i] = True
bg = np.bincount(X[X < 4].ravel(), minlength=4) / (X < 4).sum()
bg = np.array([(bg[0] + bg[3]) / 2, (bg[1] + bg[2]) / 2] * 2)[[0, 1, 1, 0]]  # strand-symmetric background
print(f'promoters: {int(has.sum())}/{n} genes; background ACGT {np.round(bg, 3)}')

# 3. JASPAR PFMs -> log2-odds PWMs (pseudocount 0.8 spread by the background)
mot, cur = [], None
for line in open('data/external/jaspar2024/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt'):
    if line.startswith('>'):
        cur = [line[1:].split('\t')[0], line.split('\t')[1].strip(), []]
        mot.append(cur)
    elif line.strip():
        cur[2].append([float(x) for x in re.findall(r'[\d.]+', line.split('[')[1])])
tf = pd.read_csv(P + 'TF.csv', index_col=0)
idx = {s.upper(): int(i) for s, i in zip(tf.TF, tf['index'])}
assert all(genes[i].upper() == s for s, i in idx.items())
by_tf = {}
for mid, name, cnt in mot:
    for part in re.sub(r'\(var\.\d\)', '', name).upper().split('::'):
        if part in idx: by_tf.setdefault(part, []).append((mid, np.array(cnt).T))


def pwm(c):
    p = (c + 0.8 * bg) / (c.sum(1, keepdims=True) + 0.8)
    return np.log2(p / bg)


def threshold(w, res=100):  # smallest score with background P(score >= thr) <= PV, by dynamic programming on scores rounded to 1/res
    d = {0: 1.0}
    for col in w:
        s = np.round(col * res).astype(int)
        nd = {}
        for a, pa in d.items():
            for b in range(4): nd[a + s[b]] = nd.get(a + s[b], 0) + pa * bg[b]
        d = nd
    tail = 0
    for sc in sorted(d, reverse=True):
        tail += d[sc]
        if tail > PV: return (sc + 1) / res
    return min(d) / res


def scan(w):  # best window score and hit count over both strands, per gene
    out = []
    for m in (w, w[::-1, ::-1]):
        m5 = np.vstack([m.T, np.full((1, len(m)), -1e3)]).T  # N never matches
        L = len(m)
        s = sum(m5[j][X[:, j:T - L + 1 + j]] for j in range(L))
        out.append(s)
    return np.concatenate(out, 1)


motif_max = np.full((n, n), np.nan, np.float32)
motif_hits = np.full((n, n), np.nan, np.float32)
for t, ms in by_tf.items():
    best = np.full(n, -np.inf)
    hits = np.zeros(n)
    for mid, c in ms:
        w = pwm(c)
        s, thr = scan(w), threshold(w)
        best = np.maximum(best, s.max(1))
        hits += (s >= thr).sum(1)
    best[~has] = np.nan; hits[~has] = np.nan
    best[best < -100] = np.nan
    motif_max[idx[t]] = best; motif_hits[idx[t]] = hits
np.save(P + 'motif/motif_max.npy', motif_max); np.save(P + 'motif/motif_hits.npy', motif_hits)

# 4. coverage
lab = set(pd.read_csv(P + 'Train_set.csv', index_col=0).TF) | set(pd.read_csv(P + 'Test_set.csv', index_col=0).TF)
lab_cov = [genes[i] for i in lab if genes[i].upper() in by_tf]
print(f'genes with a promoter: {int(has.sum())}/{n} (unresolved symbols: {n - len(info)}, e.g. {[g for g in genes if g.upper() not in info][:12]})')
print(f'TFs in TF.csv with a JASPAR motif: {len(by_tf)}/{len(tf)} ({sum(len(v) for v in by_tf.values())} motif-TF links, {len({m for v in by_tf.values() for m, _ in v})} distinct matrices)')
print(f'labelled TFs with a motif: {len(lab_cov)}/{len(lab)}: {sorted(lab_cov)}')
print(f'labelled TFs without a motif: {sorted(genes[i] for i in lab if genes[i].upper() not in by_tf)}')
print(f'motif_max finite fraction in covered rows: {np.isfinite(motif_max[[idx[t] for t in by_tf]]).mean():.3f}; mean hits per pair {np.nanmean(motif_hits):.2f}')
