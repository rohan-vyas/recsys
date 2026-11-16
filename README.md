# recsys: two-stage movie recommendation on MovieLens 25M

Implicit-feedback top-K recommendation on MovieLens 25M with a leak-free global temporal split,
full-catalogue evaluation, bootstrap confidence intervals and paired significance tests. Four
model families on the same split and candidate set: popularity, iALS matrix factorisation, a
PyTorch two-tower retrieval model, and a LightGBM LambdaRank second stage over the two-tower's
top 100.

Every number below was produced by the code in this repo in one session on 2026-11-16 and is
copied from `results/metrics.json`. Nothing is rounded up or selected after looking at test.

## Problem

Given a user's rating history up to a point in time, rank the whole movie catalogue so that the
movies the user goes on to rate highly appear near the top. Positives are ratings >= 4.0
(implicit feedback; the rating value is not otherwise used). Items the user already rated before
the cutoff are removed from their ranking.

## Data and split

MovieLens 25M (GroupLens): 25,000,095 ratings, 162,541 users, 59,047 rated movies,
12,452,811 positives (rating >= 4.0). The licence forbids redistribution, so `data/` is gitignored;
`make data` downloads `ml-25m.zip` and checks its MD5 (`6b51fb2759a8657d3bfcbfc42b592ada`).

**Global temporal split** (one wall-clock cutoff for everyone, not per-user leave-last-out, so no
model ever sees an interaction that happened after any evaluation interaction):

| Phase | Fit on | Evaluate on | Used for |
|---|---|---|---|
| val | positives before 2018-03-01 | positives 2018-03-01 to 2018-12-31 | all tuning, early stopping, ranker training |
| test | positives before 2019-01-01 | positives 2019-01-01 to 2019-11-21 (end of data) | scored once with settings fixed on val |

| | val phase | test phase |
|---|---|---|
| Fit positives | 11,310,254 | 11,857,435 |
| Fit users / items | 151,421 / 31,711 | 156,640 / 35,521 |
| Candidate items (full catalogue) | 31,711 | 35,521 |
| Evaluated users | 5,233 | 4,606 |
| Target positives | 104,556 | 97,016 |
| Median targets per user | 9 | 9 |
| Targets dropped: item unseen before cutoff | 8,699 | 8,282 |
| Users dropped: no history before cutoff | 5,219 | 5,702 |

Evaluated users are those with at least one positive before the cutoff and at least one positive
in the window. Candidate items are those with at least one positive before the cutoff (an ID-based
model cannot rank an unseen item); target positives on newer items are dropped and counted above.
Tag features use only tags applied before the phase cutoff. The window dates were chosen from user
counts alone, before any model was trained (an earlier 4-month window left only about 3,300 test
users).

## Models

1. **Popularity (all-time)**: count of positives before the cutoff.
2. **Popularity (last 30 days)**: same, over the 30 days before the cutoff. Added because temporal
   drift makes recent popularity a much stronger baseline than all-time popularity.
3. **iALS** (`implicit` library, Hu, Koren and Volinsky 2008): 128 factors, 15 iterations,
   regularisation 0.05, alpha tuned on val over {1, 5, 20} (5 chosen).
4. **Two-tower** (plain PyTorch, `src/recsys/models/two_tower.py`):
   * User tower: no user-ID embedding. Mean embedding of the last 50 positives, mean of the last 5,
     log history length, then an MLP. So it serves users whose history grew after training.
   * Item tower: item-ID embedding plus genre multi-hot, release-year bucket and a bag of the item's
     top user tags, then an MLP. Both towers are L2-normalised; score is a dot product / 0.05.
   * Training: each positive is predicted from the positives strictly before it. In-batch softmax
     (batch 4096) with logQ sampling-bias correction and masking of in-batch duplicates. AdamW, LR
     2e-3, linear warmup then cosine decay, early stopping on val Recall@50 (patience 2).
   * Tuned on val by short screening runs over logQ on/off and a training-target recency window
     (all history, 3 years, 1 year). Chosen: logQ on, targets from the last 365 days (histories
     still use all earlier positives). Early stopping picked epoch 3; the test refit runs 3 epochs.
5. **Two-tower + LambdaRank** (`src/recsys/ranker.py`): LightGBM LambdaRank reranks the two-tower
   top 100 with 13 features: stage-1 score and rank, all-time / 30-day / 7-day popularity, trend,
   item age, year, genre count, user-genre affinity, history length, days since last activity,
   mean rating. Trained only on the val phase (80/20 user split for early stopping, 49 trees), then
   applied unchanged to each test seed's candidates with features rebuilt from data before
   2019-01-01.

Test-phase training set sizes: iALS fits 11,857,435 positives (156,640 users, 35,521 items). The
two-tower trains on the 643,908 positives from 2018 (each with its full earlier history as input)
for 3 epochs. The ranker trains on 4,162 val users (1,071 held out) x 100 candidates.

## Results (test window, 4,606 users, full-catalogue ranking)

Point estimates with 95% percentile bootstrap CIs over users (2,000 resamples). Two-tower and
ranker rows are mean ± sample std over 3 seeds (0, 1, 2); their CI is seed 0's.

| Model | Recall@10 | Recall@50 | NDCG@10 | MRR | Coverage@10 |
|---|---|---|---|---|---|
| Popularity (all-time) | 0.0289 [0.0262, 0.0317] | 0.0859 [0.0815, 0.0907] | 0.0601 [0.0561, 0.0641] | 0.1378 [0.1299, 0.1457] | 0.0086 |
| Popularity (last 30 days) | 0.0526 [0.0490, 0.0564] | 0.1653 [0.1589, 0.1715] | 0.0924 [0.0877, 0.0973] | 0.1934 [0.1849, 0.2022] | 0.0061 |
| iALS | 0.0582 [0.0547, 0.0619] | 0.1608 [0.1550, 0.1667] | 0.1124 [0.1074, 0.1177] | 0.2336 [0.2243, 0.2437] | 0.0530 |
| Two-tower | 0.0616 ± 0.0019 [0.0568, 0.0644] | 0.1853 ± 0.0025 [0.1789, 0.1918] | 0.1022 ± 0.0039 [0.0954, 0.1048] | 0.2153 ± 0.0057 [0.2022, 0.2202] | 0.0785 ± 0.0037 |
| Two-tower + LambdaRank | 0.0806 ± 0.0004 [0.0765, 0.0849] | 0.2088 ± 0.0035 [0.2031, 0.2164] | 0.1270 ± 0.0013 [0.1222, 0.1329] | 0.2537 ± 0.0033 [0.2440, 0.2637] | 0.0485 ± 0.0038 |

Recall@K = hits in top K / number of target positives; NDCG@K uses binary gains with the ideal
over min(K, targets); MRR uses the rank of the first relevant item in the full ranking (not
truncated); Coverage@10 = distinct items in anyone's top 10 / candidate items.

Per-seed test values (seeds 0, 1, 2):

* Two-tower Recall@50: 0.1853, 0.1878, 0.1828; NDCG@10: 0.0999, 0.1067, 0.0999.
* Ranker Recall@50: 0.2095, 0.2120, 0.2051; NDCG@10: 0.1275, 0.1280, 0.1256.

### Significance

Paired bootstrap on per-user differences (2,000 resamples; p < 0.001 means no resample crossed
zero) and an exact sign test on per-user NDCG@10 wins/losses (ties dropped).

**Two-tower vs iALS is a split decision, consistent across all 3 seeds:**

| Seed | Recall@50 diff [95% CI] | NDCG@10 diff [95% CI] | MRR diff [95% CI] | NDCG@10 sign test W/L, p |
|---|---|---|---|---|
| 0 | +0.0245 [+0.0169, +0.0317], p < 0.001 | -0.0125 [-0.0182, -0.0068], p < 0.001 | -0.0229 [-0.0344, -0.0114], p < 0.001 | 1246/1413, p = 0.0013 |
| 1 | +0.0270 [+0.0195, +0.0348], p < 0.001 | -0.0057 [-0.0113, -0.0001], p = 0.042 | -0.0119 [-0.0232, -0.0005], p = 0.040 | 1299/1367, p = 0.19 |
| 2 | +0.0220 [+0.0145, +0.0291], p < 0.001 | -0.0125 [-0.0180, -0.0071], p < 0.001 | -0.0202 [-0.0320, -0.0083], p < 0.001 | 1235/1416, p = 0.0005 |

The two-tower retrieves more of each user's future positives in the top 50 (Recall@50, the
metric it was early-stopped on), but iALS puts relevant items higher at the very top (NDCG@10,
MRR). Recall@10 differences are +0.0024 (p = 0.37), +0.0056 (p = 0.02), +0.0022 (p = 0.37).

**Ranker vs iALS** (all seeds): better on every metric, p < 0.01 everywhere. Seed 0: Recall@50
+0.0487 [+0.0410, +0.0567], NDCG@10 +0.0150 [+0.0088, +0.0212], MRR +0.0199 [+0.0072, +0.0324];
NDCG@10 sign test 1564/1337, p = 2.7e-5.

**Ranker vs its own two-tower candidates** (all seeds): NDCG@10 +0.0276 / +0.0213 / +0.0257,
Recall@10 +0.0202 / +0.0172 / +0.0198, all p < 0.001.

**Two-tower vs recent popularity** (all seeds): better on every metric, p <= 0.004.

Full tables for every comparison and metric: `results/report.md`.

## Analysis

### Users by history length (test, seed 0, 95% CI)

| Positives before cutoff | Users | Metric | Recent pop. | iALS | Two-tower | + Ranker |
|---|---|---|---|---|---|---|
| 1 to 9 | 90 | Recall@50 | 0.208 [0.173, 0.244] | 0.170 [0.138, 0.203] | 0.202 [0.167, 0.239] | 0.218 [0.182, 0.254] |
| | | NDCG@10 | 0.275 [0.212, 0.337] | 0.198 [0.154, 0.243] | 0.218 [0.170, 0.269] | 0.250 [0.196, 0.308] |
| 10 to 49 | 646 | Recall@50 | 0.212 [0.191, 0.233] | 0.240 [0.220, 0.260] | 0.239 [0.218, 0.260] | 0.247 [0.226, 0.267] |
| | | NDCG@10 | 0.116 [0.100, 0.130] | 0.162 [0.146, 0.180] | 0.117 [0.102, 0.131] | 0.127 [0.113, 0.142] |
| 50 to 199 | 1,976 | Recall@50 | 0.160 [0.150, 0.170] | 0.181 [0.172, 0.190] | 0.182 [0.172, 0.192] | 0.212 [0.201, 0.222] |
| | | NDCG@10 | 0.079 [0.073, 0.086] | 0.113 [0.105, 0.121] | 0.087 [0.080, 0.094] | 0.107 [0.099, 0.114] |
| 200+ | 1,894 | Recall@50 | 0.152 [0.144, 0.161] | 0.112 [0.105, 0.119] | 0.169 [0.161, 0.179] | 0.194 [0.185, 0.204] |
| | | NDCG@10 | 0.090 [0.082, 0.097] | 0.091 [0.083, 0.098] | 0.102 [0.095, 0.109] | 0.143 [0.136, 0.152] |

* Light users (1 to 9 positives, only 90 of them, wide CIs): nothing beats recent popularity on
  NDCG@10; personalisation has little to work with.
* Light-to-mid users (10 to 199): iALS has the best NDCG@10; the ranker is level with or above it on
  Recall@50 (clearly above for 50 to 199).
* Heavy users (200+): iALS degrades badly (Recall@50 0.112, below recent popularity) while the
  two-tower holds up (0.169). The two-tower summarises only the last 50 positives, which tracks a
  long-time user's current taste better than one static factor fitted to decades of ratings. The
  ranker gains most here (NDCG@10 0.143 vs 0.091 for iALS).

### Popularity bias

Head = top 1% of candidate items by positives before 2019-01-01 (355 items). 21.6% of test
positives are on head items.

| Model | Top-10 share from head | Recall@50 on non-head targets |
|---|---|---|
| Popularity (all-time) | 99.9% | 0.0001 |
| Popularity (last 30 days) | 82.2% | 0.075 |
| iALS | 57.7% | 0.098 |
| Two-tower (seeds 0/1/2) | 18.5% / 22.7% / 19.6% | 0.165 / 0.159 / 0.161 |
| Two-tower + ranker (seeds 0/1/2) | 23.0% / 26.3% / 23.3% | 0.177 / 0.169 / 0.172 |

With logQ correction the two-tower's recommendations come from the head about as often as the
targets do, and it finds far more long-tail positives than iALS. The price is precision at the
very top, where popular items are usually the right answer; this is the NDCG@10 / MRR gap
against iALS. The ranker, which leans heavily on popularity features, pulls some head items back
up and wins on both.

## What helped and what did not (all on val unless stated)

* **logQ correction: essential.** Without it the two-tower collapsed (3-epoch screening val Recall@50
  0.038 vs 0.164 with it, all-history training).
* **Training only on recent targets: the biggest single gain.** Screening val Recall@50 by target
  window: all history 0.164, 3 years 0.182, 1 year 0.201 (best epoch). The test window rewards
  current tastes and current popularity.
* **Longer training: did not help.** With the 1-year window val Recall@50 peaked at epoch 3 (0.2020)
  and fell at epochs 4 and 5 (0.1861, 0.1880).
* **Second-stage ranker: helped clearly**, on val holdout users (NDCG@10 0.112 to 0.168) and on
  test (table above). By gain, its most useful features were 30-day popularity, 7-day popularity,
  stage-1 rank and all-time popularity.
* **iALS alpha:** 5 beat 1 and 20 on val Recall@50 (0.1611 vs 0.1410 and 0.1593).
* **Popularity recency:** 30-day popularity roughly doubles all-time popularity on test Recall@50
  (0.165 vs 0.086).

## Limitations

* **The two-tower vs iALS comparison is not apples to apples on recency.** The two-tower was tuned
  over a training-target recency window and iALS was not given an equivalent (for example
  time-decayed confidence weights). On val, the all-history two-tower screening run (Recall@50
  0.164 after 3 short epochs) was roughly level with iALS (0.161). Most of the two-tower's Recall@50
  advantage plausibly comes from that recency choice, not the architecture.
* **Evaluated users are returning users.** A global cutoff evaluates only users active on both sides
  of it: 4,606 test users, mostly heavy raters (median 161 positives before the cutoff, 90 users
  with fewer than 10). Brand-new users (5,702 in the test window) and brand-new items (8,282 target
  positives) are excluded; the item tower's content features could score new items, but this was
  not evaluated.
* **One test window, small tuning budget.** Hyperparameters were screened with short runs and a
  handful of configurations; ALS factors and regularisation were not tuned. Two-tower seed variance
  is reported, but the iALS, popularity and ranker training seeds were fixed (the ranker was trained
  once, on seed 0's val candidates).
* **The ranker's val numbers come from a 20% holdout of val users** that also drove its early
  stopping, so they are slightly optimistic; the test numbers are clean.
* **Offline only.** Ratings >= 4 are a proxy for satisfaction, the data are from 1995 to 2019, and
  offline metrics do not establish online impact.

## Reproduce

```bash
make venv     # Python 3.12 venv, dependencies from public PyPI only
make data     # download ml-25m.zip (MD5 checked) and build data/processed (~11 s)
make tune     # iALS grid + two-tower screening and early-stopped val run
make final    # refit on data before 2019-01-01, score the test window once
make ranker   # LambdaRank stage (trained on val, applied to test)
make report   # bootstrap CIs, paired tests, analysis -> results/metrics.json, results/report.md
make test     # pytest
make lint     # ruff
```

Settings live in `configs/default.yaml`. The ranker has its own entry point
(`python -m recsys.ranker`) because LightGBM and PyTorch each bundle libomp on macOS and loading
both in one process aborts. Model weights and per-user arrays (`models/`, `results/per_user/`)
are gitignored.

Tests: 22 passed (`make test`): temporal split has no leakage (fit strictly before the cutoff,
cold users and unseen items excluded, exclusions cover all pre-cutoff ratings), hand-computed
Recall/NDCG/MRR/coverage fixtures plus a vectorised-vs-reference check, an end-to-end evaluator
check of masking and full-catalogue rank, two-tower history gathering, forward pass, loss
decrease and duplicate masking on tiny data, and the bootstrap/sign-test helpers.

Layout: `src/recsys/{data,split,metrics,evaluate,stats,train_two_tower,ranker,report,cli}.py`,
`src/recsys/models/{popularity,als,two_tower}.py`, tests in `tests/`.

## Hardware and timings

Apple M4 Max, 36 GB, macOS, Python 3.12.12, PyTorch 2.15.0 (two-tower on MPS), implicit 0.7.3
(iALS on CPU), LightGBM 4.7.0. The machine was heavily loaded by other work throughout (load average
800 to 1000, almost no idle CPU), which mostly slowed the CPU-bound iALS; timings are wall-clock as
measured, not best case.

| Step | Time (s) |
|---|---|
| iALS val fits (alpha 5, 1, 20) | 339.4, 172.1, 463.1 |
| Two-tower screening runs (5 configs x 3 short epochs) | 46.0, 37.8, 29.8, 31.2, 16.4 |
| Two-tower val run with early stopping (5 epochs) | 51.7 |
| iALS test refit | 385.6 |
| Two-tower test refit, seeds 0/1/2 (3 epochs each) | 19.9, 18.3, 20.2 |
| LambdaRank fit | 2.2 |
| **Total model fitting** | **1,633.7 (27.2 min)** |

The whole pipeline (data prep, tuning, final runs, ranker, report) ran from 18:07 to 18:36 local
time, with iALS tuning and two-tower tuning running concurrently. Raw log: `results/train_log.txt`.

## Licence

Code: MIT (see `LICENSE`). MovieLens data: GroupLens terms, not redistributed here.
