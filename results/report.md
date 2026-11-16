| Model | Recall@10 | Recall@50 | NDCG@10 | MRR | Coverage@10 |
|---|---|---|---|---|---|
| Popularity (all-time) | 0.0289 [0.0262, 0.0317] | 0.0859 [0.0815, 0.0907] | 0.0601 [0.0561, 0.0641] | 0.1378 [0.1299, 0.1457] | 0.0086 |
| Popularity (last 30 days) | 0.0526 [0.0490, 0.0564] | 0.1653 [0.1589, 0.1715] | 0.0924 [0.0877, 0.0973] | 0.1934 [0.1849, 0.2022] | 0.0061 |
| iALS matrix factorisation | 0.0582 [0.0547, 0.0619] | 0.1608 [0.1550, 0.1667] | 0.1124 [0.1074, 0.1177] | 0.2336 [0.2243, 0.2437] | 0.0530 |
| Two-tower (3 seeds) | 0.0616 ± 0.0019 (seed 0 CI 0.0568, 0.0644) | 0.1853 ± 0.0025 (seed 0 CI 0.1789, 0.1918) | 0.1022 ± 0.0039 (seed 0 CI 0.0954, 0.1048) | 0.2153 ± 0.0057 (seed 0 CI 0.2022, 0.2202) | 0.0785 ± 0.0037 |
| Two-tower + LambdaRank | 0.0806 ± 0.0004 (seed 0 CI 0.0765, 0.0849) | 0.2088 ± 0.0035 (seed 0 CI 0.2031, 0.2164) | 0.1270 ± 0.0013 (seed 0 CI 0.1222, 0.1329) | 0.2537 ± 0.0033 (seed 0 CI 0.2440, 0.2637) | 0.0485 ± 0.0038 |

Significance (paired bootstrap on per-user differences, sign test on NDCG@10):

| Comparison | Metric | Diff | 95% CI | p (bootstrap) | Sign test W/L, p |
|---|---|---|---|---|---|
| two_tower_seed0_vs_als | recall@10 | +0.0024 | [-0.0025, +0.0072] | 0.3710 | 1246/1413, 1.28e-03 |
| two_tower_seed0_vs_als | recall@50 | +0.0245 | [+0.0169, +0.0317] | 0.0005 | 1246/1413, 1.28e-03 |
| two_tower_seed0_vs_als | ndcg@10 | -0.0125 | [-0.0182, -0.0068] | 0.0005 | 1246/1413, 1.28e-03 |
| two_tower_seed0_vs_als | mrr | -0.0229 | [-0.0344, -0.0114] | 0.0005 | 1246/1413, 1.28e-03 |
| two_tower_seed0_vs_popularity_recent | recall@10 | +0.0080 | [+0.0030, +0.0129] | 0.0005 | 1400/1139, 2.41e-07 |
| two_tower_seed0_vs_popularity_recent | recall@50 | +0.0200 | [+0.0128, +0.0274] | 0.0005 | 1400/1139, 2.41e-07 |
| two_tower_seed0_vs_popularity_recent | ndcg@10 | +0.0075 | [+0.0023, +0.0129] | 0.0040 | 1400/1139, 2.41e-07 |
| two_tower_seed0_vs_popularity_recent | mrr | +0.0173 | [+0.0067, +0.0286] | 0.0010 | 1400/1139, 2.41e-07 |
| ranker_seed0_vs_two_tower_seed0 | recall@10 | +0.0202 | [+0.0161, +0.0244] | 0.0005 | 1515/998, 5.17e-25 |
| ranker_seed0_vs_two_tower_seed0 | recall@50 | +0.0242 | [+0.0203, +0.0285] | 0.0005 | 1515/998, 5.17e-25 |
| ranker_seed0_vs_two_tower_seed0 | ndcg@10 | +0.0276 | [+0.0233, +0.0322] | 0.0005 | 1515/998, 5.17e-25 |
| ranker_seed0_vs_two_tower_seed0 | mrr | +0.0428 | [+0.0334, +0.0525] | 0.0005 | 1515/998, 5.17e-25 |
| ranker_seed0_vs_als | recall@10 | +0.0226 | [+0.0173, +0.0278] | 0.0005 | 1564/1337, 2.69e-05 |
| ranker_seed0_vs_als | recall@50 | +0.0487 | [+0.0410, +0.0567] | 0.0005 | 1564/1337, 2.69e-05 |
| ranker_seed0_vs_als | ndcg@10 | +0.0150 | [+0.0088, +0.0212] | 0.0005 | 1564/1337, 2.69e-05 |
| ranker_seed0_vs_als | mrr | +0.0199 | [+0.0072, +0.0324] | 0.0020 | 1564/1337, 2.69e-05 |
| ranker_seed0_vs_popularity_recent | recall@10 | +0.0282 | [+0.0237, +0.0327] | 0.0005 | 1678/981, 5.56e-42 |
| ranker_seed0_vs_popularity_recent | recall@50 | +0.0442 | [+0.0377, +0.0507] | 0.0005 | 1678/981, 5.56e-42 |
| ranker_seed0_vs_popularity_recent | ndcg@10 | +0.0351 | [+0.0300, +0.0405] | 0.0005 | 1678/981, 5.56e-42 |
| ranker_seed0_vs_popularity_recent | mrr | +0.0601 | [+0.0488, +0.0713] | 0.0005 | 1678/981, 5.56e-42 |
| two_tower_seed1_vs_als | recall@10 | +0.0056 | [+0.0007, +0.0105] | 0.0200 | 1299/1367, 1.94e-01 |
| two_tower_seed1_vs_als | recall@50 | +0.0270 | [+0.0195, +0.0348] | 0.0005 | 1299/1367, 1.94e-01 |
| two_tower_seed1_vs_als | ndcg@10 | -0.0057 | [-0.0113, -0.0001] | 0.0420 | 1299/1367, 1.94e-01 |
| two_tower_seed1_vs_als | mrr | -0.0119 | [-0.0232, -0.0005] | 0.0400 | 1299/1367, 1.94e-01 |
| two_tower_seed1_vs_popularity_recent | recall@10 | +0.0113 | [+0.0065, +0.0160] | 0.0005 | 1431/1108, 1.57e-10 |
| two_tower_seed1_vs_popularity_recent | recall@50 | +0.0225 | [+0.0154, +0.0296] | 0.0005 | 1431/1108, 1.57e-10 |
| two_tower_seed1_vs_popularity_recent | ndcg@10 | +0.0143 | [+0.0092, +0.0194] | 0.0005 | 1431/1108, 1.57e-10 |
| two_tower_seed1_vs_popularity_recent | mrr | +0.0283 | [+0.0178, +0.0392] | 0.0005 | 1431/1108, 1.57e-10 |
| ranker_seed1_vs_two_tower_seed1 | recall@10 | +0.0172 | [+0.0130, +0.0213] | 0.0005 | 1488/1066, 6.87e-17 |
| ranker_seed1_vs_two_tower_seed1 | recall@50 | +0.0242 | [+0.0200, +0.0284] | 0.0005 | 1488/1066, 6.87e-17 |
| ranker_seed1_vs_two_tower_seed1 | ndcg@10 | +0.0213 | [+0.0171, +0.0257] | 0.0005 | 1488/1066, 6.87e-17 |
| ranker_seed1_vs_two_tower_seed1 | mrr | +0.0353 | [+0.0252, +0.0453] | 0.0005 | 1488/1066, 6.87e-17 |
| ranker_seed1_vs_als | recall@10 | +0.0228 | [+0.0175, +0.0280] | 0.0005 | 1557/1303, 2.20e-06 |
| ranker_seed1_vs_als | recall@50 | +0.0512 | [+0.0435, +0.0589] | 0.0005 | 1557/1303, 2.20e-06 |
| ranker_seed1_vs_als | ndcg@10 | +0.0156 | [+0.0095, +0.0215] | 0.0005 | 1557/1303, 2.20e-06 |
| ranker_seed1_vs_als | mrr | +0.0234 | [+0.0109, +0.0356] | 0.0005 | 1557/1303, 2.20e-06 |
| ranker_seed1_vs_popularity_recent | recall@10 | +0.0284 | [+0.0239, +0.0331] | 0.0005 | 1694/936, 5.51e-50 |
| ranker_seed1_vs_popularity_recent | recall@50 | +0.0467 | [+0.0403, +0.0532] | 0.0005 | 1694/936, 5.51e-50 |
| ranker_seed1_vs_popularity_recent | ndcg@10 | +0.0356 | [+0.0308, +0.0409] | 0.0005 | 1694/936, 5.51e-50 |
| ranker_seed1_vs_popularity_recent | mrr | +0.0636 | [+0.0527, +0.0750] | 0.0005 | 1694/936, 5.51e-50 |
| two_tower_seed2_vs_als | recall@10 | +0.0022 | [-0.0026, +0.0067] | 0.3650 | 1235/1416, 4.70e-04 |
| two_tower_seed2_vs_als | recall@50 | +0.0220 | [+0.0145, +0.0291] | 0.0005 | 1235/1416, 4.70e-04 |
| two_tower_seed2_vs_als | ndcg@10 | -0.0125 | [-0.0180, -0.0071] | 0.0005 | 1235/1416, 4.70e-04 |
| two_tower_seed2_vs_als | mrr | -0.0202 | [-0.0320, -0.0083] | 0.0005 | 1235/1416, 4.70e-04 |
| two_tower_seed2_vs_popularity_recent | recall@10 | +0.0079 | [+0.0030, +0.0128] | 0.0020 | 1400/1138, 2.16e-07 |
| two_tower_seed2_vs_popularity_recent | recall@50 | +0.0175 | [+0.0101, +0.0245] | 0.0005 | 1400/1138, 2.16e-07 |
| two_tower_seed2_vs_popularity_recent | ndcg@10 | +0.0075 | [+0.0021, +0.0128] | 0.0020 | 1400/1138, 2.16e-07 |
| two_tower_seed2_vs_popularity_recent | mrr | +0.0200 | [+0.0093, +0.0315] | 0.0010 | 1400/1138, 2.16e-07 |
| ranker_seed2_vs_two_tower_seed2 | recall@10 | +0.0198 | [+0.0155, +0.0238] | 0.0005 | 1514/1024, 2.10e-22 |
| ranker_seed2_vs_two_tower_seed2 | recall@50 | +0.0223 | [+0.0183, +0.0264] | 0.0005 | 1514/1024, 2.10e-22 |
| ranker_seed2_vs_two_tower_seed2 | ndcg@10 | +0.0257 | [+0.0216, +0.0300] | 0.0005 | 1514/1024, 2.10e-22 |
| ranker_seed2_vs_two_tower_seed2 | mrr | +0.0369 | [+0.0272, +0.0464] | 0.0005 | 1514/1024, 2.10e-22 |
| ranker_seed2_vs_als | recall@10 | +0.0220 | [+0.0165, +0.0273] | 0.0005 | 1538/1328, 9.40e-05 |
| ranker_seed2_vs_als | recall@50 | +0.0443 | [+0.0365, +0.0520] | 0.0005 | 1538/1328, 9.40e-05 |
| ranker_seed2_vs_als | ndcg@10 | +0.0132 | [+0.0070, +0.0191] | 0.0005 | 1538/1328, 9.40e-05 |
| ranker_seed2_vs_als | mrr | +0.0168 | [+0.0047, +0.0289] | 0.0070 | 1538/1328, 9.40e-05 |
| ranker_seed2_vs_popularity_recent | recall@10 | +0.0276 | [+0.0231, +0.0322] | 0.0005 | 1675/977, 3.30e-42 |
| ranker_seed2_vs_popularity_recent | recall@50 | +0.0398 | [+0.0332, +0.0461] | 0.0005 | 1675/977, 3.30e-42 |
| ranker_seed2_vs_popularity_recent | ndcg@10 | +0.0332 | [+0.0282, +0.0385] | 0.0005 | 1675/977, 3.30e-42 |
| ranker_seed2_vs_popularity_recent | mrr | +0.0570 | [+0.0457, +0.0683] | 0.0005 | 1675/977, 3.30e-42 |
