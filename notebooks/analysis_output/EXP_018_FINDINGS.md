# EXP_018: Paired failure analysis (VALIDATION ONLY)

Do not interpret these results as independent confirmation: all models were selected/compared using the same validation period.

## Verified model performance

| experiment   |     n |   fraud |   average_precision |   roc_auc |   alerts_0.001 |   fraud_caught_0.001 |   fraud_recall_0.001 |   false_alarms_0.001 |   alerts_0.005 |   fraud_caught_0.005 |   fraud_recall_0.005 |   false_alarms_0.005 |   alerts_0.01 |   fraud_caught_0.01 |   fraud_recall_0.01 |   false_alarms_0.01 |
|:-------------|------:|--------:|--------------------:|----------:|---------------:|---------------------:|---------------------:|---------------------:|---------------:|---------------------:|---------------------:|---------------------:|--------------:|--------------------:|--------------------:|--------------------:|
| EXP_012      | 56746 |      57 |            0.786158 |  0.980974 |             57 |                   43 |             0.754386 |                   14 |            284 |                   47 |             0.824561 |                  237 |           568 |                  48 |            0.842105 |                 520 |
| EXP_011      | 56746 |      57 |            0.781462 |  0.980041 |             57 |                   43 |             0.754386 |                   14 |            284 |                   49 |             0.859649 |                  235 |           568 |                  49 |            0.859649 |                 519 |
| EXP_016      | 56746 |      57 |            0.778371 |  0.968239 |             57 |                   42 |             0.736842 |                   15 |            284 |                   47 |             0.824561 |                  237 |           568 |                  48 |            0.842105 |                 520 |
| EXP_008      | 56746 |      57 |            0.773451 |  0.982885 |             57 |                   41 |             0.719298 |                   16 |            284 |                   47 |             0.824561 |                  237 |           568 |                  48 |            0.842105 |                 520 |

## Pairwise top-0.5% alert overlap

| model_a   | model_b   |   common_alerts |   a_only_alerts |   b_only_alerts |   frauds_both |   frauds_a_only |   frauds_b_only |   frauds_neither |
|:----------|:----------|----------------:|----------------:|----------------:|--------------:|----------------:|----------------:|-----------------:|
| EXP_008   | EXP_011   |             203 |              81 |              81 |            47 |               0 |               2 |                8 |
| EXP_008   | EXP_012   |             197 |              87 |              87 |            46 |               1 |               1 |                9 |
| EXP_008   | EXP_016   |             181 |             103 |             103 |            46 |               1 |               1 |                9 |
| EXP_011   | EXP_012   |             171 |             113 |             113 |            47 |               2 |               0 |                8 |
| EXP_011   | EXP_016   |             166 |             118 |             118 |            47 |               2 |               0 |                8 |
| EXP_012   | EXP_016   |             174 |             110 |             110 |            47 |               0 |               0 |               10 |

At 0.5% capacity (284 alerts), **8 frauds** were missed by all four models.

## Eight missed frauds

|   validation_row |   time |   EXP_008_rank |   EXP_011_rank |   EXP_012_rank |   EXP_016_rank |   Amount |
|-----------------:|-------:|---------------:|---------------:|---------------:|---------------:|---------:|
|             5148 | 122608 |           6418 |          15020 |           8654 |          18728 |  2125.87 |
|            20139 | 129095 |           6644 |           1117 |           4053 |           7618 |   720.8  |
|            20332 | 129186 |           3241 |           6989 |           7189 |          18993 |     7.18 |
|            21746 | 129808 |           2781 |           3241 |           1985 |           4379 |   276.17 |
|            33114 | 135102 |          10291 |          11319 |          10577 |          13801 |   108.51 |
|            43782 | 139816 |           7204 |           4736 |           4149 |          19080 |     0.69 |
|            48014 | 141565 |          12544 |          11702 |          18284 |           2104 |     4.49 |
|            51376 | 142961 |           5130 |          10198 |           2470 |          15052 |     0    |

## Two within-validation temporal segments

| period      | experiment   |     n |   fraud |   fraud_prevalence |   average_precision |
|:------------|:-------------|------:|--------:|-------------------:|--------------------:|
| first_half  | EXP_008      | 28373 |      24 |        0.000845875 |            0.677957 |
| first_half  | EXP_011      | 28373 |      24 |        0.000845875 |            0.683308 |
| first_half  | EXP_012      | 28373 |      24 |        0.000845875 |            0.721286 |
| first_half  | EXP_016      | 28373 |      24 |        0.000845875 |            0.701187 |
| second_half | EXP_008      | 28373 |      33 |        0.00116308  |            0.836464 |
| second_half | EXP_011      | 28373 |      33 |        0.00116308  |            0.832294 |
| second_half | EXP_012      | 28373 |      33 |        0.00116308  |            0.830057 |
| second_half | EXP_016      | 28373 |      33 |        0.00116308  |            0.835714 |

## Exploratory stratified paired bootstrap

| comparison                       | metric                 |   observed_difference |   bootstrap_median |   percentile_2_5 |   percentile_97_5 |   bootstrap_repetitions |     seed |
|:---------------------------------|:-----------------------|----------------------:|-------------------:|-----------------:|------------------:|------------------------:|---------:|
| CatBoost minus weighted LightGBM | average_precision      |            0.00469596 |         0.00242332 |       -0.016176  |         0.0319043 |                     500 | 20261022 |
| CatBoost minus weighted LightGBM | fraud_recall_at_0.5pct |           -0.0350877  |        -0.0350877  |       -0.0877193 |         0         |                     500 | 20261022 |

These percentile intervals are descriptive only and do not account for within-time dependence or the repeated model selection on validation.

## Interpretation limits

- Predictions do not contain a stable originating row ID; consistent ordered time and class sequences support alignment but do not cryptographically prove it.
- The original dataset's PCA-transformed V1–V28 lack feature semantics, so do not invent behavioural explanations for individual missed frauds.
- This work reads no final TEST predictions or TEST evaluation labels. Original CSV enrichment reconstructs only the validation partition.
- The same review capacity forces false alarms = capacity - true alerts, so it is not independent evidence of financial savings.
- For real temporal replication and delay-aware monitoring, a longer dataset with appropriate time features is required.
