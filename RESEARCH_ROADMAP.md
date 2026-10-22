# Plain-language research roadmap and manuscript status

## What the old editorial note means

You can **write the paper now**, but do not submit a manuscript that says you performed tests you haven't performed. The notebook includes all original 17 training configurations, paired analysis and a *disabled* final test. It does not prove that models work on other years or datasets.

## Already completed (historical)

- Experimental development: EXP_001..EXP_017 on original ULB chronological training/validation split.
- Four-model paired validation analysis EXP_018: 8 fraudulent cases missed by all four at 0.5% alert budget.
- Dataset provenance audit and duplicate decision disclosed.

## Outstanding scientific tasks

1. **Pre-register an independent replication** on IEEE-CIS or another accessible longer-window benchmark. Document data access, features, a train-only transformation policy, chronological split, eligible models, metrics, selection criteria and the final evaluation before training. This is a new dataset and needs its own schema preprocessing; never treat its scores as directly comparable to ULB without discussion.
2. **Duplicate-removal sensitivity**: quantify how deduplication affects evaluation; prevent identical rows from crossing partitions when studying raw duplicate retention. Re-running all models under a different split changes the scientific question; treat this as a separate sensitivity experiment, not as a corrected replacement for prior results.
3. **Statistical robustness**: explore performance across sufficiently long time windows; current two-day dataset is too brief for claims of long-term drift resistance.
4. **One locked ULB held-out evaluation**: freeze the model, threshold, metadata and all decisions, run `final_test_once` once, report it once, and do not keep tuning on the test.
5. **Write results and limitations**: clarify 17 reused-validation comparisons, 1,081 removed exact rows, no stable transaction ID, unknowable PCA feature meanings, and no banking deployment/financial savings evidence.

## Paper writing status

Start now: introduction, related work, methods, audited data summary, descriptive validation analysis.
Wait to finalize: strong generalization claims, independent comparison conclusions, final results table and discussion.
