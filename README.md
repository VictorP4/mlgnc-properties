# Multi-Label Node Classification on Graphs

Code for the Bachelor thesis *Property-Driven Comparison of GNNs on Multi-Label Graphs* (Victor Paiu, TU Delft CSE3000). We compare GCN and H2GCN across structural, feature, and label properties of multi-label graphs, using synthetic graphs to vary one property at a time and seven real-world datasets as anchors. A pooled Ridge regression over all 97 trained graphs weighs the properties jointly.

## Project structure

```
generators/   Synthetic dataset + graph construction
  hypersphere.py        MLDataGen hypersphere generator (Tomas 2014)
  sda.py                Social Distance Attachment edge sampler (Zhao 2023 Eq. 2)
  sweep_homophily.py    Binary-search alpha at each b to hit a target homophily
  add_random_edges.py   Uniform random edge addition (Exp 7)
  rewire_clustering.py  Homophily-preserving rewiring (Exp 4 - not used in the paper)
  properties.py         Homophily, CCNS, LI, clustering, summary

models/       GCN and H2GCN training harness
  harness.py            train_one_run(model, data_dir, seed) entry point
  earlystopping.py

metric/       Multi-label metrics (F1, AP, AUC)

scripts/      Paper-narrated experiments + infrastructure
  convert_real_world.py             Real-world datasets -> standard layout
  compute_real_world_properties.py  LI + imbalance metrics on real-world graphs
  summarize_all_graphs.py           Build the Ridge pool (results/all_trained_graphs.csv)
  run_ridge_horse_race.py           Pooled Ridge regression
  run_exp2b_paperbase_homophily_sweep.py  Homophily sweep on the Synthetic1 base
  run_exp7_edge_addition.py               Random-edge addition (structural noise)
  run_exp7_matched_homophily.py           Matched-h clean SDA control for Exp 7
  run_exp11_label_imbalance.py            Label imbalance + unlabeled nodes
  run_exp13_coverage_gaps.py              Gap-filler graphs for the Ridge pool
  auxiliary/                              Experiments whose graphs feed the
                                          Ridge pool but are not narrated

run_batch.py  Sweep (model, dataset, seed) -> append results CSV
data/         Real-world datasets + generated synthetic graphs
results/      Per-experiment training CSVs + aggregate Ridge pool
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/convert_real_world.py          # one-time, downloads Yelp / OGB
python scripts/run_exp2b_paperbase_homophily_sweep.py
python scripts/summarize_all_graphs.py        # rebuild the aggregate
python scripts/run_ridge_horse_race.py
```

Each `run_exp*.py` script generates its synthetic graphs and trains both models over three seeds (`run_batch.py` is the underlying sweeper). `summarize_all_graphs.py` walks the per-experiment CSVs and writes `results/all_trained_graphs.csv`, which the Ridge runner consumes.

Default hyperparameters and metrics follow Zhao et al. 2023 (multi-label, paper Table 7).

## License

MIT - see [LICENSE](LICENSE).
