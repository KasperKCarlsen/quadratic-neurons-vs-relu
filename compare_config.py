"""Paired bootstrap confidence interval on the mean accuracy difference
between two finished configs (reads their Done_models/*/seed_results.json)."""
import json, argparse
import numpy as np


def bootstrap_diff_ci(accs_a, accs_b, n_boot=10000, ci=95, seed=0):
    """Paired bootstrap CI on the mean difference (a - b) across seeds.
    accs_a/accs_b: per-seed best test accuracies, paired by position (same seed).
    Returns (mean_diff, lo, hi). A CI excluding 0 indicates a reliable effect."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(accs_a, float), np.asarray(accs_b, float)
    assert len(a) == len(b), "configs must share the same seeds for a paired CI"
    diff = a - b
    boots = np.array([rng.choice(diff, size=len(diff), replace=True).mean()
                      for _ in range(n_boot)])
    half = (100 - ci) / 2
    return float(diff.mean()), float(np.percentile(boots, half)), float(np.percentile(boots, 100 - half))


def load_seed_accs(config_dir):
    with open(f"{config_dir}/seed_results.json") as f:
        d = json.load(f)
    # dict {str(seed): acc} -> ordered by seed for paired alignment
    return {int(s): v for s, v in d["best_acc_per_seed"].items()}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_a", help="e.g. Done_models/Quad_neuron_ReLU0_arc_[5, 5, 5]")
    p.add_argument("config_b", help="the other config dir")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--ci", type=float, default=95)
    args = p.parse_args()

    a, b = load_seed_accs(args.config_a), load_seed_accs(args.config_b)
    common = sorted(set(a) & set(b))
    if len(common) < len(a) or len(common) < len(b):
        print(f"[warn] using {len(common)} shared seeds: {common}")
    aa = [a[s] for s in common]
    bb = [b[s] for s in common]

    mean_a, mean_b = np.mean(aa), np.mean(bb)
    md, lo, hi = bootstrap_diff_ci(aa, bb, args.n_boot, args.ci)
    print(f"A: {mean_a:.3f} ± {np.std(aa, ddof=1):.3f}   ({args.config_a})")
    print(f"B: {mean_b:.3f} ± {np.std(bb, ddof=1):.3f}   ({args.config_b})")
    print(f"mean diff (A-B): {md:+.3f}   {args.ci:.0f}% CI [{lo:+.3f}, {hi:+.3f}]")
    print("  -> reliable difference" if (lo > 0 or hi < 0)
          else "  -> CI straddles 0: not clearly distinguishable")