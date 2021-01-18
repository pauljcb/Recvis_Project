import argparse
import collections
import math
import multiprocessing as mp
import os
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from loguru import logger
from tqdm import tqdm

from level_snippet_dataset import LevelSnippetDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="mario")
    parser.add_argument("--tags", nargs="*", type=str, default=["similarity"])
    parser.add_argument("--job-type", type=str, default="eval")
    parser.add_argument("--level-dir", type=str,
                        metavar="DIR", default="input/ascii_levels")
    parser.add_argument("--run-dir", type=str, metavar="DIR")
    parser.add_argument("--slice-width", type=int, default=16)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--pattern-sizes", nargs="+",
                        type=int, default=[4, 3, 2])
    hparams = parser.parse_args()
    return hparams


def pattern_key(level_slice):
    """
    Computes a hashable key from a level slice.
    """
    key = ""
    for line in level_slice:
        for token in line:
            key += str(token)
    return key


def get_pattern_counts(level, pattern_size):
    """
    Collects counts from all patterns in the level of the given size.
    """
    pattern_counts = collections.defaultdict(int)
    for up in range(level.shape[0] - pattern_size + 1):
        for left in range(level.shape[1] - pattern_size + 1):
            down = up + pattern_size
            right = left + pattern_size
            level_slice = level[up:down, left:right]
            pattern_counts[pattern_key(level_slice)] += 1
    return pattern_counts


def compute_pattern_counts(dataset, pattern_size):
    """
    Compute pattern counts in parallel from a given dataset.
    """
    levels = [level.argmax(dim=0).numpy() for level in dataset.levels]
    with mp.Pool() as pool:
        counts_per_level = pool.map(
            partial(get_pattern_counts, pattern_size=pattern_size), levels,
        )
    pattern_counts = collections.defaultdict(int)
    for counts in counts_per_level:
        for pattern, count in counts.items():
            pattern_counts[pattern] += count
    return pattern_counts


def compute_prob(pattern_count, num_patterns, epsilon=1e-7):
    """
    Compute probability of a pattern.
    """
    return (pattern_count + epsilon) / ((num_patterns + epsilon) * (1 + epsilon))


def compute_kl_divergence(hparams):
    print("Computing KL-Divergence for generated levels")
    dataset = LevelSnippetDataset(
        level_dir='input',
        slice_width=hparams.slice_width
    )
    print("Original level : ", dataset.level_names)
    test_dataset = LevelSnippetDataset(
        level_dir=hparams.level_dir,
        slice_width=hparams.slice_width
    )
    print("Generated levels : ",test_dataset.level_names)
    kl_divergences = []
    for pattern_size in hparams.pattern_sizes:
        print("Computing original pattern counts...")
        pattern_counts = compute_pattern_counts(dataset, pattern_size)
        print("Computing test pattern counts...")
        test_pattern_counts = compute_pattern_counts(test_dataset, pattern_size)

        num_patterns = sum(pattern_counts.values())
        num_test_patterns = sum(test_pattern_counts.values())

        kl_divergence = 0
        for pattern, count in tqdm(pattern_counts.items()):
            prob_p = compute_prob(count, num_patterns)
            prob_q = compute_prob(
                test_pattern_counts[pattern], num_test_patterns)
            kl_divergence += hparams.weight * prob_p * math.log(prob_p / prob_q) + (
                1 - hparams.weight
            ) * prob_q * math.log(prob_q / prob_p)

        kl_divergences.append(kl_divergence)
        print("KL-Divergence @ ", pattern_size,"x",pattern_size, " :", round(kl_divergence, 2))
    mean_kl_divergence = np.mean(kl_divergences)
    print("Average KL-Divergence: ", round(mean_kl_divergence, 2))
    return mean_kl_divergence


if __name__ == "__main__":
    hparams = parse_args()
    compute_kl_divergence(hparams)
