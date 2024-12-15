#!/usr/bin/env python
"""
Functions to compute EER and min t-DCF.

They are imported from eval_metrics.py in official t-DCF computatio package 
https://www.asvspoof.org/resources/tDCF_python_v2.zip

"""
from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import sys
import numpy as np

__author__ = "ASVspoof consortium"
__copyright__ = "Copyright 2022, ASVspoof consortium"

def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds




def compute_eer(target_scores, nontarget_scores):
    # Define thresholds as 100 evenly spaced values between the min and max of the scores
    thresholds = np.linspace(min(target_scores.min(), nontarget_scores.min()),
                             max(target_scores.max(), nontarget_scores.max()), 100)
    
    far = []
    frr = []

    # Calculate FAR and FRR for each threshold
    for threshold in thresholds:
        # FAR: False Acceptance Rate (percentage of non-target scores >= threshold)
        far.append(np.mean(nontarget_scores >= threshold))
        # FRR: False Rejection Rate (percentage of target scores < threshold)
        frr.append(np.mean(target_scores < threshold))

    # Find the index where FAR and FRR are closest
    idx = np.argmin(np.abs(np.array(far) - np.array(frr)))
    eer_threshold = thresholds[idx]  # Threshold where FAR equals FRR
    eer_value = (far[idx] + frr[idx]) / 2  # EER value 

    # Return both the threshold and EER value
    return eer_threshold, eer_value

if __name__ == "__main__":
    print("Evaluation Metrics Loaded.")
