#!/usr/bin/env python
"""
========
Notes 
========
This is the API to compute EERs and min t-DCFs on ASVspoof2021 evaluation data.

Requirements:
 numpy
 pandas
 matplotlib 

Example usage:
1. Download key and meta label
   bash download.sh
   
   The downloaded directory contains key and meta label files

   keys
   |- LA
   |  |- CM 
   |  |   |- trial_metadata.txt
   |  |   |- LFCC-GMM
   |  |       |- score.txt
   |  |- ASV
   |      |- trial_metadata.txt
   |      |- ASVtorch_kaldi
   |          |- score.txt
   |- DF ...
   |- PA ...

   trial_metadata.txt contains the key and meta data
   score.txt is the score file

2. Compute EER and min t-DCF 

   Let's use LA evaluation subset as example.
   Assume meta labels are in ./keys (default folder)

   Case 1 (most common case)
   Compute results using pre-computed C012 cofficients

   python main.py --cm-score-file score.txt --track LA --subset eval
   
   Case 2
   Recompute C012 using official ASV scores, save it to ./LA-c012.npy,
   and use the new C012 to compute EER and min tDCFs
   
   python main.py --cm-score-file score.txt --track LA --subset eval 
                  --recompute-c012 --c012-path ./LA-c012.npy

   Case 3
   Recompute C012 using my own ASV scores, save it to ./LA-c012.npy
   and use the new C012 to compute EER and min tDCFs
   
   python main.py --cm-score-file score.txt --track LA --subset eval 
                  --recompute-c012 --c012-path ./LA-c012.npy 
                  --asv-score-file ./asv-score.txt

   Case 4
   Compute min tDCF using my own C012 coeffs ./LA-c012.npy

   python main.py --cm-score-file score.txt --track LA --subset eval 
                  --c012-path ./LA-c012.npy
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
import eval_wrapper as ew
import eval_metrics as em

__author__ = "ASVspoof consortium"
__copyright__ = "Copyright 2022, ASVspoof consortium"

#====================
# Main Evaluation Script
#====================

def load_scores(score_file):
    """ Load scores from the score file.

    input
    -----
      score_file : str, path to the score file

    output
    ------
      scores     : dict, with keys ['File_name', 'score']
    """
    file_names = []
    scores = []
    with open(score_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                file_name, score = parts
                file_names.append(file_name)
                scores.append(float(score))

    return {
        'File_name': np.array(file_names),
        'score': np.array(scores)
    }

def load_protocol(protocol_file):
    """ Load true labels from the protocol file.

    input
    -----
      protocol_file : str, path to the protocol file

    output
    ------
      protocol      : dict, mapping File_name to true label
    """
    protocol = {}
    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:  # Ensure there are enough columns
                file_name = parts[1]  # File_name is in the second column
                label = parts[3]  # Label ('bonafide' or 'spoof') is in the fourth column
                protocol[file_name] = label

    return protocol

def split_scores(score_data, protocol):
    """ Split scores into bonafide and spoofed based on protocol labels.

    input
    -----
      score_data : dict, with keys ['File_name', 'score']
      protocol   : dict, mapping File_name to true label

    output
    ------
      bonafide_scores : np.array, scores of bonafide trials
      spoof_scores    : np.array, scores of spoofed trials
    """
    bonafide_scores = []
    spoof_scores = []

    for file_name, score in zip(score_data['File_name'], score_data['score']):
        label = protocol.get(file_name, None)
        if label == 'bonafide':
            bonafide_scores.append(score)
        elif label == 'spoof':
            spoof_scores.append(score)


    return np.array(bonafide_scores), np.array(spoof_scores)

def debug_classification(score_file, protocol_file):
    """
    Debug function to evaluate classification performance based on the EER threshold.
    
    Parameters:
    - score_file (str): Path to the score file.
    - protocol_file (str): Path to the protocol file.
    
    Output:
    - Prints the number of properly and improperly classified samples, accuracy, and EER threshold.
    """
    # Load scores and protocol
    scores = load_scores(score_file)
    protocol = load_protocol(protocol_file)

    # Split scores into target and non-target groups
    bonafide_scores, spoof_scores = split_scores(scores, protocol)

    # Compute the EER threshold
    eer_threshold, eer = ew.get_eer(bonafide_scores, spoof_scores)
    print(f"\nComputed EER Threshold: {eer_threshold:.4f}")

    # Generate true labels and predictions
    y_true = []
    y_pred = []

    for file_name, score in zip(scores['File_name'], scores['score']):
        true_label = protocol.get(file_name, None)
        if true_label is not None:
            y_true.append(true_label)
            # Classify based on EER threshold
            predicted_label = 'bonafide' if score >= eer_threshold else 'spoof'
            y_pred.append(predicted_label)

    # Validate that y_true and y_pred have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("Mismatch between true labels and predictions.")

    # Calculate proper and improper classifications
    correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    incorrect = len(y_true) - correct

    # Calculate accuracy
    accuracy = correct / len(y_true) if len(y_true) > 0 else 0

    # Print results
    print("\nDebugging Classification Results:")
    print(f"Total samples: {len(y_true)}")
    print(f"Properly classified: {correct}")
    print(f"Improperly classified: {incorrect}")
    print(f"Accuracy: {accuracy:.2%}")


def evaluate(score_file, protocol_file, cost_model):
    """ Evaluate EER and min t-DCF for a given score file.

    input
    -----
      score_file   : str, path to the score file
      protocol_file: str, path to the protocol file
      cost_model   : dict, cost parameters for t-DCF

    output
    ------
      results      : dict, containing EER and min t-DCF
    """
    print(f"Evaluating scores from {score_file} using protocol {protocol_file}")

    # Load scores and protocol
    scores = load_scores(score_file)
    protocol = load_protocol(protocol_file)

    # Split into bonafide and spoof scores
    bonafide_scores, spoof_scores = split_scores(scores, protocol)

    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        raise ValueError("Error: Bonafide or spoof scores are empty. Check the protocol and score file alignment.")

    # Compute EER
    threshold, eer = ew.get_eer(bonafide_scores, spoof_scores)
    print(f"EER: {eer:.4f} Threshold: {threshold:.4f}")

    # Compute t-DCF coefficients
    C0, C1, C2 = ew.get_tDCF_C012(0.01, 0.05, 0.5, cost_model)

    # Compute min t-DCF
    mintDCF, _ = ew.get_mintDCF_eer(bonafide_scores, spoof_scores, C0, C1, C2)
    print(f"Min t-DCF: {mintDCF:.4f}")

    eer_percentage = eer * 100


    return {
        'EER': eer_percentage,
        'min_tDCF': mintDCF
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py <score_file> <protocol_file>")
        sys.exit(1)

    score_file = sys.argv[1]
    protocol_file = sys.argv[2]

    # Define cost model parameters
    cost_model = {
    'Ptar': 0.20,  # Target speaker (bona fide trials)
    'Pnon': 0.00,  # Non-target speaker (zero-effort impostor)
    'Pspoof': 0.8,  # Spoofing attack
    'Cmiss': 1,
    'Cfa': 10,
    'Cfa_spoof': 10
    }
    results = evaluate(score_file, protocol_file, cost_model)
    debug_classification(score_file, protocol_file)
    print("\nEvaluation Results:")
    print(f"  EER: {results['EER']:.4f}")
    print(f"  Min t-DCF: {results['min_tDCF']:.4f}")