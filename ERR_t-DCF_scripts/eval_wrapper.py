#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np
import eval_metrics as em

__author__ = "ASVspoof consortium"
__copyright__ = "Copyright 2022, ASVspoof consortium"

#=================
# Helper functions
#=================

def dump_C012_dict(data_dict, filepath):
    np.array(data_dict).dump(filepath)
    return 

def load_C012_dict(filepath):
    return dict(np.load(filepath, allow_pickle=True).tolist())

#=================
# Wrappers
#=================

def load_asv_metrics(tar_asv, non_asv, spoof_asv):
    """ Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv = load_asv_metrics(
           tar_asv, non_asv, spoof_asv)
    input
    -----
      tar_asv    np.array, score of target speaker trials
      non_asv    np.array, score of non-target speaker trials
      spoof_asv  np.array, score of spoofed trials
    
    output
    ------
      Pfa_asv           scalar, value of ASV false accept rate
      Pmiss_asv         scalar, value of ASV miss rate
      Pmiss_spoof_asv   scalar, 
      P_fa_spoof_asv    scalar
    """
    eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
    
    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv = em.obtain_asv_error_rates(
        tar_asv, non_asv, spoof_asv, asv_threshold)
    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv

def get_eer(bonafide_score_cm, spoof_score_cm):
    """ eer_val, threshold = get_eer(bonafide_score_cm, spoof_score_cm)

    input
    -----
      bonafide_score_cm np.array, score of bonafide data
      spoof_score_cm    np.array, score of spoofed data
    
    output
    ------
      eer_val           scalar, value of EER
      threshold         scalar, value of the threshold corresponding to EER
    """
    threshold, eer = em.compute_eer(bonafide_score_cm, spoof_score_cm)
    return threshold, eer

def get_tDCF_C012(Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model):
    """C0, C1, C2 = get_tDCF_C012(Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model)
    
    compute_tDCF can be factorized into two parts: 
    C012 computation and min t-DCF computation.

    This is for C012 computation.
    
    input
    -----
      Pfa_asv           scalar, value of ASV false accept rate
      Pmiss_asv         scalar, value of ASV miss rate
      Pfa_spoof_asv     scalar, value of ASV spoof false accept rate
      cost_model        dict, cost model parameters
    
    output
    ------
      C0, C1, C2        scalars, coefficients for min tDCF computation
    """
    if cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] != 1:
        sys.exit('ERROR: Prior probabilities must sum to one.')

    C0 = cost_model['Ptar'] * cost_model['Cmiss'] * Pmiss_asv + \
         cost_model['Pnon'] * cost_model['Cfa'] * Pfa_asv

    C1 = cost_model['Ptar'] * cost_model['Cmiss'] - C0
    
    C2 = cost_model['Pspoof'] * cost_model['Cfa_spoof'] * Pfa_spoof_asv
    
    return C0, C1, C2

def get_mintDCF_eer(bonafide_score_cm, spoof_score_cm, C0, C1, C2):
    """ mintDCF, eer = get_mintDCF_eer(bonafide_score_cm, 
                                       spoof_score_cm, C0, C1, C2)
    
    compute_tDCF can be factorized into two parts: 
    C012 computation and min t-DCF computation.

    This is for min t-DCF computation, given the values of C012.
    
    input
    -----
      bonafide_score_cm  np.array, score of bonafide data
      spoof_score_cm     np.array, score of spoofed data
      C0, C1, C2         scalars, coefficients for min tDCF computation
    
    output
    ------
      mintDCF            scalar, value of min tDCF
      eer                scalar, value of EER
    """
    Pmiss_cm, Pfa_cm, CM_thresholds = em.compute_det_curve(
        bonafide_score_cm, spoof_score_cm)

    tDCF = C0 + C1 * Pmiss_cm + C2 * Pfa_cm
    tDCF_norm = tDCF / (C0 + min(C1, C2))
    mintDCF = np.min(tDCF_norm)

    abs_diffs = np.abs(Pmiss_cm - Pfa_cm)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((Pmiss_cm[min_index], Pfa_cm[min_index]))

    return mintDCF, eer

if __name__ == "__main__":
    print("Evaluation Wrapper Loaded.")
