import os
import time

from Receptor2Odorant.datasets.m2or_screening_confidence import *
from Receptor2Odorant.datasets.m2or_cv import *
from Receptor2Odorant.datasets.m2or_split import *
from Receptor2Odorant.datasets.m2or_pp import *
from Receptor2Odorant.datasets.m2or_pp_mixtures import *
from Receptor2Odorant.datasets.m2or_pp_data_quality import *
from Receptor2Odorant.datasets.m2or_pp_orphans import *
from Receptor2Odorant.datasets.m2or_pp_discard_test import *


def treat_orphans(working_dir, case):
    if case == 'keep':
        keep_orphans = CVPP_OrphansKeep(data_dir = working_dir)
        keep_orphans.postprocess()
        return keep_orphans.working_dir
    elif case == 'discard_OR':
        discard_orphan_OR = CVPP_OrphansDiscardOR(data_dir = working_dir)
        discard_orphan_OR.postprocess()
        return discard_orphan_OR.working_dir
    elif case == 'discard_molecules':
        discard_orphan_molecules = CVPP_OrphansDiscardMols(data_dir = working_dir)
        discard_orphan_molecules.postprocess()
        return discard_orphan_molecules.working_dir
    elif case == 'discard_both':
        discard_orphan_both = CVPP_OrphansDiscardBoth(data_dir = working_dir)
        discard_orphan_both.postprocess()
        return discard_orphan_both.working_dir
    else:
        raise NotImplementedError('{} is not implemented'.format(case))


def add_imbalance_weights(working_dir):
    addWeights_LogHarmonic = CVPP_addWeights_LogHarmonic(data_dir = working_dir)
    addWeights_LogHarmonic.postprocess()
    class_dist = CVPP_class_dist(data_dir = working_dir)
    class_dist.postprocess()
    addWeights_Class = CVPP_addWeights_Class(data_dir = working_dir,
                                            auxiliary_data_path = {'class_dist' : os.path.join(working_dir, 'class_dist.json')})
    addWeights_Class.postprocess()


def data_quality(working_dir, case):
    """
    """
    # ----- Weighting by confidence:
    if case == 'weight':
        weights_data_quality = CVPP_addWeights_DataQuality(data_dir = working_dir, 
                                                        auxiliary_data_path = {'screening_confidence_probs' : screening_confidence_probs_path,
                                                                                })
        weights_data_quality.postprocess()
        return weights_data_quality.working_dir
    # ----- Discarding primary screening:        
    elif case == 'discard_primary':
        discard_primary = CVPP_Discard_Primary(data_dir = working_dir)
        discard_primary.postprocess()
        return discard_primary.working_dir
    # ----- Naive approach:
    elif case == 'naive':
        naive_data_quality = CVPP_Naive_DataQuality(data_dir = working_dir)
        naive_data_quality.postprocess()
        return naive_data_quality.working_dir
    else:
        raise NotImplementedError('{} is not implemented'.format(case))


def mixtures(working_dir, case):
    """
    """
    # ----- Concat Graph:
    if case == 'concatGraph':
        mix_concatGraph = CVPP_Mixture_ConcatGraph(data_dir = working_dir, 
                                    auxiliary_data_path = {'map_inchikey_to_isomericSMILES' : None,
                                                           'map_inchikey_to_canonicalSMILES' : None,
                                                            })
        mix_concatGraph.postprocess()
        return mix_concatGraph.working_dir
    # ----- Use canonical SMILES for sum of isomers:
    elif case == 'racemic':
        mix_racemic = CVPP_Mixture_Racemic(data_dir = working_dir, 
                                    auxiliary_data_path = {'map_inchikey_to_isomericSMILES' : None,
                                                           'map_inchikey_to_canonicalSMILES' : None,
                                                            })
        mix_racemic.postprocess()
        return mix_racemic.working_dir
    # ----- Mixture: Discard sum of isomers:
    elif case == 'discard':
        mix_discard = CVPP_Mixture_Discard(data_dir = working_dir, 
                                    auxiliary_data_path = {'map_inchikey_to_isomericSMILES' : None,
                                                            })
        mix_discard.postprocess()
        return mix_discard.working_dir
    else:
        raise NotImplementedError('{} is not implemented'.format(case))


def combine_weights_and_size_cut(working_dir, weight_cols = ['class_weight', 'logHarmonic_weight', 'dataQuality_weight']):
    """
    """
    _combine_weights = CVPP_combineWeights(data_dir = working_dir, weight_cols = weight_cols)
    _combine_weights.postprocess()
    _sizecut = CVPP_SizeCut(data_dir = _combine_weights.working_dir, n_node_thresholds = [32, 128], n_edge_thresholds = [64, 256])
    _sizecut.postprocess()
    return None


def deorphanization_split(working_dir, case, sim_dist_dir = None):
    """
    """
    if sim_dist_dir is None:
        sim_dist_dir = '/mnt/Receptor2Odorant/RawData/m2or/_seq_dist_matrices'

    if case == 'OR_random':
        i = 1
        passed = False
        while not passed:
            try:
                ec50_leaveOut_OR = EC50_LeaveOut_OR(data_dir = working_dir,
                                                    seed = int(time.time()),
                                                    split_kwargs = {'relative_valid_ratio' : 0.8, # Number of measurements in validation set: relative_valid_ratio*n_ec50
                                                                    'portion_seqs' : 0.35,
                                                                    'min_n_ligands' : 3,
                                                                    'max_test_size' : 0.4,
                                                                    'min_test_size' : 0.15,
                                                                    })
                ec50_leaveOut_OR.CV_split()
                split_working_dir = ec50_leaveOut_OR.working_dir
                passed = True
            except InadequateTestSetSizeError:
                i += 1
                passed = False
        print('------\nNumber of trials in OR_radnom: {}\n------'.format(i))
        return split_working_dir
    elif case == 'OR_cluster':
        i = 1
        passed = False
        while not passed:
            try:
                ec50_leaveClusterOut_OR = EC50_LeaveClusterOut_OR(data_dir = working_dir,
                                                    seed = int(time.time()),
                                                    split_kwargs = {'relative_valid_ratio' : 0.8, # Number of measurements in validation set: relative_valid_ratio*n_ec50
                                                                'min_n_ligands' : 3,
                                                                'n_cluster_sample' : 9,
                                                                'max_test_size' : 0.4,
                                                                'min_test_size' : 0.15,
                                                                'hdbscan_min_samples' : 1,
                                                                'hdbscan_min_cluster_size' : 15,
                                                                'auxiliary_data_path' : {'seq_dist' : os.path.join(sim_dist_dir, 'sim_dist_matrix.csv'),
                                                                                        'seq_dist_ids' : os.path.join(sim_dist_dir, 'seqs.csv'),
                                                                                        }})
                ec50_leaveClusterOut_OR.CV_split()
                split_working_dir = ec50_leaveClusterOut_OR.working_dir
                passed = True
            except InadequateTestSetSizeError:
                i += 1
                passed = False
        print('------\nNumber of trials in OR_cluster: {}\n------'.format(i))
        return split_working_dir
    elif case == 'Mol_random':
        i = 1
        passed = False
        while not passed:
            try:
                ec50_leaveOut_Mol = EC50_LeaveOut_Mol(data_dir = working_dir,
                                                    seed = int(time.time()),
                                                    split_kwargs = {'relative_valid_ratio' : 0.8, # Number of measurements in validation set: relative_valid_ratio*n_ec50
                                                                    'portion_mols' : 0.25,
                                                                    'min_n_ligands' : 2,
                                                                    'max_test_size' : 0.4,
                                                                    'min_test_size' : 0.15,
                                                                    })
                ec50_leaveOut_Mol.CV_split()
                split_working_dir = ec50_leaveOut_Mol.working_dir
                passed = True
            except InadequateTestSetSizeError:
                i += 1
                passed = False
        print('------\nNumber of trials in Mol_random: {}\n------'.format(i))
        return split_working_dir
    elif case == 'Mol_cluster':
        i = 1
        passed = False
        while not passed:
            try:
                ec50_leaveClusterOut_Mol = EC50_LeaveClusterOut_Mol(data_dir = working_dir,
                                                    seed = int(time.time()),
                                                    split_kwargs = {'relative_valid_ratio' : 0.8, # Number of measurements in validation set: relative_valid_ratio*n_ec50
                                                                'min_n_ligands' : 2,
                                                                'n_cluster_sample' : 6,
                                                                'max_test_size' : 0.4,
                                                                'min_test_size' : 0.15,
                                                                'hdbscan_min_samples' : 1,
                                                                'hdbscan_min_cluster_size' : 10,
                                                                'auxiliary_data_path' : {'map_inchikey_to_canonicalSMILES' : None,
                                                                                        }})
                ec50_leaveClusterOut_Mol.CV_split()
                split_working_dir = ec50_leaveClusterOut_Mol.working_dir
                passed = True
            except InadequateTestSetSizeError:
                i += 0
                passed = False
        print('------\nNumber of trials in Mol_cluster: {}\n------'.format(i))
        return split_working_dir
    else:
        raise NotImplementedError('{} is not implemented'.format(case))


def discrad_occuring_in_test(working_dir, case):
    """
    """
    if case == 'OR_DiscardNeg':
        discardTest_OR_DiscardNeg = CVPP_DiscardTest_OR_DiscardNeg(data_dir = working_dir)
        discardTest_OR_DiscardNeg.postprocess()
        return discardTest_OR_DiscardNeg.working_dir
    elif case == 'OR_KeepNeg':
        discardTest_OR_KeepNeg = CVPP_DiscardTest_OR_KeepNeg(data_dir = working_dir)
        discardTest_OR_KeepNeg.postprocess()
        return discardTest_OR_KeepNeg.working_dir
    elif case == 'Mol_DiscardNeg':
        discardTest_Mol_DiscardNeg = CVPP_DiscardTest_Mol_DiscardNeg(data_dir = working_dir)
        discardTest_Mol_DiscardNeg.postprocess()
        return discardTest_Mol_DiscardNeg.working_dir
    else:
        raise NotImplementedError('{} is not implemented'.format(case))



# -------------------
# main preprocess I.:
# -------------------
def main_postprocess_I(split_working_dir):
    """
    """
    orphan_working_dir = treat_orphans(split_working_dir, case = 'keep')
    # ---------------------------
    # Post-process - add weights:
    # ---------------------------
    add_imbalance_weights(orphan_working_dir)
    # ------------------------------------------------------
    # Post-process - Data quality: weighting + Mixture: ALL:
    # ------------------------------------------------------
    # ----- Data quality:
    quality_working_dir = data_quality(orphan_working_dir, case = 'weight')
    # ----- Mixture: multi-graph approach:
    mix_working_dir = mixtures(quality_working_dir, case = 'concatGraph')
    combine_weights_and_size_cut(mix_working_dir, weight_cols = ['class_weight', 'logHarmonic_weight', 'dataQuality_weight'])
    del mix_working_dir
    # ----- Mixture: Use canonical SMILES for sum of isomers:
    mix_working_dir = mixtures(quality_working_dir, case = 'racemic')
    combine_weights_and_size_cut(mix_working_dir, weight_cols = ['class_weight', 'logHarmonic_weight', 'dataQuality_weight'])
    del mix_working_dir
    # ----- Mixture: Discard sum of isomers:
    mix_working_dir = mixtures(quality_working_dir, case = 'discard')
    combine_weights_and_size_cut(mix_working_dir, weight_cols = ['class_weight', 'logHarmonic_weight', 'dataQuality_weight'])
    del mix_working_dir
    del quality_working_dir


# --------------------
# main preprocess II.:
# --------------------
def main_postprocess_II(split_working_dir):
    """
    """
    orphan_working_dir = treat_orphans(split_working_dir, case = 'keep')
    # ---------------------------
    # Post-process - add weights:
    # ---------------------------
    add_imbalance_weights(orphan_working_dir)
    # ------------------------------------------------------
    # Post-process - Data quality: ALL + Mixture: concat:
    # ------------------------------------------------------
    # ----- Data quality - weight: - already generated in I.
    # quality_working_dir = data_quality(orphan_working_dir, case = 'weight')
    # mix_working_dir = mixtures(quality_working_dir, case = 'concatGraph')
    # combine_weights_and_size_cut(mix_working_dir, weight_cols = ['class_weight', 'logHarmonic_weight', 'dataQuality_weight'])
    # del mix_working_dir
    # ----- Data quality - discard_primary:
    quality_working_dir = data_quality(orphan_working_dir, case = 'discard_primary')
    mix_working_dir = mixtures(quality_working_dir, case = 'concatGraph')
    combine_weights_and_size_cut(mix_working_dir, weight_cols = ['class_weight', 'logHarmonic_weight'])
    del mix_working_dir
    # ----- Data quality - naive:
    quality_working_dir = data_quality(orphan_working_dir, case = 'naive')
    mix_working_dir = mixtures(quality_working_dir, case = 'concatGraph')
    combine_weights_and_size_cut(mix_working_dir, weight_cols = ['class_weight', 'logHarmonic_weight'])
    del mix_working_dir


# ---------------------
# main preprocess III.:
# ---------------------
def main_postprocess_III(split_working_dir):
    """
    """
    def _quality__weight_mix__concat(_orphan_working_dir):
        # ---------------------------
        # Post-process - add weights:
        # ---------------------------
        add_imbalance_weights(_orphan_working_dir)
        # ------------------------------------------------------
        # Post-process - Data quality: weight + Mixture: concat:
        # ------------------------------------------------------
        # ----- Data quality: weight approach:
        quality_working_dir = data_quality(_orphan_working_dir, case = 'weight')
        # ----- Mixture: multi-graph approach:
        mix_working_dir = mixtures(quality_working_dir, case = 'concatGraph')
        combine_weights_and_size_cut(mix_working_dir, weight_cols = ['class_weight', 'logHarmonic_weight', 'dataQuality_weight'])
        del mix_working_dir
        del quality_working_dir

    # ----- Train orphan - keep: - already generated in I and II.
    # orphan_working_dir = treat_orphans(split_working_dir, case = 'discard_molecules')
    # _quality__weight_mix__concat(orphan_working_dir)
    # del orphan_working_dir
    # ----- Train orphan - discard Mols:
    orphan_working_dir = treat_orphans(split_working_dir, case = 'discard_molecules')
    _quality__weight_mix__concat(orphan_working_dir)
    del orphan_working_dir
    # ----- Train orphan - discard ORs:
    orphan_working_dir = treat_orphans(split_working_dir, case = 'discard_OR')
    _quality__weight_mix__concat(orphan_working_dir)
    del orphan_working_dir
    # ----- Train orphan - discard Both:
    orphan_working_dir = treat_orphans(split_working_dir, case = 'discard_both')
    _quality__weight_mix__concat(orphan_working_dir)
    del orphan_working_dir


# -------------------------------------------
# main preprocess IV.-VII. (Deorpahnization):
# -------------------------------------------
def main_postprocess_IV_to_VII(pairs_working_dir, sim_dist_dir = None):
    """
    """
    def _orphans__keep_quality__weight_mix__concat(_split_working_dir):
        orphan_working_dir = treat_orphans(_split_working_dir, case = 'keep')
        # ---------------------------
        # Post-process - add weights:
        # ---------------------------
        add_imbalance_weights(orphan_working_dir)
        # ------------------------------------------------------
        # Post-process - Data quality: weight + Mixture: concat:
        # ------------------------------------------------------
        # ----- Data quality: weight approach:
        quality_working_dir = data_quality(orphan_working_dir, case = 'weight')
        # ----- Mixture: multi-graph approach:
        mix_working_dir = mixtures(quality_working_dir, case = 'concatGraph')
        combine_weights_and_size_cut(mix_working_dir, weight_cols = ['class_weight', 'logHarmonic_weight', 'dataQuality_weight'])
        del orphan_working_dir
        del mix_working_dir
        del quality_working_dir

    # ----- OR random:
    split_working_dir = deorphanization_split(pairs_working_dir, case = 'OR_random', sim_dist_dir = None)
    #  Discard Neg:
    _working_dir = discrad_occuring_in_test(split_working_dir, case = 'OR_DiscardNeg')
    _orphans__keep_quality__weight_mix__concat(_working_dir)
    del _working_dir
    # Keep Neg:
    _working_dir = discrad_occuring_in_test(split_working_dir, case = 'OR_KeepNeg')
    _orphans__keep_quality__weight_mix__concat(_working_dir)
    del _working_dir
    del split_working_dir
    # ----- OR cluster:
    split_working_dir = deorphanization_split(pairs_working_dir, case = 'OR_cluster', sim_dist_dir = None)
    #  Discard Neg:
    _working_dir = discrad_occuring_in_test(split_working_dir, case = 'OR_DiscardNeg')
    _orphans__keep_quality__weight_mix__concat(_working_dir)
    del _working_dir
    # Keep Neg:
    _working_dir = discrad_occuring_in_test(split_working_dir, case = 'OR_KeepNeg')
    _orphans__keep_quality__weight_mix__concat(_working_dir)
    del _working_dir
    del split_working_dir
    # ----- Mol random:
    split_working_dir = deorphanization_split(pairs_working_dir, case = 'Mol_random', sim_dist_dir = None)
    #  Discard Neg:
    _working_dir = discrad_occuring_in_test(split_working_dir, case = 'Mol_DiscardNeg')
    _orphans__keep_quality__weight_mix__concat(_working_dir)
    del _working_dir, split_working_dir
    # ----- Mol cluster:
    split_working_dir = deorphanization_split(pairs_working_dir, case = 'Mol_cluster', sim_dist_dir = None)
    #  Discard Neg:
    _working_dir = discrad_occuring_in_test(split_working_dir, case = 'Mol_DiscardNeg')
    _orphans__keep_quality__weight_mix__concat(_working_dir)
    del _working_dir, split_working_dir





# ----------
# main misc:
# ----------
def misc_postprocess_orphan(split_working_dir):
    """
    """
    def _quality__weight_mix__concat(_orphan_working_dir):
        # ---------------------------
        # Post-process - add weights:
        # ---------------------------
        add_imbalance_weights(_orphan_working_dir)
        # ------------------------------------------------------
        # Post-process - Data quality: weighting + Mixture: concat:
        # ------------------------------------------------------
        # ----- Data quality:
        quality_working_dir = data_quality(_orphan_working_dir, case = 'weight')
        # ----- Mixture: multi-graph approach:
        mix_working_dir = mixtures(quality_working_dir, case = 'concatGraph')
        combine_weights_and_size_cut(mix_working_dir, weight_cols = ['class_weight', 'logHarmonic_weight', 'dataQuality_weight'])
        del mix_working_dir
        del quality_working_dir

    orphan_working_dir = treat_orphans(split_working_dir, case = 'discard_molecules')
    _quality__weight_mix__concat(orphan_working_dir)
    del orphan_working_dir
    orphan_working_dir = treat_orphans(split_working_dir, case = 'discard_OR')
    _quality__weight_mix__concat(orphan_working_dir)
    del orphan_working_dir
    orphan_working_dir = treat_orphans(split_working_dir, case = 'discard_both')
    _quality__weight_mix__concat(orphan_working_dir)
    del orphan_working_dir

# -------------------------
# main misc - leave_out OR:
# -------------------------
def misc_postprocess_leave_out_OR(OR_split_working_dir):
    """
    """
    def _orphans__keep_quality__weight_mix__concat(_discardTest_working_dir):
        _orphan_working_dir = treat_orphans(_discardTest_working_dir, case = 'keep')
        # ---------------------------
        # Post-process - add weights:
        # ---------------------------
        add_imbalance_weights(_orphan_working_dir)
        # ------------------------------------------------------
        # Post-process - Data quality: weighting + Mixture: concat:
        # ------------------------------------------------------
        # ----- Data quality:
        quality_working_dir = data_quality(_orphan_working_dir, case = 'weight')
        # ----- Mixture: multi-graph approach:
        mix_working_dir = mixtures(quality_working_dir, case = 'concatGraph')
        combine_weights_and_size_cut(mix_working_dir, weight_cols = ['class_weight', 'logHarmonic_weight', 'dataQuality_weight'])
        del mix_working_dir
        del quality_working_dir

    discardTest_OR_PosNeg = CVPP_DiscardTest_OR_DiscardNeg(data_dir = OR_split_working_dir)
    discardTest_OR_PosNeg.postprocess()
    _discardTest_working_dir = discardTest_OR_PosNeg.working_dir
    _orphans__keep_quality__weight_mix__concat(_discardTest_working_dir)
    del _discardTest_working_dir

    discardTest_OR_KeepNeg = CVPP_DiscardTest_OR_KeepNeg(data_dir = OR_split_working_dir)
    discardTest_OR_KeepNeg.postprocess()
    _discardTest_working_dir = discardTest_OR_KeepNeg.working_dir
    _orphans__keep_quality__weight_mix__concat(_discardTest_working_dir)
    del _discardTest_working_dir

# ---------------------------
# main misc - leave_out Mols:
# ---------------------------
def misc_postprocess_leave_out_mols(mols_split_working_dir):
    """
    """
    def _orphans__keep_quality__weight_mix__concat(_discardTest_working_dir):
        _orphan_working_dir = treat_orphans(_discardTest_working_dir, case = 'keep')
        # ---------------------------
        # Post-process - add weights:
        # ---------------------------
        add_imbalance_weights(_orphan_working_dir)
        # ------------------------------------------------------
        # Post-process - Data quality: weighting + Mixture: concat:
        # ------------------------------------------------------
        # ----- Data quality:
        quality_working_dir = data_quality(_orphan_working_dir, case = 'weight')
        # ----- Mixture: multi-graph approach:
        mix_working_dir = mixtures(quality_working_dir, case = 'concatGraph')
        combine_weights_and_size_cut(mix_working_dir, weight_cols = ['class_weight', 'logHarmonic_weight', 'dataQuality_weight'])
        del mix_working_dir
        del quality_working_dir

    discardTest_Mol_PosNeg = CVPP_DiscardTest_Mol_DiscardNeg(data_dir = mols_split_working_dir)
    discardTest_Mol_PosNeg.postprocess()
    _discardTest_working_dir = discardTest_Mol_PosNeg.working_dir
    _orphans__keep_quality__weight_mix__concat(_discardTest_working_dir)
    del _discardTest_working_dir

    


def misc_postprocess_train_orphan(split_working_dir):
    """
    """
    def _quality__weight_mix__concat(_orphan_working_dir):
        # ---------------------------
        # Post-process - add weights:
        # ---------------------------
        add_imbalance_weights(_orphan_working_dir)
        # ------------------------------------------------------
        # Post-process - Data quality: weighting + Mixture: concat:
        # ------------------------------------------------------
        # ----- Data quality:
        quality_working_dir = data_quality(_orphan_working_dir, case = 'weight')
        # ----- Mixture: multi-graph approach:
        mix_working_dir = mixtures(quality_working_dir, case = 'concatGraph')
        combine_weights_and_size_cut(mix_working_dir, weight_cols = ['class_weight', 'logHarmonic_weight', 'dataQuality_weight'])
        del mix_working_dir
        del quality_working_dir

    orphan_working_dir = treat_orphans(split_working_dir, case = 'discard_molecules')
    _quality__weight_mix__concat(orphan_working_dir)
    del orphan_working_dir
    orphan_working_dir = treat_orphans(split_working_dir, case = 'discard_OR')
    _quality__weight_mix__concat(orphan_working_dir)
    del orphan_working_dir
    orphan_working_dir = treat_orphans(split_working_dir, case = 'discard_both')
    _quality__weight_mix__concat(orphan_working_dir)
    del orphan_working_dir



def main_postprocess(split_working_dir):
    """
    """
    # ---------------------------
    # Post-process - add weights:
    # ---------------------------
    add_imbalance_weights(split_working_dir)
    # ------------------------------------------------------
    # Post-process - Data quality: weighting + Mixture: ALL:
    # ------------------------------------------------------
    # ----- Data quality:
    quality_working_dir = data_quality(split_working_dir, case = 'weight')
    # ----- Mixture: multi-graph approach:
    mix_working_dir = mixtures(quality_working_dir, case = 'concatGraph')
    combine_weights_and_size_cut(mix_working_dir, weight_cols = ['class_weight', 'logHarmonic_weight', 'dataQuality_weight'])
    del mix_working_dir
    # ----- Mixture: Use canonical SMILES for sum of isomers:
    mix_working_dir = mixtures(quality_working_dir, case = 'racemic')
    combine_weights_and_size_cut(mix_working_dir, weight_cols = ['class_weight', 'logHarmonic_weight', 'dataQuality_weight'])
    del mix_working_dir
    # ----- Mixture: Discard sum of isomers:
    mix_working_dir = mixtures(quality_working_dir, case = 'discard')
    combine_weights_and_size_cut(mix_working_dir, weight_cols = ['class_weight', 'logHarmonic_weight', 'dataQuality_weight'])
    del mix_working_dir
    del quality_working_dir
    # ------------------------------------------------------------
    # Post-process - Data quality: primary + Mixture: multi-graph:
    # ------------------------------------------------------------
    # ----- Data quality: Discard primary
    quality_working_dir = data_quality(split_working_dir, case = 'discard_primary')
    # ----- Mixture: multi-graph approach:
    mix_working_dir = mixtures(quality_working_dir, case = 'concatGraph')
    combine_weights_and_size_cut(mix_working_dir, weight_cols = ['class_weight', 'logHarmonic_weight'])
    del mix_working_dir
    del quality_working_dir
    # ----------------------------------------------------------
    # Post-process - Data quality: naive + Mixture: multi-graph:
    # ----------------------------------------------------------
    # ----- Data quality: Naive
    quality_working_dir = data_quality(split_working_dir, case = 'naive')
    # ----- Mixture: multi-graph approach:
    mix_working_dir = mixtures(quality_working_dir, case = 'concatGraph')
    combine_weights_and_size_cut(mix_working_dir, weight_cols = ['class_weight', 'logHarmonic_weight'])
    del mix_working_dir
    del quality_working_dir





if __name__ == '__main__':
    run_screening_confidence = True # True
    # screening_confidence_probs_path = '/mnt/Receptor2Odorant/Data/m2or/Screening_confidence_probabilities.json'

    run_preprocess = True # True
    # ORpairs_dir = '/mnt/Receptor2Odorant/Data/m2or/mixDiscard_20230428-103542'

    # --------------
    # Prerequisites:
    # --------------
    if run_screening_confidence:
        screening_confidence = ScreeningConfidence(base_working_dir = '/mnt/Receptor2Odorant/Data/m2or',
                                            data_path = {'pairs': '/mnt/Receptor2Odorant/RawData/m2or/export_20220512-135815.csv',
                                                        'uniprot_sequences': '/mnt/Receptor2Odorant/RawData/m2or/uniprot_sequences.csv',
                                                        })
        screening_confidence.main()
        screening_confidence_probs_path = os.path.join(screening_confidence.base_working_dir, 'Screening_confidence_probabilities.json')

    # -----------
    # Preprocess:
    # -----------
    if run_preprocess:
        ORpairs = CV_ORpairs_DiscardMix(base_working_dir = '/mnt/Receptor2Odorant/Data/m2or',
                                    data_path = {'pairs': '/mnt/Receptor2Odorant/RawData/m2or/export_20220512-135815.csv',
                                                'uniprot_sequences': '/mnt/Receptor2Odorant/RawData/m2or/uniprot_sequences.csv',
                                                'map_inchikey_to_isomericSMILES' : None,
                                                'map_inchikey_to_canonicalSMILES' : None,
                                                })
        ORpairs.CV_data()
        print(ORpairs.working_dir)
        ORpairs_dir = ORpairs.working_dir

    # -------------
    # Random Split:
    # -------------
    if True:
        ec50_random = EC50_Random(data_dir = ORpairs_dir,
                                            seed = int(time.time()),
                                            split_kwargs = {'relative_valid_ratio' : 0.8, # Number of measurements in validation set: relative_valid_ratio*n_ec50
                                                            'ec50_test_ratio' : 0.3, # This referes to the percentage of EC50
                                                            }) 
        ec50_random.CV_split()

        main_postprocess_I(ec50_random.working_dir)

    if True:
        main_postprocess_II(ec50_random.working_dir)
        # main_postprocess_II('/mnt/Receptor2Odorant/Data/m2or/mixDiscard_20230428-103542/EC50_random_data/20220608-145239')
        # main_postprocess_II('/mnt/Receptor2Odorant/Data/m2or/mixDiscard_20230428-103542/EC50_random_data/20220608-145758')
        # main_postprocess_II('/mnt/Receptor2Odorant/Data/m2or/mixDiscard_20230428-103542/EC50_random_data/20220608-145957')
        # main_postprocess_II('/mnt/Receptor2Odorant/Data/m2or/mixDiscard_20230428-103542/EC50_random_data/20220608-150147')

    if True:
        main_postprocess_III(ec50_random.working_dir)
        # main_postprocess_III('/mnt/Receptor2Odorant/Data/m2or/mixDiscard_20230428-103542/EC50_random_data/20220608-145239')
        # main_postprocess_III('/mnt/Receptor2Odorant/Data/m2or/mixDiscard_20230428-103542/EC50_random_data/20220608-145758')
        # main_postprocess_III('/mnt/Receptor2Odorant/Data/m2or/mixDiscard_20230428-103542/EC50_random_data/20220608-145957')
        # main_postprocess_III('/mnt/Receptor2Odorant/Data/m2or/mixDiscard_20230428-103542/EC50_random_data/20220608-150147')

    if True:
        main_postprocess_IV_to_VII(ORpairs.working_dir, sim_dist_dir = None)
        # main_postprocess_IV_to_VII('/mnt/Receptor2Odorant/Data/m2or/mixDiscard_20230428-103542', sim_dist_dir = None)
        # main_postprocess_IV_to_VII('/mnt/Receptor2Odorant/Data/m2or/mixDiscard_20230428-103542', sim_dist_dir = None)
        # main_postprocess_IV_to_VII('/mnt/Receptor2Odorant/Data/m2or/mixDiscard_20230428-103542', sim_dist_dir = None)
        # main_postprocess_IV_to_VII('/mnt/Receptor2Odorant/Data/m2or/mixDiscard_20230428-103542', sim_dist_dir = None)


    # ---------------------
    # Misc - test mixtures:
    # ---------------------
    if False:
        ORpairs = CV_ORpairs_OnlyMix(base_working_dir = '/mnt/Receptor2Odorant/Data/m2or',
                                    data_path = {'pairs': '/mnt/Receptor2Odorant/RawData/m2or/export_20220512-135815.csv',
                                                'uniprot_sequences': '/mnt/Receptor2Odorant/RawData/m2or/uniprot_sequences.csv',
                                                'map_inchikey_to_isomericSMILES' : None,
                                                'map_inchikey_to_canonicalSMILES' : None,
                                                })
        ORpairs.CV_data()
        
        # All:
        test_only = TestOnly(data_dir = ORpairs.working_dir)
        test_only.CV_split()
        mixtures(test_only.working_dir, case = 'concatGraph')
        # EC50:
        ec50_test_only = EC50_TestOnly(data_dir = ORpairs.working_dir)
        ec50_test_only.CV_split()
        mixtures(ec50_test_only.working_dir, case = 'concatGraph')
    