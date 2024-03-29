# -*- coding: utf-8 -*-

import numpy as np

from ..evaluators_base import EvaluatorBase
from ...registry import EVALUATORS

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

from typing import Dict, List


@EVALUATORS.register
class OverAll(EvaluatorBase):
    """
    A evaluator for mAP and recall computation.

    Hyper-Params
        recall_k (sequence): positions of recalls to be calculated.
    """
    default_hyper_params = {
        "recall_k": [1, 2, 4, 8, 10, 15, 20, 30, 40],
    }

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(OverAll, self).__init__(hps)
        self._hyper_params["recall_k"] = np.sort(self._hyper_params["recall_k"])

    def compute_recall_at_k(self, gt: List[bool], result_dict: Dict) -> None:
        """
        Calculate the recall at each position.

        Args:
            gt (sequence): a list of bool indicating if the result is equal to the label.
            result_dict (dict): a dict of indexing results.
        """
        ks = self._hyper_params["recall_k"]
        gt = gt[:ks[-1]]
        first_tp = np.where(gt)[0]
        if len(first_tp) == 0:
            return
        for k in ks:
            if k >= first_tp[0] + 1:
                result_dict[k] = result_dict[k] + 1

    def __call__(self, query_result: List, gallery_info: List) -> (float, Dict):
        """
        Calculate the mAP and recall for the indexing results.

        Args:
            query_result (list): a list of indexing results.
            gallery_info (list): a list of gallery set information.

        Returns:
            tuple (float, dict): mean average precision and recall for each position.
        """
        aps = list()
        aps2 = list()

        apsAuc = list()
        apsAuc2 = list()

        # For mAP calculation
        pseudo_score = np.arange(0, len(gallery_info))[::-1]

        recall_at_k = dict()
        recall_at_k2 = dict()
        for k in self._hyper_params["recall_k"]:
            recall_at_k[k] = 0
            recall_at_k2[k] = 0

        gallery_label = np.array([gallery_info[idx]["label_idx"] for idx in range(len(gallery_info))])
        for i in range(len(query_result)):
            ranked_idx = query_result[i]["ranked_neighbors_idx"]
            ranked_idx2 = query_result[i]["ranked_neighbors_idx2"]

            gt = (gallery_label[query_result[i]["ranked_neighbors_idx"]] == query_result[i]["label_idx"])
            gt2 = (gallery_label[query_result[i]["ranked_neighbors_idx2"]] == query_result[i]["label_idx"])

            aps.append(average_precision_score(gt, pseudo_score[:len(gt)]))
            aps2.append(average_precision_score(gt2, pseudo_score[:len(gt2)]))

            apsAuc.append(roc_auc_score(gt, pseudo_score[:len(gt)]))
            apsAuc2.append(roc_auc_score(gt2, pseudo_score[:len(gt2)]))

            # deal with 'gallery as query' test
            if gallery_info[ranked_idx[0]]["path"] == query_result[i]["path"]:
                gt.pop(0)

            if gallery_info[ranked_idx2[0]]["path"] == query_result[i]["path"]:
                gt2.pop(0)

            self.compute_recall_at_k(gt, recall_at_k)
            self.compute_recall_at_k(gt2, recall_at_k2)

        mAP = np.mean(aps) * 100
        mAP2 = np.mean(aps2) * 100

        auc = np.mean(apsAuc)
        auc2 = np.mean(apsAuc2)

        for k in recall_at_k:
            recall_at_k[k] = recall_at_k[k] * 100 / len(query_result)

        for k in recall_at_k2:
            recall_at_k2[k] = recall_at_k2[k] * 100 / len(query_result)

        return mAP, mAP2, recall_at_k, recall_at_k2, auc, auc2
        #return mAP, recall_at_k
