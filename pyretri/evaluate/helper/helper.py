# -*- coding: utf-8 -*-

import torch
from ..evaluator import EvaluatorBase

from typing import Dict, List

class EvaluateHelper:
    """
    A helper class to evaluate query results.
    """
    def __init__(self, evaluator: EvaluatorBase):
        """
        Args:
            evaluator: a evaluator class.
        """
        self.evaluator = evaluator
        self.recall_k = evaluator.default_hyper_params["recall_k"]

    def show_results(self, mAP: float, recall_at_k: Dict, auc: float) -> None:
        """
        Show the evaluate results.

        Args:
            mAP (float): mean average precision.
            recall_at_k (Dict): recall at the k position.
        """
        #repr_str = "Precision: {:.1f}\n".format(mAP)
        repr_str = "Precision: {}\n".format(mAP)

        repr_str += "Area Under Curve (AUC): {}\n".format(auc * 100)

        f1 = 2 * (mAP * recall_at_k[1]) / (mAP + recall_at_k[1])
        repr_str += "F1 Score: {}\n".format(f1)

        for k in self.recall_k:
            #repr_str += "Recall@{}: {:.1f}\t".format(k, recall_at_k[k])
            repr_str += "Recall@{}: {}\n".format(k, recall_at_k[k])

        print('--------------- Retrieval Evaluation ------------')
        print(repr_str)

    def do_eval(self, query_result_info: List, gallery_info: List) -> (float, Dict):
        """
        Get the evaluate results.

        Args:
            query_result_info (list): a list of indexing results.
            gallery_info (list): a list of gallery set information.

        Returns:
            tuple (float, Dict): mean average precision and recall for each position.
        """
        mAP, mAP2, recall_at_k, recall_at_k2, auc, auc2 = self.evaluator(query_result_info, gallery_info)
        #mAP, recall_at_k = self.evaluator(query_result_info, gallery_info)

        return mAP, mAP2, recall_at_k, recall_at_k2, auc, auc2
        #return mAP, recall_at_k
