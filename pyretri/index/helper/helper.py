# -*- coding: utf-8 -*-

import os
import shutil

import numpy
import torch

import numpy as np
import itertools
import vptree

from ..dim_processor import DimProcessorBase
from ..feature_enhancer import EnhanceBase
from ..metric import MetricBase
from ..metric2 import Metric2Base
from ..re_ranker import ReRankerBase
from ..utils import feature_loader
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import minkowski
from scipy.spatial.distance import chebyshev
from scipy.spatial.distance import cosine

import matplotlib.pyplot as plt

from typing import Dict, List


class IndexHelper:
    """
    A helper class to index features.
    """
    def __init__(
            self,
            dim_processors: List[DimProcessorBase],
            feature_enhancer: EnhanceBase,
            metric: MetricBase,
            metric2: Metric2Base,
            re_ranker: ReRankerBase,
    ):
        """
        Args:
            dim_processors (list):
            feature_enhancer (EnhanceBase):
            metric (MetricBase):
            metric2 (Metric2Base):
            re_ranker (ReRankerBase):
        """
        self.dim_procs = dim_processors
        self.feature_enhance = feature_enhancer
        self.metric = metric
        self.metric2 = metric2
        self.re_rank = re_ranker

    def show_topk_retrieved_images(self, single_query_info: Dict, topk: int, gallery_info: List[Dict]) -> None:
        """
        Show the top-k retrieved images of one query.

        Args:
            single_query_info (dict): a dict of single query information.
            topk (int): number of the nearest images to be showed.
            gallery_info (list): a list of gallery set information.
        """
        query_idx = single_query_info["ranked_neighbors_idx"]
        query_topk_idx = query_idx[:topk]

        for idx in query_topk_idx:
            img_path = gallery_info[idx]["path"]
            plt.figure()
            plt.imshow(img_path)
            plt.show()

    def save_topk_retrieved_images(self, save_path: str, single_query_info: Dict, topk: int, gallery_info: List[Dict]) -> None:
        """
        Save the top-k retrieved images of one query.

        Args:
            save_path (str): the path to save the retrieved images.
            single_query_info (dict): a dict of single query information.
            topk (int): number of the nearest images to be saved.
            gallery_info (list): a list of gallery set information.
        """
        query_idx = single_query_info["ranked_neighbors_idx"]
        query_topk_idx = query_idx[:topk]

        for idx in query_topk_idx:
            img_path = gallery_info[idx]["path"]
            shutil.copy(img_path, os.path.join(save_path, str(idx)+'.png'))

    def do_index(self, query_fea: np.ndarray, query_info: List, gallery_fea: np.ndarray) -> (List, np.ndarray, np.ndarray):
        """
        Index the query features.

        Args:
            query_fea (np.ndarray): query set features.
            query_info (list): a list of gallery set information.
            gallery_fea (np.ndarray): gallery set features.

        Returns:
            tuple(List, np.ndarray, np.ndarray): query feature information, query features and gallery features after process.
        """
        for dim_proc in self.dim_procs:
            query_fea, gallery_fea = dim_proc(query_fea), dim_proc(gallery_fea)

        dis2, sorted_index2 = self.metric2(query_fea, gallery_fea)

        #print("# VP-TREE")
        #print("# dis: ", dis2)
        #print("# sorted_index: ", sorted_index2)

        query_fea, gallery_fea = torch.Tensor(query_fea), torch.Tensor(gallery_fea)
        # if torch.cuda.is_available():
        #     query_fea = query_fea.cuda()
        #     gallery_fea = gallery_fea.cuda()

        gallery_fea = self.feature_enhance(gallery_fea)

        dis, sorted_index = self.metric(query_fea, gallery_fea)

        #print("# KNN")
        #print("# dis: ", dis)
        #print("# sorted_index: ", sorted_index)

        #print("---------------")
        #print("# Metric Result")
        #print("sorted_index: ", sorted_index)

        sorted_index = self.re_rank(query_fea, gallery_fea, dis=dis, sorted_index=sorted_index)
        sorted_index2 = self.re_rank(query_fea, gallery_fea, dis=dis2, sorted_index=sorted_index2)
        #print("# Re_rank Result")
        #sorted_index_elements = itertools.islice(sorted_index[0], 10)
        #print("Jaccard sorted_index: ", list(sorted_index_elements))
        for i, info in enumerate(query_info):
            info["ranked_neighbors_idx"] = sorted_index[i].tolist()
            #info["ranked_neighbors_idx2"] = sorted_index2[i]
            info["ranked_neighbors_idx2"] = sorted_index2[i].tolist()

        #print("# query_info: ", query_info)
        return query_info, query_fea, gallery_fea
