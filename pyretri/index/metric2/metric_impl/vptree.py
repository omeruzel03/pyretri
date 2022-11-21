# -*- coding: utf-8 -*-

import torch
import numpy as np
import itertools

import vptree
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import minkowski
from scipy.spatial.distance import chebyshev
from scipy.spatial.distance import cosine
from scipy.spatial.distance import canberra
from scipy.spatial.distance import braycurtis

from ..metric2_base import Metric2Base
from ...registry import METRICS

from typing import Dict

@METRICS.register
class VPTREE(Metric2Base):
    """
    Similarity measure based on the euclidean distance.

    Hyper-Params:
        top_k (int): top_k nearest neighbors will be output in sorted order. If it is 0, all neighbors will be output.
    """
    default_hyper_params = {
        "top_k": 0,
    }

    def euclidean(self, p1, p2):
        return np.sqrt(np.sum(np.power(p2 - p1, 2)))

    def manhattan(self, p1, p2):
        return cityblock(p1, p2)

    def minkowski(self, p1, p2):
        return minkowski(p1, p2)

    def chebyshev(self, p1, p2):
        return chebyshev(p1, p2)

    def cosine(self, p1, p2):
        return cosine(p1, p2)

    def canberra(self, p1, p2):
        return canberra(p1, p2)

    def braycurtis(self, p1, p2):
        return braycurtis(p1, p2)

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(VPTREE, self).__init__(hps)

    def __call__(self, query_fea: np.ndarray, gallery_fea: np.ndarray) -> (torch.tensor, np.ndarray):

        #print("----query_fea: ", query_fea)
        #print("----gallery_fea: ", gallery_fea)
        #print("----Begin VP-Tree...")

        tree = vptree.VPTree(gallery_fea, self.euclidean)
        #tree = vptree.VPTree(gallery_fea, self.manhattan)
        #tree = vptree.VPTree(gallery_fea, self.minkowski)
        #tree = vptree.VPTree(gallery_fea, self.chebyshev) -- HATA EVALUATION
        #tree = vptree.VPTree(gallery_fea, self.cosine) -- HATA EVALUATION
        #tree = vptree.VPTree(gallery_fea, self.canberra)
        #tree = vptree.VPTree(gallery_fea, self.braycurtis)

        #nn = tree.get_nearest_neighbor(query_fea)
        #print("get_nearest_neighbor: ", nn)

        #range = tree.get_all_in_range(query_fea, 10)
        #print("nrange: ", range)

        dis = []
        sorted_index = []
        for query in query_fea:
            nnn = tree.get_n_nearest_neighbors(query, len(gallery_fea))
            #print("get_n_nearest_neighbors: ", nnn)

            vp_arr_single = []
            for fea in gallery_fea:
                index = 0
                for nfea in nnn:
                    if np.array_equal(nfea[1], fea):
                        vp_arr_single.append(nfea)
                        break
                    index = index + 1

            #print("vp_arr_single: ", vp_arr_single)
            dis_single = []
            for data in vp_arr_single:
                dis_single.append(data[0])

            #print("dis_single: ", dis_single)
            dis.append(dis_single)

            sorted_index_single = np.argsort(dis_single)
            sort_index_elements = itertools.islice(sorted_index_single, len(gallery_fea))
            sorted_index_single_result = list(sort_index_elements)
            #print("sorted_index_single: ", sorted_index_single_result)

            sorted_index.append(sorted_index_single_result)

        #print("----End VP-Tree...")

        dis, sorted_index = torch.Tensor(dis), sorted_index

        return dis, sorted_index
