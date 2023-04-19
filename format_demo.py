# experiment script for MPDL benchmark
import os
import sys
import time
import json
import pickle
import logging
import calendar
import argparse
from typing import List, Dict

if sys.version <= "3.7":
    try:
        from collections.abc import OrderedDict
    except ImportError:
        from collections import OrderedDict
else:
    OrderedDict = dict

import numpy as np
import scipy.sparse as sp
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
import matplotlib as mpl
import matplotlib.pyplot as plt 

import EleNetX.mpdl as mpdl
from EleNetX.utils import obj_attr_cat_to_int
from EleNetX.visualize import plot_ele_nx

import utils
import visualize
from parme import get_parme_model

# Global Gurobi setting
current_GMT = time.gmtime()
timestamp = calendar.timegm(current_GMT)
log_file = "gurobi.{}.log".format(timestamp)


def load_graphs() -> Dict[str, nx.Graph]:
    """
    """
    graphs: OrderedDict[str, nx.Graph] = OrderedDict()
    # load all benchmark graphs
    for neural_net in mpdl.MPDL_BENCHMARKS.keys():
        num_cfg = mpdl.MPDL_BENCHMARKS[neural_net]['num_cfg']
        # enumerate configs
        for i in range(num_cfg):
            gpath = mpdl.compose_graph_path(neural_net, i)
            G = nx.read_gpickle(gpath)

            mpdl.assign_modules(G)

            name = neural_net + ':' + str(i)
            graphs[name] = G
    
    return graphs


def get_module_index(Gs: Dict[str, nx.Graph],
                     key: str = "module") -> Dict:
    # we must consider all graphs to generate module type index
    # in case some graphs won't include all types of modules
    names = Gs.keys(); graphs = Gs.values()
    G = nx.union_all(graphs, rename=[name + ":" for name in names])
    # get module indices
    module_indices = obj_attr_cat_to_int(G.nodes(), key)
    return module_indices


def get_R0(G: nx.Graph,
           module_indices: OrderedDict,
           key: str = "module") -> np.ndarray:
    """
    """
    l = len(G.nodes); r = len(module_indices)
    R0 = np.zeros((r, l), dtype=int)
    for j, v in enumerate(G.nodes):
        k = module_indices[G.nodes[v][key]]
        R0[k, j] = 1
    return R0



def graph_to_nodes(G):
    # networkx.classes.reportviews.NodeDataView
    nodes = G.nodes(data=True)
    # list of tuple(id, attrs)
    nodes = list(nodes)

    # raw node dict by Si-Ze
    nodes = [
        {'id': v_tuple[0], **v_tuple[-1]}
        for v_tuple in nodes
    ]

    # filter attributes
    nodes = [
        {v_key:v_dict[v_key] for v_key in ['id', 'module']}
        for v_dict in nodes
    ]

    return nodes


def graph_to_edges(G):
    # networkx.classes.reportviews.EdgeDataView
    edges = G.edges(data=True)
    # list of tuple(v_1, .., v_n, attrs)
    edges = list(edges)
    
    # raw edge dict by Si-Ze
    edges = [
        {'nodes': e_tuple[:-1], **e_tuple[-1]}
        for e_tuple in edges
    ]

    return(edges)



if __name__ == "__main__":
    # FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    # logging.basicConfig(format=FORMAT, level=logging.INFO)
    np.set_printoptions(edgeitems=30, linewidth=200)

    Gs = load_graphs()

    print(get_module_index(Gs))


    for gid in Gs:
        print(gid)
        G_obj = Gs[gid]
        print(G_obj)

        nodes = graph_to_nodes(G_obj)
        edges = graph_to_edges(G_obj)

        soc = {
            'nodes' : nodes,
            'nets'  : edges
        }
        with open('%s.soc' % gid, 'w') as fp:
            json.dump(soc, fp, indent=2, cls=utils.NpEncoder)