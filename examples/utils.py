import torch
import numpy as np
import networkx as nx

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def add_random_edges_weight(graph, attr="weight", min=2, max=10):
    for _, _, attributes in graph.edges(data=True):
        value = np.random.choice(np.arange(min,max))

        attributes[attr] = value


def preprocess_graphs(graphs):
    """Returns graphs preprocessed and the maximum eccentricity found"""
    max = 0

    # Set distance (number of intermediate nodes) for each graph
    for g in graphs:
        add_random_edges_weight(g)

        # Select random goal node
        goal = np.random.choice(g.number_of_nodes(), replace=True)

        # Find shortest path
        paths = nx.shortest_path_length(g, source=None, target=goal,
                                        weight="weight",
                                        method='dijkstra')

        # Set that value to each node
        for k, v in paths.items():
            g.nodes(data=True)[k]['y'] = float(v) #+ 1 # +1 because goal is 1

        # --------------------------------------------------------------------
        # Set value x to inf
        nx.set_node_attributes(g, 10000000000.0, 'x')

        # Set value of x randomly
        # values = np.random.uniform(low=-1.0, high=0.0, 
        #                            size=g.number_of_nodes()) # [-1,0) and will be inversed,
        #                                                      # as we don't want 0 to be included
        # for n, v in enumerate(values):
        #     g.nodes(data=True)[n]['x'] = float(-v)


        # Except the goal, to 0
        g.nodes(data=True)[goal]['x'] = 0.0

        # Remove global attributes
        g.graph = {}

        # --------------------------------------------------------------------
        # Add graph diameter as global attribute
        # g.graph = {'diameter': nx.diameter(g)}

        # Add goal eccentricity as global attribute
        ecc = nx.eccentricity(g, v=goal)
        g.graph = {'eccentricity': ecc}

        # Update max eccentricity
        max = ecc if ecc > max else max

    return max


# Only y because x is 0 (min) in all nodes except in goal where it is one (max)

def normalize_labels(data_list, max=None, norm="min-max"):
    # Join y
    y_list = []
    for data in data_list:
        # Min-max scaling, but we know min is 0
        if max is not None:
            y_list.append(data.y / max)
        else:
            y_list.append(data.y / data.eccentricity)

    return y_list


# ------------------------------------------------------------------------------

# Credit to:
# https://stackoverflow.com/questions/61958360/how-to-create-random-graph-where-each-node-has-at-least-1-edge-using-networkx

from itertools import combinations, groupby
import random

def gnp_random_connected_graph(n, p):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi 
    graph, but enforcing that the resulting graph is conneted
    """
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < p:
                G.add_edge(*e)
    return G

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

from scipy.stats import rankdata

def graphs_similarity(true, pred, decimals=0):
    """Returns percentage of similarity and ranked pred tensor"""
    true = true.cpu().detach() if torch.cuda.is_available() else true.detach()
    pred = pred.cpu().detach() if torch.cuda.is_available() else pred.detach()

    # Round to 2 decimals
    true_rounded = torch.round(true, decimals=decimals)
    pred_rounded = torch.round(pred, decimals=decimals)

    # Rank before compare (same values get same rank, next is immediately after)
    true_ranked = rankdata(true_rounded, method='dense')
    pred_ranked = rankdata(pred_rounded, method='dense')

    # Return percentage of similarity between two ranked vectors (and ranked pred)
    return np.mean(true_ranked == pred_ranked), pred_ranked


def print_parameters(model):
    for name, param in model.named_parameters():
        print(name)
        print(param)
        print(param.shape)
        print("\n")