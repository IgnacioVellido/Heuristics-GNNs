from torch import nn, min, equal
from GCN import GCNConvNoNorm

# ------------------------------------------------------------------------------

class TurnOff(nn.Module):
    def __init__(self):
        super(TurnOff, self).__init__()

    def forward(self, X, mask):
        """mask: boolean tensor of same dimension as X, indicating which
        values should be masked"""
        return X * mask

# ------------------------------------------------------------------------------

class FRGCN(nn.Module):
    """
    Propagates values from some starting nodes to the rest of the graph. Technically
    the same as an algorithm of propagation (set the value of a node based on the
    neighbors), but with the "novelty" that we can use a NN to propagate. Also, it
    can propagate simultaneously through all graph, with multiple starting points
    (must pass the criteria).

    Generalizes to unseen graphs of different structure and size.

    TurnOff
    - Layer to mask a vector, in this case, nodes according to a given criteria

    FRGCN
    - Uses a criteria to define with nodes are propagating its value in each 
    iteration
    - It is called iteratively until all nodes have passed the criteria or for a 
    number of passes "p"
    - Those nodes are "turned-off" in the next iteration, i.e. its edges are removed
    and are not further updated (like dropout, but retaining their old value)
    - Uses a criteria to aggregate values

    - Not considered for now: But technically could accept batch-normalization,
    non-linear activations and so on, but they might affect the evaluation of the 
    criteria
    """

    def __init__(self, in_channels, out_channels, criteria=None, aggr="min"):
        super().__init__()
        
        self.gcn = GCNConvNoNorm(
            in_channels,
            out_channels,
            bias=True,
            aggr=aggr,
            normalize=False,
            add_self_loops=False
        )
        
        self.turnOff = TurnOff()

        self.criteria = criteria if criteria is not None else lambda x: x < 1e4


    def forward(self, x, edge_index, edge_weight=None, passes=1):

        # Propagate values in minimum number of steps order
        for _ in range(passes):            
            # ------------------------------------------------------------------
            # Nodes that won't change (new ones will still forward its value)
            nodes_to_remove = self.criteria(x)

            # start = time.time()
            masked_x = self.turnOff(x, nodes_to_remove)           # a

            # Convolve
            prop = self.gcn(x, edge_index, edge_weight)        # c

            # Nodes that have changed
            masked_prop = self.turnOff(prop, ~nodes_to_remove)    # d

            # x is the combination of both masked values
            x = masked_x + masked_prop

            # EXTREMELY TIME CONSUMING
            # If all nodes/rows pass the criteria, stop looping
            # if sum(nodes_to_remove) == x.size(0):
            #     break


        # ----------------------------------------------------------------------
        # When there are edge weights, shortest path could not be the same as
        # the minimum number of steps

        # Until network converges (no changes) or passes
        for i in range(passes//2):
            # Get min of local neighbors
            prop = self.gcn(x, edge_index, edge_weight)
            
            # Check if min than actual value
            new_x = min(x, prop)

            # If no change in this iteration, break
            if equal(x, new_x):
                break
            
            x = new_x

        return x



