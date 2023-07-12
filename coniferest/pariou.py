# pariou.py is a collection of methods revolving around feature importance.
# it takes its name after the Pariou montain, climbed by the SNAD team in the Clermont workshop

# Python imports
import numpy as np

# local imports
from .pineforest import PineForest
from .utils import average_path_length

def n_expected_visits(data,n_subsamples=1024,n_trees=1):
    """returns the expected number each data element is visited during the training time
       (under the hypothesis of perfectly balanced trees)

       data: array [n_elements,n_features]

       n_subsamples: the n_subsamples parameter when building trees
           (assuming tree depth is log2(n_subsample)
           default: 1014
       n_trees: the number of trees of the forest
           default: 1"""

    return n_trees*n_subsamples*np.log2(n_subsamples)/np.prod(np.shape(data))

#lookup table for speed-up (specs not done...)
def build_apl_table(n_subsamples):
    """ builds a lookup table of average_path_length up to n_subsample
    returns as an array"""

    return average_path_length(np.arange(n_subsamples+1,dtype=float))

def calc_node_depth(tree,selection=True):
    """traverses the tree wstructure and records node depth for selected features
    tree: a sklearn tree
    selection: if set to True, all nodes are computed.
        else performs only the computation for selected nodes

    returns the depth of each active node, but keeping the total tree structure
        nodes which are not visited are set to 0
    """


    if np.isscalar(selection):
        selection=np.ones_like(tree.children_left,dtype=bool)*selection

    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if selection[node_id] and is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
    return node_depth

# the smallest non-0 number (useful to avoid
epsilon=1./np.finfo(np.float64).max*np.finfo(np.float64).resolution

def comp_signature(pineforest,data,anomalies_indices,anomalies_weight=1.,full_output=False):
    """Computes the signature for the features of data elements indexed by anomaly_indices (theses may be also non-anomalies)
        signature is defined as the average depth each feature is contributing to change
        For a given data element, the average of ( signature * weight ) + initial average depth
            is equal to the average depth within the tree.

    pineforest : a sklearn trained forest object
    data : numpy array of dimension [n_elements, n_features].
    anomalies_indices : list of indices within data n_elements.
        data elements not indexed will not be used in the computation.
    anomalies_weight : a specific weight applied to each anomaly (for instance, -1. for normal events)
        default: 1.
    full_output: if set to True, returns signature and weights
        if set to False, returns only signatures
    """

    apl_table=build_apl_table(pineforest.n_subsamples)

    n_features = pineforest.trees[0].n_features
    # placeholder for the scores accumulation (set to 0)
    sumweights=np.ones((n_features,len(anomalies_indices)))*epsilon #to avoid /0
    sumdeltas=np.zeros((n_features,len(anomalies_indices)))

    # loop on the forest
    for i_t,t in enumerate(pineforest.trees):
        decision_path=t.decision_path(data)

        # number of anomalies reaching the nodes : used to cut useless branches and leaves
        sub_in_class=np.ravel(np.sum(decision_path[anomalies_indices],axis=0)) #ravel : because sum returns a matrix
        # removing also leaves (because later we loop only on decision nodes)
        selection = (sub_in_class>0)&(t.feature>-2)

        features = t.feature[selection] # list of active features
        # active left and right nodes only
        cl=t.children_left[selection]
        cr=t.children_right[selection]

        # spread the event weights along the decision path
        weights=decision_path[anomalies_indices].T*anomalies_weight # note the transpose here -> hence a transpose at the end
        # get the depth of each node of the tree
        node_depth=calc_node_depth(t,selection) #size len of tree, but only filled with relevant values for selection

        # get the average expected depth of each active node
        depth=(apl_table[t.n_node_samples]+node_depth) # size len of tree (needed to select correct left and right sub-trees)
        # ventilate the depth in order to have the delta depth
        deltas=np.multiply(weights[cl].toarray(),depth[cl,np.newaxis]) \
            +np.multiply(weights[cr].toarray(),depth[cr,np.newaxis]) \
            -np.multiply(weights[selection].toarray(),depth[selection,np.newaxis])

        # prepare the indices where to accumulate
        whereadd=(t.feature[selection][weights[selection].nonzero()[0]],weights[selection].nonzero()[1])

        # accumulate weights
        np.add.at(sumweights,whereadd,weights[selection].toarray()[weights[selection].nonzero()])
        # accumulate delta depth
        np.add.at(sumdeltas,whereadd,deltas[weights[selection].nonzero()])

        # activate for a debug return after the 1st tree
        #return sumweights,sumdeltas

    if full_output:
        # average depth gained [event , feature] + weights
        return (sumdeltas/sumweights).T, sumweights.T
    else:
        # average depth gained [event , feature]
        return (sumdeltas/sumweights).T
