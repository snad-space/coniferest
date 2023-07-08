Feature Signature
=================

Introduction
------------

Feature signature is implemented as a tool to help the analyst understanding which features contributed to the score of an anomaly or a nominal instance. It can also provide the average signature for a set of instances.

It is build on top of isolation forest, once the forest is built. As such, it can operate the same way both on isolation forest or pineforest.

The key assumption is to decompose the score of an instance into a sum of contributions linked to the features that were used at the decision nodes used to score this instance. Then, an average of the contribution of a given feature can be built.

As the anomaly region is defined by the lower scores, features that contribute the most to ranking an anomaly will have a negative signature. On the other hand, features that contribute to rank an instance as nominal will have a positive value. For nominals, the highest signatures then corresponds to features useful to separate them from anomalies, while this is the opposite for anomalies. A signature of 0 is the indication that the feature plays no role for the score.

Computational aspects
---------------------

For isolation forest, the score of a given instance is :math:`-2^{-\dfrac{1}{N_t \bar{d}(N_s)}\sum_i d_i` where :math:`N_t` is the number of trees in the forest, :math:`\bar{d}(N_s)` is the expected average depth of a tree built from :math:`N_s` subsamples, and :math:`d_i` is the depth of the leaf node reached by the instance within the tree :math:`i`, starting at 0 for the root node.

Without loss of generality, the ranking is left invariant considering the average depth instead of the score :
:math:`<d> = \dfrac{1}{N_t \bar{d}(N_s)}\sum_i d_i`. Signature is therefore based on the depth, and normalized such as the size of the subsample does not affect the expected values.

For a given tree, the depth can be understood as :math:`d=d_n` which is the actual depth at the leaf node of depth :math:`n` reached by the instance. Note that in the case of tree of maximal depth :math:`D`, :math:`d_n=n` if
:math:`n<D` or :math:`d_n=n+\bar{d}(N_n)` if :math:`n=D` , where :math:`(N_n)` is the number of subsamples having reached the terminal leaf of the instance at training time.

The depth can then be decomposed as :math:`d_n=d_n-d_{n-1}+d_{n-1} = d_n+\sum_j=0^{n-1}(-d_j+d_j) = d_0 + \sum_j=1^{n}(d_j-d_{j-1})` where :math:`d_j=j+\bar{d}(N_j)`, that is, the depth of the node plus the average expected depth of a tree built out of the :math:`(N_j)` subsamples that reached this node at training time.

As :math:`(d_j-d_{j-1})` is driven by the feature :math:`f_{j-1}` used to split the node, it is possible to rewrite the sum as
:math:`d_n=d_0 + \sum_j=1^{n}\delta_{f_{j-1}}` where :math:`\delta_{f_{j-1}}=(d_j-d_{j-1})`.

The signature of a feature :math:`f^0` is then given by the average over all trees :
:math:`d_{f^0}=\dfrac{1}{N_{f^0} \bar{d}(N_s)}\sum_{i,j|f_{i,j-1}=f^0} \delta_{f_{i,j-1}}` where :math:`i` runs on all trees, and the sum is only performed for features maching the feature of interest :math:`f^0`, :math:`N_{f^0}`being the number of occurences within the sum.

Implementation
--------------

TBD...
