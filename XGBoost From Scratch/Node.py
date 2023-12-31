import numpy
import pandas
from math import e


class Node:

    """
    A node object that is recursivly called within itslef to construct a regression tree. Based on Tianqi Chen's XGBoost
    the internal gain used to find the optimal split value uses both the gradient and hessian. Also a weighted quantlie sketch
    and optimal leaf values all follow Chen's description in "XGBoost: A Scalable Tree Boosting System" the only thing not
    implemented in this version is sparsity aware fitting or the ability to handle NA values with a default direction.

    Inputs
    ------------------------------------------------------------------------------------------------------------------
    x: pandas datframe of the training data
    gradient: negative gradient of the loss function
    hessian: second order derivative of the loss function
    idxs: used to keep track of samples within the tree structure
    subsample_cols: is an implementation of layerwise column subsample randomizing the structure of the trees
    (complexity parameter)
    min_leaf: minimum number of samples for a node to be considered a node (complexity parameter)
    min_child_weight: sum of the heassian inside a node is a meaure of purity (complexity parameter)
    depth: limits the number of layers in the tree
    lambda: L2 regularization term on weights. Increasing this value will make model more conservative.
    gamma: This parameter also prevents over fitting and is present in the the calculation of the gain (structure score).
    As this is subtracted from the gain it essentially sets a minimum gain amount to make a split in a node.
    eps: This parameter is used in the quantile weighted skecth or 'approx' tree method roughly translates to
    (1 / sketch_eps) number of bins

    Outputs
    --------------------------------------------------------------------------------------------------------------------
    A single tree object that will be used for gradient boosintg.
    """

    def __init__(
        self,
        x,
        gradient,
        hessian,
        idxs,
        subsample_cols=0.8,
        min_leaf=5,
        min_child_weight=1,
        depth=10,
        lambda_=1,
        gamma=1,
        eps=0.1,
    ):
        self.x, self.gradient, self.hessian = x, gradient, hessian
        self.idxs = idxs
        self.depth = depth
        self.min_leaf = min_leaf
        self.lambda_ = lambda_
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.row_count = len(idxs)
        self.col_count = x.shape[1]
        self.subsample_cols = subsample_cols
        self.eps = eps
        self.column_subsample = numpy.random.permutation(self.col_count)[
            : round(self.subsample_cols * self.col_count)
        ]

        self.val = self.compute_gamma(self.gradient[self.idxs], self.hessian[self.idxs])

        self.score = float("-inf")
        self.find_varsplit()

    def compute_gamma(self, gradient, hessian):
        """
        Calculates the optimal leaf value equation (5) in "XGBoost: A Scalable Tree Boosting System"
        """
        return -numpy.sum(gradient) / (numpy.sum(hessian) + self.lambda_)

    def find_varsplit(self):
        """
        Scans through every column and calcuates the best split point.
        The node is then split at this point and two new nodes are created.
        Depth is only parameter to change as we have added a new layer to tre structure.
        If no split is better than the score initalised at the begining then no splits further splits are made
        """
        for c in self.column_subsample:
            self.find_greedy_split(c)
        if self.is_leaf:
            return
        x = self.split_col
        lhs = numpy.nonzero(x <= self.split)[0]
        rhs = numpy.nonzero(x > self.split)[0]
        self.lhs = Node(
            x=self.x,
            gradient=self.gradient,
            hessian=self.hessian,
            idxs=self.idxs[lhs],
            min_leaf=self.min_leaf,
            depth=self.depth - 1,
            lambda_=self.lambda_,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            eps=self.eps,
            subsample_cols=self.subsample_cols,
        )
        self.rhs = Node(
            x=self.x,
            gradient=self.gradient,
            hessian=self.hessian,
            idxs=self.idxs[rhs],
            min_leaf=self.min_leaf,
            depth=self.depth - 1,
            lambda_=self.lambda_,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            eps=self.eps,
            subsample_cols=self.subsample_cols,
        )

    def find_greedy_split(self, var_idx):
        """
        For a given feature greedily calculates the gain at each split.
        Globally updates the best score and split point if a better split point is found
        """
        x = self.x[self.idxs, var_idx]

        for r in range(self.row_count):
            lhs = x <= x[r]
            rhs = x > x[r]

            lhs_indices = numpy.nonzero(x <= x[r])[0]
            rhs_indices = numpy.nonzero(x > x[r])[0]
            if (
                rhs.sum() < self.min_leaf
                or lhs.sum() < self.min_leaf
                or self.hessian[lhs_indices].sum() < self.min_child_weight
                or self.hessian[rhs_indices].sum() < self.min_child_weight
            ):
                continue

            curr_score = self.gain(lhs, rhs)
            if curr_score > self.score:
                self.var_idx = var_idx
                self.score = curr_score
                self.split = x[r]

    def weighted_qauntile_sketch(self, var_idx):
        """
        XGBOOST Mini-Version
        Yiyang "Joe" Zeng
        Is an approximation to the eact greedy approach faster for bigger datasets wher it is not feasible
        to calculate the gain at every split point. Uses equation (8) and (9) from "XGBoost: A Scalable Tree Boosting System"
        """
        x = self.x[self.idxs, var_idx]
        hessian_ = self.hessian[self.idxs]
        df = pandas.DataFrame({"feature": x, "hess": hessian_})

        df.sort_values(by=["feature"], ascending=True, inplace=True)
        hess_sum = df["hess"].sum()
        df["rank"] = df.apply(
            lambda x: (1 / hess_sum) * sum(df[df["feature"] < x["feature"]]["hess"]),
            axis=1,
        )

        for row in range(df.shape[0] - 1):
            # look at the current rank and the next ran
            rk_sk_j, rk_sk_j_1 = df["rank"].iloc[row : row + 2]
            diff = abs(rk_sk_j - rk_sk_j_1)
            if diff >= self.eps:
                continue

            split_value = (df["rank"].iloc[row + 1] + df["rank"].iloc[row]) / 2
            lhs = x <= split_value
            rhs = x > split_value

            lhs_indices = numpy.nonzero(x <= split_value)[0]
            rhs_indices = numpy.nonzero(x > split_value)[0]
            if (
                rhs.sum() < self.min_leaf
                or lhs.sum() < self.min_leaf
                or self.hessian[lhs_indices].sum() < self.min_child_weight
                or self.hessian[rhs_indices].sum() < self.min_child_weight
            ):
                continue

            curr_score = self.gain(lhs, rhs)
            if curr_score > self.score:
                self.var_idx = var_idx
                self.score = curr_score
                self.split = split_value

    def gain(self, lhs, rhs):
        """
        Calculates the gain at a particular split point bases on equation (7) from
        "XGBoost: A Scalable Tree Boosting System"
        """
        gradient = self.gradient[self.idxs]
        hessian = self.hessian[self.idxs]

        lhs_gradient = gradient[lhs].sum()
        lhs_hessian = hessian[lhs].sum()

        rhs_gradient = gradient[rhs].sum()
        rhs_hessian = hessian[rhs].sum()

        gain = (
            0.5
            * (
                (lhs_gradient**2 / (lhs_hessian + self.lambda_))
                + (rhs_gradient**2 / (rhs_hessian + self.lambda_))
                - (
                    (lhs_gradient + rhs_gradient) ** 2
                    / (lhs_hessian + rhs_hessian + self.lambda_)
                )
            )
            - self.gamma
        )
        return gain

    @property
    def split_col(self):
        """
        splits a column
        """
        return self.x[self.idxs, self.var_idx]

    @property
    def is_leaf(self):
        """
        checks if node is a leaf
        """
        return self.score == float("-inf") or self.depth <= 0

    def predict(self, x):
        return numpy.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf:
            return self.val

        node = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return node.predict_row(xi)
