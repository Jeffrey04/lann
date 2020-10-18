from __future__ import annotations

import logging
from collections import namedtuple
from collections.abc import Mapping
from functools import partial, reduce
from itertools import combinations, product
from math import inf, isclose, sqrt
from operator import sub
from random import choices, gauss, sample, uniform
from typing import Iterator, Optional, Tuple, Union
from uuid import uuid4

import dask.bag as db
import matplotlib.pyplot as plt
from toolz.functoolz import do

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("lann")


class Plane:
    def __init__(self, dimension: int, normal: Vector, constant: float):
        assert dimension == normal.dimension

        self.dimension = dimension
        self.normal = normal
        self.constant = constant


class Forest(Mapping):
    def __init__(self, dimension, leaf_max, split_max, shrinkage, data):
        self.dimension = dimension
        self.leaf_max = leaf_max
        self.split_max = split_max
        self.shrinkage = shrinkage
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self.idx_current = 0

        for item in self.data:
            yield item

    def __next__(self):
        self.idx_current += 1

        try:
            return self.data[self.idx_current]
        except IndexError:
            raise StopIteration

    def __repr__(self):
        return str(
            {
                "dimension": self.dimension,
                "leaf_max": self.leaf_max,
                "split_max": self.split_max,
                "shrinkage": self.shrinkage,
                "data": self.data,
            }
        )

    def __str__(self):
        return self.__repr__()


class Tree(dict):
    def __init__(
        self, dimension, root, leaf_max, split_max, idx_pool, target, *args, **kwargs
    ):
        super(Tree, self).__init__(*args, **kwargs)

        self.dimension = dimension
        self.root = root
        self.leaf_max = leaf_max
        self.split_max = split_max
        self.idx_pool = idx_pool
        self.target = target

    def node_root(self):
        return self[self.root]

    def node(self, node_id):
        return self[node_id]


class Node:
    def __init__(
        self,
        node_id: int,
        is_leaf: bool,
        count: int,
        children: Tuple[int, ...],
        decider: decider = None,
    ):
        self.id = node_id
        self.is_leaf = is_leaf
        self.count = count
        self.children = children
        self.decider = decider


class Vectors(Mapping):
    def __init__(self, dimension, data):
        self.dimension = dimension
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self.idx_current = 0

        for item in self.data:
            yield item

    def __next__(self):
        self.idx_current += 1

        try:
            return self.data[self.idx_current]
        except IndexError:
            raise StopIteration

    def __repr__(self):
        return str({"dimension": self.dimension, "data": self.data})

    def __str__(self):
        return self.__repr__()


class Vector(dict):
    def __init__(self, dimension, *args, **kwargs):
        super(Vector, self).__init__(*args, **kwargs)

        self.dimension = dimension


def branch_decider_get(points: Vectors):
    normal = vector_normal(*points)

    return (
        lambda point: point_to_plane_calculate_distance(
            point,
            Plane(
                points_x.dimension,
                normal,
                plane_equation_get_constant(point_middle(*points), normal),
            ),
        )
        < 0
    )


def cv_build_forest(
    kfold,
    points_x,
    points_y,
    splitter,
    size,
    is_bootstrap=False,
    leaf_max=5,
    shrinkage=None,
    split_max=inf,
):
    result = {"mse": {"test": inf}}

    subset_x = sample_get_fold(points_x, kfold)
    subset_y = sample_get_fold(points_y, kfold)

    for k in range(kfold):
        logger.info("Generating forest %s/%s", k + 1, kfold)
        test_x, training_x = subset_x(k)
        test_y, training_y = subset_y(k)

        forest = forest_build(
            Vectors(points_x.dimension, training_x),
            tuple(training_y),
            splitter,
            size,
            is_bootstrap,
            leaf_max,
            shrinkage,
            split_max,
        )

        logger.info("Calculating mse %s/%s", k + 1, kfold)
        _result = {
            "mse": {
                "training": mse_calculate(forest, training_x, training_y, training_y),
                "test": mse_calculate(forest, test_x, test_y, training_y),
            },
            "training_x": training_x,
            "training_y": training_y,
            "forest": forest,
        }

        if _result["mse"]["test"] < result["mse"]["test"]:
            result = _result

    return result


def distance_euclidean(
    alpha: dict[int, float], beta: dict[int, float], dimension: int
) -> float:
    return sqrt(sum(pow(alpha.get(i, 0) - beta.get(i, 0), 2) for i in range(dimension)))


def forest_build(
    points_x,
    points_y,
    splitter,
    size,
    is_bootstrap=False,
    leaf_max=5,
    shrinkage=None,
    split_max=inf,
):
    result = None

    def _builder(tree_idx):
        logger.info("Building tree %s / %s", tree_idx + 1, size)

        idx_pool = samples_get(range(len(points_x)), is_bootstrap)

        return tree_build(points_x, points_y, idx_pool, splitter, leaf_max, split_max)

    if shrinkage:
        assert not is_bootstrap

        forest = []

        points_r = points_y

        for tree_idx in range(size):
            logger.info("Building tree %s / %s", tree_idx + 1, size)

            forest.append(
                tree_build(
                    points_x,
                    points_r,
                    range(len(points_x)),
                    splitter,
                    leaf_max,
                    split_max,
                )
            )

            points_r = tuple(
                r - shrinkage * tree_predict(forest[-1], x)
                for x, r in zip(points_x, points_r)
            )

        result = Forest(points_x.dimension, leaf_max, split_max, shrinkage, forest)
    else:
        result = Forest(
            points_x.dimension,
            leaf_max,
            split_max,
            shrinkage,
            db.from_sequence(range(size)).map(_builder).compute(),
        )

    return result


def forest_predict(forest, x, points_y):
    result = None

    if forest.shrinkage:
        result = sum(forest.shrinkage * tree_predict(tree, x) for tree in forest)
    else:
        result = sum(tree_predict(tree, x) for tree in forest) / len(forest)

    return result


def mse_calculate(forest, subset_x, subset_y, points_y):
    return sum(
        pow(y - forest_predict(forest, x, points_y), 2)
        for x, y in zip(subset_x, subset_y)
    ) / len(subset_x)


def node_build(
    points_x, points_y, splitter, idx_pool, node_id, leaf_max, is_leaf=False
):
    node, children = None, []

    if is_leaf or len(idx_pool) <= leaf_max:
        node = Node(node_id, True, len(idx_pool), idx_pool)
    else:
        children_nodes = [uuid4() for _ in range(2)]

        decider, branches = splitter(points_x, points_y, idx_pool)

        node = Node(node_id, False, len(idx_pool), children_nodes, decider)

        children.extend(
            partial(
                node_build,
                points_x,
                points_y,
                splitter,
                branches[branch],
                child_id,
                leaf_max,
            )
            for branch, child_id in zip(branches.keys(), children_nodes)
        )

    return node, children


def plane_equation_get_constant(point_on_plane, vector_normal):
    return sum(
        -(vector_normal[idx] * point_on_plane[idx])
        for idx in range(point_on_plane.dimension)
    )


def points_are_different(points: Points):
    return not all(
        sub(points[0].get(i, 0), points[1].get(i, 0)) == 0
        for i in range(points.dimension)
    )


def point_middle(alpha, beta):
    assert alpha.dimension == beta.dimension

    return Vector(
        alpha.dimension,
        {
            idx: (alpha.get(idx, 0) + beta.get(idx, 0)) / 2.0
            for idx in range(alpha.dimension)
        },
    )


def point_to_plane_calculate_distance(point, plane):
    assert point.dimension == plane.dimension

    return (
        sum([point.get(i, 0) * plane.normal.get(i, 0) for i in range(point.dimension)])
        + plane.constant
    ) / sqrt(sum([pow(value, 2) for _, value in plane.normal.items()]))


def sample_get_fold(points, n):
    return lambda k: (
        tuple(points[idx] for idx in range(len(points)) if idx % n == 0),
        tuple(points[idx] for idx in range(len(points)) if not idx % n == 0),
    )


def samples_get(idx_pool, is_bootstrap):
    result = idx_pool

    if is_bootstrap:
        result = choices(idx_pool, k=len(idx_pool))

    return result


def samples_generate(n: int, dimension: int) -> Vectors:
    return Vectors(
        dimension,
        tuple(
            Vector(dimension, {idx: gauss(0, 1) for idx in range(dimension)})
            for _ in range(n)
        ),
    )


def splitter_best(points_x, points_y, idx_pool):
    best, result = inf, None
    candidates = splitter_get_candidates(points_x, idx_pool)

    for current_idx, points in enumerate(candidates):
        # logger.info(f"Finding best split %s/%s", current_idx + 1, len(candidates))

        decider = branch_decider_get(points)

        branches = reduce(
            lambda branches, idx: branches[decider(points_x[idx])].append(idx)
            or branches,
            idx_pool,
            {False: [], True: []},
        )

        rss = sum(
            sum(
                pow(
                    points_y[idx]
                    - (sum(points_y[idx] for idx in sub_pool) / len(sub_pool)),
                    2,
                )
                for idx in sub_pool
            )
            for _, sub_pool in branches.items()
        )

        if rss < best:
            best = rss
            result = decider, branches

    return result


def splitter_best_concurrent(points_x, points_y, idx_pool):
    def _splitter(current, points):
        decider = branch_decider_get(points)

        branches = reduce(
            lambda branches, idx: branches[decider(points_x[idx])].append(idx)
            or branches,
            idx_pool,
            {False: [], True: []},
        )

        rss = sum(
            sum(
                pow(
                    points_y[idx]
                    - (sum(points_y[idx] for idx in sub_pool) / len(sub_pool)),
                    2,
                )
                for idx in sub_pool
            )
            for _, sub_pool in branches.items()
        )

        return (
            {"best": rss, "result": (decider, branches)}
            if rss < current["best"]
            else current
        )

    return (
        db.from_sequence(splitter_get_candidates(points_x, idx_pool))
        .fold(
            _splitter,
            lambda current, incoming: incoming
            if incoming["best"] < current["best"]
            else current,
            initial={"best": inf, "result": None},
        )
        .compute()
        .get("result")
    )


def splitter_get_candidates(points_x, idx_pool):
    return tuple(
        filter(
            points_are_different,
            (
                Vectors(points_x.dimension, tuple(points_x[idx] for idx in pair))
                for pair in combinations(idx_pool, 2)
            ),
        )
    )


def splitter_random(points_x: Vectors, _points_y, idx_pool: tuple):
    result = None

    while not result:
        points = Vectors(
            points_x.dimension, tuple(points_x[idx] for idx in sample(idx_pool, 2))
        )

        if points_are_different(points):
            decider = branch_decider_get(points)

            result = decider, reduce(
                lambda branches, idx: branches[decider(points_x[idx])].append(idx)
                or branches,
                idx_pool,
                {False: [], True: []},
            )

    return result


def tree_build(
    points_x: Vectors,
    points_y: Vectors,
    idx_pool,
    splitter,
    leaf_max: int = 5,
    split_max: int = inf,
) -> dict:
    assert split_max > 0

    tree, splits, progress = {}, 0, [0]

    root_id = uuid4()
    builders = (
        partial(
            node_build,
            points_x,
            points_y,
            splitter,
            idx_pool,
            root_id,
            leaf_max,
        ),
    )

    while builders:
        builders_next = []

        for builder in builders:
            node, children = builder() if splits < split_max else builder(is_leaf=True)

            if node.is_leaf:
                progress.append(progress[-1] + node.count)

                # logger.info(
                #    "Tree building progress: (%s terminals) %s/%s",
                #    len(progress) - 1,
                #    progress[-1],
                #    len(points_x),
                # )
            else:
                splits += 1

            tree[node.id] = node

            builders_next.extend(children)

        builders = builders_next

    return Tree(
        points_x.dimension, root_id, leaf_max, split_max, idx_pool, points_y, tree
    )


def tree_predict(tree, x):
    result = None
    node_next = tree.node_root().children[0 if tree.node_root().decider(x) else -1]

    while not result:
        if tree.node(node_next).is_leaf:
            result = sum(
                tree.target[idx] for idx in tree.node(node_next).children
            ) / len(tree.node(node_next).children)
        else:
            node_next = tree.node(node_next).children[
                0 if tree.node(node_next).decider(x) else -1
            ]

    return result


def vector_normal(alpha, beta):
    assert alpha.dimension == beta.dimension

    return Vector(
        alpha.dimension,
        {idx: alpha.get(idx, 0) - beta.get(idx, 0) for idx in range(alpha.dimension)},
    )


# some quick tests
assert distance_euclidean({}, {1: 1}, 2) == 1.0


if __name__ == "__main__":
    from dask.diagnostics import ProgressBar
    from sklearn.datasets import load_boston
    from sklearn.preprocessing import MinMaxScaler

    ds = load_boston()
    points_y = ds.target
    points_x = Vectors(
        ds.data.shape[1],
        tuple(
            Vector(ds.data.shape[1], {idx: value for idx, value in enumerate(point)})
            for point in MinMaxScaler().fit_transform(ds.data)
        ),
    )

    # ProgressBar().register()
    sizes = (1, 5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)

    # n = 1000
    # points_x = samples_generate(n, 2)
    # points_y = tuple(uniform(60, 5000) for _ in range(n))

    print("Random split full sample")
    print("========================")
    print()

    print("forest_size,mse_training,mse_test")
    for size in sizes:
        result = cv_build_forest(5, points_x, points_y, splitter_random, size)
        print(f"{size},{result['mse']['training']},{result['mse']['test']}", flush=True)

    print()
    print()
    print("Random split with shrinkage")
    print("===========================")
    print()

    for size in sizes:
        result = cv_build_forest(
            5, points_x, points_y, splitter_random, size, shrinkage=0.001, split_max=1
        )
        print(f"{size},{result['mse']['training']},{result['mse']['test']}", flush=True)

    print()
    print()
    print("Random split bootstrap sampling")
    print("===============================")
    print()

    for size in sizes:
        result = cv_build_forest(
            5, points_x, points_y, splitter_random, size, is_bootstrap=True
        )
        print(
            f"{len(result['forest'])},{result['mse']['training']},{result['mse']['test']}",
            flush=True,
        )

    print()
    print()
    print("Best split bootstrap sampling")
    print("=============================")
    print()

    for size in sizes:
        result = cv_build_forest(
            5, points_x, points_y, splitter_best, size, is_bootstrap=True
        )
        print(
            f"{len(result['forest'])},{result['mse']['training']},{result['mse']['test']}",
            flush=True,
        )

    # tree = tree_build(points_x, points_y, range(n), splitter_random, 20)
    # print(tree_predict(tree, points_x[0]), points_y[0])

    # import matplotlib.pyplot as plt

    # for node in (node for _, node in tree.items() if node.is_leaf):
    #    points = [points_x[idx] for idx in node.children]

    #    plt.plot(
    #        [point[0] for point in points],
    #        [point[1] for point in points],
    #        "x",
    #        fillstyle="none",
    #    )

    # plt.show()
