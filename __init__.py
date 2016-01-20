from random import sample
from math import pow, sqrt
from uuid import uuid4
from itertools import chain
from functools import reduce, partial
from scipy.spatial.distance import euclidean

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def point_add(id, vector, result={}):
    result[id] = vector

    return result

def points_add(vectors, identifiers=None):
    if identifiers:
        return reduce(lambda result, i: point_add(identifiers[i], vectors[i], result),
                      range(len(identifiers)),
                      {})
    else:
        return reduce(lambda result, vector: point_add(uuid4(), vector, result),
                      vectors,
                      {})

def node_build(points, point_ids, leaf_max=5, node_id='ROOT'):
    node, children = {}, []

    if len(point_ids) <= leaf_max:
        node = {
            'type': 'leaf',
            'count': len(point_ids),
            'id': node_id,
            'children': point_ids
        }
    else:
        split = split_points([point for key, point in points.items() if key in point_ids])
        distance_calc = plane_point_distance_calculator(
            plane_normal(*split),
            plane_point(*split))
        branches = {False: [], True: []}
        child_ids = {False: uuid4(), True: uuid4()}

        for idx in point_ids:
            branches[distance_calc(points[idx]) > 0].append(idx)

        node = {
            'type': 'branch',
            'func': distance_calc,
            'count': len(point_ids),
            'id': node_id,
            'children': [idx for _, idx in child_ids.items()]
        }

        children.append(partial(node_build,
                                points,
                                branches[False],
                                leaf_max,
                                child_ids[False]))
        children.append(partial(node_build,
                                points,
                                branches[True],
                                leaf_max,
                                child_ids[True]))

    return (node_id, node), children

def split_points(points):
    result = sample(points, 2)

    return result if reduce(lambda result, incoming: result or incoming[0] - incoming[1] != 0, zip(*result), False) else split_points(points)

def plane_normal(*points):
    return tuple(map(lambda incoming: incoming[0] - incoming[1],
                    zip(*points)))

def plane_point(*points):
    return tuple(map(lambda component: (component[0] + component[1]) / 2,
                 zip(*points)))

def plane_point_distance_calculator(normal, point):
    return partial(_ppd_calculator,
                   normal=normal,
                   d= reduce(lambda result, incoming: \
                                result + -(incoming[0] * incoming[1]),
                             zip(normal, point),
                             0))

def _ppd_calculator(point, normal, d):
    return ((reduce(lambda result, incoming: result + incoming[0] * incoming[1], zip(normal, point), 0) + d) / sqrt(reduce(lambda result, incoming: result + pow(incoming, 2), normal, 0)))

def tree_build(points, leaf_max=5):
    result = {}

    builders = [partial(node_build, points, points.keys(), leaf_max)]
    while builders:
        builders_next = []

        for builder in builders:
            (node_id, node), builders_sub = builder()

            builders_next = chain.from_iterable([builders_next, builders_sub])
            result[node_id] = node

        builders = builders_next

    return result

def query_neighbourhood(query, tree, threshold=0, start_id='ROOT'):
    result = []

    branch_ids = [start_id]

    while branch_ids:
        branches_next = []

        for branch_id in branch_ids:
            if tree[branch_id]['type'] == 'leaf':
                result.append(tree[branch_id])
            else:
                delta = tree[branch_id]['func'](query)

                if threshold > 0 and -threshold < delta and delta < threshold:
                    branches_next.append(tree[branch_id]['children'][False])
                    branches_next.append(tree[branch_id]['children'][True])
                else:
                    branches_next.append(tree[branch_id]['children'][delta > 0])

            branch_ids = branches_next

    return result

def search(query, points, neighbourhood):
    candidates = tuple(chain.from_iterable([leaf['children'] for leaf in neighbourhood]))
    distances = (euclidean(query, points[idx]) for idx in candidates)

    return sorted(zip(candidates, distances), key=lambda _: _[-1])
