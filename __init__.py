from random import sample
from math import pow, sqrt
from uuid import uuid4
from itertools import chain
from functools import reduce, partial

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def common(alpha, beta):
    idx_alpha, idx_beta = 0, 0
    result = []
    while alpha[idx_alpha:] and beta[idx_beta:]:
        if alpha[idx_alpha][0] > beta[idx_beta][0]:
            idx_beta = idx_beta + 1
        elif alpha[idx_alpha][0] < beta[idx_beta][0]:
            idx_alpha = idx_alpha + 1
        else:
            result.append((alpha[idx_alpha][-1], beta[idx_beta][-1]))
            idx_alpha = idx_alpha + 1
            idx_beta = idx_beta + 1

    return zip(*result)

def distance_euclidean(alpha, beta, dimension):
    return sqrt(sum(pow(element(alpha, i) - element(beta, i), 2) for i in range(dimension)))

def element(point, i):
    position = point['positions'].get(i, False)

    return point['point'][position][-1] if position is not False else 0

def forest_build(points, tree_count, leaf_max=5, n_jobs=1):
    if n_jobs == 1:
        return tuple(tree_build(points, leaf_max) for _ in range(tree_count))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            jobs = []
            for _ in range(tree_count):
                jobs.append(pool.submit(tree_build,
                                        points,
                                        leaf_max))

            return tuple(job.result() for job in jobs)

def forest_query_neighbourhood(query, forest, threshold=0, n_jobs=1):
    if n_jobs == 1:
        return chain.from_iterable(query_neighbourhood(query, tree, threshold)
                                   for tree in forest)
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            jobs = []
            for tree in forest:
                jobs.append(pool.submit(query_neighbourhood,
                                        query,
                                        tree,
                                        threshold))

            return chain.from_iterable(job.result() for job in jobs)

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
        split = split_points([points['points'][idx] for idx in point_ids])
        distance_calc = plane_point_distance_calculator(
            plane_normal(*split, points['dimension']),
            plane_point(*split, points['dimension']),
            points['dimension'])
        branches = {False: [], True: []}
        child_ids = {False: uuid4(), True: uuid4()}

        for idx in point_ids:
            branches[distance_calc(points['points'][idx]) > 0].append(idx)

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

def plane_normal(alpha, beta, dimension):
    return tuple(element(alpha, i) - element(beta, i) for i in range(dimension))

def plane_point(alpha, beta, dimension):
    return tuple((element(alpha, i) + element(beta, i)) / 2. for i in range(dimension))

def plane_point_distance_calculator(normal, point, dimension):
    return partial(_ppd_calculator,
                   normal=normal,
                   d=reduce(lambda result, incoming: \
                               result + -(incoming[0] * incoming[1]),
                            zip(normal, point),
                            0),
                   dimension=dimension)

def _ppd_calculator(point, normal, d, dimension):
    return ((sum([element(point, i) * normal[i] for i in range(dimension)]) + d) / sqrt(sum([pow(i, 2) for i in normal])))

def point_add(id, vector, dimension, ptype, result={}):
    result[id] = point_convert(vector, ptype)

    return result

def point_convert(vector, ptype):
    ptype_builder = {
        'gensim': point_convert_gensim,
        'list': point_convert_list
    }

    return ptype_builder.get(ptype, point_convert_invalid)(vector)

def point_convert_gensim(vector):
    return {
        'positions': dict([(idx, position) for position, (idx, _) in enumerate(vector)]),
        'point': vector
    }

def point_convert_list(vector):
    point = [(idx, value) for idx, value in enumerate(vector) if value != 0]

    return {
        'positions': dict([(idx, position) for position, (idx, _) in enumerate(point)]),
        'point': point
    }

def point_convert_invalid(*_):
    raise Exception('Not supported')

def points_add(vectors, dimension, ptype, identifiers=None):
    result = {
        'dimension': dimension,
        'points': []
    }

    if identifiers:
        result['points'] = reduce(lambda result, i: point_add(identifiers[i],
                                                              vectors[i],
                                                              dimension,
                                                              ptype,
                                                              result),
                                  range(len(identifiers)),
                                  {})
    else:
        result['points'] = reduce(lambda result, vector: point_add(uuid4(),
                                                                   vector,
                                                                   dimension,
                                                                   ptype,
                                                                   result),
                                  vectors,
                                  {})

    return result

def query_neighbourhood(query, tree, threshold=0, start_id='ROOT'):
    result = []

    branch_ids = [(1., start_id)]

    while branch_ids:
        branches_next = []

        for multiplier, branch_id in branch_ids:
            if tree[branch_id]['type'] == 'leaf':
                result.append((multiplier, tree[branch_id]))
            else:
                delta = tree[branch_id]['func'](query)

                if threshold > 0 and delta > 0 and delta <= threshold:
                    branches_next.append((multiplier * 0.9,
                                          tree[branch_id]['children'][False]))
                    branches_next.append((multiplier * 1.,
                                          tree[branch_id]['children'][True]))
                elif threshold > 0 and delta <= 0 and -threshold <= delta:
                    branches_next.append((multiplier * 1.,
                                          tree[branch_id]['children'][False]))
                    branches_next.append((multiplier * 0.9,
                                          tree[branch_id]['children'][True]))
                else:
                    branches_next.append((multiplier * 1.,
                                          tree[branch_id]['children'][delta > 0]))

            branch_ids = branches_next

    return result

def search_leaf(result, leaf, points):
    searched, rank = result

    for idx in leaf[-1]['children']:
        if idx not in searched:
            searched.append(idx)
            rank.append(distance_euclidean(query, points['points'][idx], points['dimension']))

    return searched, rank


def search(query, points, neighbourhood):
    distances = reduce(partial(search_leaf, points=points),
                       reversed(sorted(neighbourhood, key=lambda _: _[0])),
                       ([], []))

    return sorted(zip(*distances), key=lambda _: _[-1])

def split_points(points):
    result = sample(points, 2)

    return result if reduce(lambda result, incoming: result or incoming[0] - incoming[1] != 0, common(*[point['point'] for point in result]), False) else split_points(points)

def tree_build(points, leaf_max=5):
    result = {}

    builders = [partial(node_build, points, points['points'].keys(), leaf_max)]
    while builders:
        builders_next = []

        for builder in builders:
            (node_id, node), builders_sub = builder()

            builders_next = chain.from_iterable([builders_next, builders_sub])
            result[node_id] = node

        builders = builders_next

    return result
