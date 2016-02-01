from random import sample
from math import pow, sqrt
from uuid import uuid4
from itertools import chain
from functools import reduce, partial
from operator import sub

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def distance_euclidean(alpha, beta, dimension):
    return sqrt(sum(pow(alpha.get(i, 0) - beta.get(i, 0), 2) for i in range(dimension)))

def forest_build(points, tree_count, leaf_max=5, n_jobs=1):
    result = {'count': tree_count, 'leaf_max': leaf_max}

    if n_jobs == 1:
        return dict(result, forest=tuple(tree_build(points, leaf_max) for _ in range(tree_count)))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            jobs = []
            for _ in range(tree_count):
                jobs.append(pool.submit(tree_build,
                                        points,
                                        leaf_max))

            return dict(result, forest=tuple(job.result() for job in jobs))

def forest_query_neighbourhood(query, forest, threshold=0, n_jobs=1):
    if n_jobs == 1:
        return chain.from_iterable(query_neighbourhood(tree, query, threshold)
                                   for tree in forest['forest'])
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            return chain.from_iterable(
                pool.map(partial(query_neighbourhood, query=query, threshold=threshold),
                         forest['forest']))

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
        split = split_points([points['points'][idx] for idx in point_ids],
                             points['dimension'])
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
    return tuple(alpha.get(i, 0) - beta.get(i, 0) for i in range(dimension))

def plane_point(alpha, beta, dimension):
    return tuple((alpha.get(i, 0) + beta.get(i, 0)) / 2. for i in range(dimension))

def plane_point_distance_calculator(normal, point, dimension):
    return partial(_ppd_calculator,
                   normal=normal,
                   d=reduce(lambda result, incoming: \
                               result + -(incoming[0] * incoming[1]),
                            zip(normal, point),
                            0),
                   dimension=dimension)

def _ppd_calculator(point, normal, d, dimension):
    return ((sum([point.get(i, 0) * normal[i] for i in range(dimension)]) + d) / sqrt(sum([pow(i, 2) for i in normal])))

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
    return dict(vector)

def point_convert_list(vector):
    return dict([(idx, value) for idx, value in enumerate(vector) if value != 0])

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

def query_neighbourhood(tree, query, threshold=0, start_id='ROOT'):
    result = []

    branch_ids = [(1., start_id)]

    while branch_ids:
        branches_next = []

        for multiplier, branch_id in branch_ids:
            if tree[branch_id]['type'] == 'leaf':
                result.append((1 - multiplier, tree[branch_id]))
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


def search(query, points, forest, n=1, threshold=None, n_jobs=1):
    neighbourhood = forest_query_neighbourhood(query,
                                               forest,
                                               threshold if threshold else (points['dimension'] * 2e-2),
                                               n_jobs)

    result, candidate_max = {}, max(n, forest['leaf_max']) * forest['count']

    for idx in chain.from_iterable(leaf['children'] for _, leaf in sorted(neighbourhood, key=lambda _: _[0])):
        if idx not in result:
            result[idx] = distance_euclidean(query, points['points'][idx], points['dimension'])

            if len(result) >= candidate_max:
                break

    return sorted(result.items(), key=lambda _: _[-1])[:n]

def split_points(points, dimension):
    result = sample(points, 2)

    return result if reduce(lambda _result, incoming: _result or sub(*[result[i].get(i, 0) for i in range(2)]) != 0, range(dimension), False) else split_points(points)

def tree_build(points, leaf_max=5):
    result = {}

    if len(points['points']) <= leaf_max:
        raise Exception('Not enough points to generate tree')

    builders = [partial(node_build, points, points['points'].keys(), leaf_max)]
    while builders:
        builders_next = []

        for builder in builders:
            (node_id, node), builders_sub = builder()

            builders_next = chain.from_iterable([builders_next, builders_sub])
            result[node_id] = node

        builders = builders_next

    return result
