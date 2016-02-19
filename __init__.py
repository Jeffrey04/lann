from random import sample
from math import pow, sqrt
from uuid import uuid4
from itertools import chain
from functools import reduce, partial
from operator import sub
#from graphviz import Digraph

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def distance_euclidean(alpha, beta, dimension):
    return sqrt(sum(pow(alpha.get(i, 0) - beta.get(i, 0), 2) for i in range(dimension)))

def forest_build(points, pmeta, tree_count, leaf_max=5, n_jobs=1):
    forest = {}

    meta = {'count': tree_count,
            'leaf_max': leaf_max,
            'roots': [str(uuid4()) for _ in range(tree_count)]}

    if n_jobs == 1:
        builders = [partial(node_build, points, pmeta, list(points.keys()), root_id, leaf_max)
                    for root_id
                    in meta['roots']]

        while builders:
            builders_next = []

            for builder in builders:
                node, builders_sub = builder()

                forest[node['id']] = node
                builders_next.append(builders_sub)

            builders = list(chain.from_iterable(builders_next))
    else:
        # otherwise, build one tree after another
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            forest = reduce(lambda result, tree: dict(result, **tree),
                            pool.map(partial(tree_build,
                                             points=points,
                                             pmeta=pmeta,
                                             leaf_max=leaf_max),
                                     meta['roots']),
                            {})

    return meta, forest

def forest_get_dot(forest, count=1):
    result = Digraph()

    builders = [partial(node_get_dot, forest['forest'], root_id, True, result) for root_id in forest['roots']]

    while builders:
        builders_next = []

        for builder in builders:
            builders_sub, result = builder()

            builders_next = chain(builders_next, builders_sub)

        builders = builders_next

    return result

def forest_query_neighbourhood(query, forest, fmeta, threshold=0, n_jobs=1):
    if n_jobs == 1:
        return chain.from_iterable(query_neighbourhood(root_id, forest, query, threshold)
                                   for root_id in fmeta['roots'])
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            return chain.from_iterable(
                pool.map(partial(query_neighbourhood, forest=forest, query=query, threshold=threshold),
                         fmeta['roots']))

def node_build(points, pmeta, point_ids, node_id, leaf_max=5):
    node, children = {}, []

    if len(point_ids) <= leaf_max:
        node = {
            'type': 'leaf',
            'count': len(point_ids),
            'id': node_id,
            'children': point_ids
        }
    else:
        split = split_points([points[idx] for idx in point_ids],
                             pmeta['dimension'])
        distance_calc = plane_point_distance_calculator(
            plane_normal(*split, dimension=pmeta['dimension']),
            plane_point(*split, dimension=pmeta['dimension']),
            pmeta['dimension'])
        branches = {False: [], True: []}
        child_ids = {False: str(uuid4()), True: str(uuid4())}

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
                                pmeta,
                                branches[False],
                                child_ids[False],
                                leaf_max))
        children.append(partial(node_build,
                                points,
                                pmeta,
                                branches[True],
                                child_ids[True],
                                leaf_max))

    return node, children

def node_get_dot(forest, node_id, is_root, dot):
    children = []

    if is_root:
        dot.node(node_id, '')

    if forest[node_id]['type'] == 'branch':
        for child_id in forest[node_id]['children']:
            if forest[child_id]['type'] == 'branch':
                dot.node(child_id, '')
            else:
                dot.node(child_id, str(forest[child_id]['count']))

            dot.edge(node_id, child_id)

            children.append(partial(node_get_dot, forest, child_id, False, dot))

    return children, dot

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
    meta = {'dimension': dimension}
    points = {}

    if identifiers:
        points = reduce(lambda result, i: point_add(identifiers[i],
                                                    vectors[i],
                                                    dimension,
                                                    ptype,
                                                    result),
                        range(len(identifiers)),
                        {})
    else:
        points = reduce(lambda result, vector: point_add(uuid4(),
                                                         vector,
                                                         dimension,
                                                         ptype,
                                                         result),
                        vectors,
                        {})

    return meta, points

def query_neighbourhood(start_id, forest, query, threshold=0):
    result = []

    branch_ids = [(1., start_id)]

    while branch_ids:
        branches_next = []

        for multiplier, branch_id in branch_ids:
            if forest[branch_id]['type'] == 'leaf':
                result.append((1 - multiplier, forest[branch_id]))
            else:
                delta = forest[branch_id]['func'](query)

                if threshold > 0 and delta > 0 and delta <= threshold:
                    branches_next.append((multiplier * 0.9,
                                          forest[branch_id]['children'][False]))
                    branches_next.append((multiplier * 1.,
                                          forest[branch_id]['children'][True]))
                elif threshold > 0 and delta <= 0 and -threshold <= delta:
                    branches_next.append((multiplier * 1.,
                                          forest[branch_id]['children'][False]))
                    branches_next.append((multiplier * 0.9,
                                          forest[branch_id]['children'][True]))
                else:
                    branches_next.append((multiplier * 1.,
                                          forest[branch_id]['children'][delta > 0]))

            branch_ids = branches_next

    return result


def search(query, points, pmeta, forest, fmeta, n=1, threshold=None, n_jobs=1):
    neighbourhood = forest_query_neighbourhood(query,
                                               forest,
                                               fmeta,
                                               threshold if threshold else (pmeta['dimension'] * 2e-2),
                                               n_jobs)

    result, candidate_max = {}, max(n, fmeta['leaf_max']) * fmeta['count']

    for idx in chain.from_iterable(leaf['children'] for _, leaf in sorted(neighbourhood, key=lambda _: _[0])):
        if idx not in result:
            result[idx] = distance_euclidean(query, points[idx], pmeta['dimension'])

            if len(result) >= candidate_max:
                break

    return sorted(result.items(), key=lambda _: _[-1])[:n]

def split_points(points, dimension):
    result = sample(points, 2)

    return result if reduce(lambda _result, incoming: _result or sub(*[result[i].get(i, 0) for i in range(2)]) != 0, range(dimension), False) else split_points(points)

def tree_build(root_id, points, pmeta, leaf_max=5):
    result = {}

    if len(points) <= leaf_max:
        raise Exception('Not enough points to generate tree')

    builders = [partial(node_build, points, pmeta, points.keys(), root_id, leaf_max)]
    while builders:
        builders_next = []

        for builder in builders:
            node, builders_sub = builder()

            builders_next.append(builders_sub)
            result[node['id']] = node

        builders = list(chain.from_iterable(builders_next))

    return result
