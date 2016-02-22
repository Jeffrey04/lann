from random import sample
from math import pow, sqrt, floor, ceil
from uuid import uuid4
from itertools import chain
from functools import reduce, partial
from operator import sub
from graphviz import Digraph
import logging

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def batch_build(builders):
    return [builder() for builder in builders]

def batch_get_sequence(total, leaf_max):
    batches, batch_size = [], ceil(total / floor(total / (leaf_max * 100)))

    for idx in range(0, total, batch_size):
        batches.append((idx, min(idx + batch_size, total)))

    return batches

def batch_group_keys(sequence, keys, tree_count):
    return list(chain.from_iterable([(uuid4(), keys[start:end]) for start, end in sequence] for _ in range(tree_count)))

def distance_euclidean(alpha, beta, dimension):
    return sqrt(sum(pow(alpha.get(i, 0) - beta.get(i, 0), 2) for i in range(dimension)))

def forest_build(points, pmeta, tree_count, leaf_max=5, n_jobs=1, batch_size=1000):
    forest = {}

    batches = batch_group_keys(batch_get_sequence(len(points), leaf_max),
                               list(points.keys()),
                               tree_count)

    meta = {'count': tree_count,
            'leaf_max': leaf_max,
            'roots': list(zip(*batches))[0]}

    builders = [partial(node_build, points, pmeta, keys, root_id, leaf_max)
                for root_id, keys
                in batches]

    if n_jobs == 1:
        forest = forest_build_single(builders, len(points) * tree_count)
    else:
        forest = forest_build_multi(builders, len(points) * tree_count, n_jobs, batch_size)

    return meta, forest

def forest_build_multi(builders, forest_total, n_jobs, batch_size):
    forest, progress = {}, [0]

    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        while builders:
            jobs, builders_next = [], []

            for idx in range(0, len(builders), batch_size):
                jobs.append(pool.submit(batch_build, builders[idx:idx+batch_size]))

            for job in jobs:
                for node, builders_sub in job.result():
                    if node['type'] == 'leaf':
                        progress.append(progress[-1] + node['count'])

                        progress_log('Forest building progress', progress, forest_total)

                    forest[node['id']] = node
                    builders_next.append(builders_sub)

            builders = list(chain.from_iterable(builders_next))

        return forest

def forest_build_single(builders, forest_total):
    forest, progress = {}, [0]

    while builders:
        builders_next = []

        for builder in builders:
            node, builders_sub = builder()

            if node['type'] == 'leaf':
                progress.append(progress[-1] + node['count'])

                progress_log('Forest building progress', progress, forest_total)

            forest[node['id']] = node
            builders_next.append(builders_sub)

        builders = list(chain.from_iterable(builders_next))

    return forest

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

def forest_query_neighbourhood(query, forest, fmeta, threshold, n_jobs):
    result = []

    nodes = [(1., root_id) for root_id in fmeta['roots']]

    if n_jobs == 1:
        result = query_neighbourhood_single(nodes, forest, threshold)
    else:
        result = query_neighbourhood_multi(nodes, forest, threshold, n_jobs)

    return result

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

def progress_log(caption, progress, total):
    to_print = False

    if progress[-1] == total:
        to_print = True
    elif len(progress) > 1 and floor(progress[-2] / total * 10) < floor(progress[-1] / total * 10):
        to_print = True

    to_print and logging.info('{} {: >6.2f}%'.format(caption, progress[-1] / total * 100))

def query_get_children(query, forest, threshold, multiplier, node_id):
    result, delta = [], forest[node_id]['func'](query)

    if threshold > 0 and delta > 0 and delta <= threshold:
        result.append((multiplier * 0.9,
                       forest[node_id]['children'][False]))
        result.append((multiplier * 1.,
                       forest[node_id]['children'][True]))
    elif threshold > 0 and delta <= 0 and -threshold <= delta:
        result.append((multiplier * 1.,
                       forest[node_id]['children'][False]))
        result.append((multiplier * 0.9,
                       forest[node_id]['children'][True]))
    else:
        result.append((multiplier * 1.,
                       forest[node_id]['children'][delta > 0]))

    return result

def query_neighbourhood_multi(nodes, forest, threshold, n_jobs):
    result = []

    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        while nodes:
            nodes_next, jobs = [], []

            for idx in range(0, len(nodes), 1000):
                _job = []

                for multiplier, node_id in nodes[idx:1000]:
                    if forest[node_id]['type'] == 'leaf':
                        result.append((1 - multiplier, forest[node_id]))
                    else:
                        _job.append(partial(query_get_children,
                                            query,
                                            forest,
                                            threshold,
                                            multiplier,
                                            node_id))

                jobs.append(pool.submit(batch_build, _job))

            for job in jobs:
                nodes_next.append(chain.from_iterable(job.result()))

            nodes = list(chain.from_iterable(nodes_next))

        return result

def query_neighbourhood_single(nodes, forest, threshold):
    result = []

    while nodes:
        nodes_next = []

        for multiplier, node_id in nodes:
            if forest[node_id]['type'] == 'leaf':
                result.append((1 - multiplier, forest[node_id]))
            else:
                delta = forest[node_id]['func'](query)

                if threshold > 0 and delta > 0 and delta <= threshold:
                    nodes_next.append((multiplier * 0.9,
                                       forest[node_id]['children'][False]))
                    nodes_next.append((multiplier * 1.,
                                       forest[node_id]['children'][True]))
                elif threshold > 0 and delta <= 0 and -threshold <= delta:
                    nodes_next.append((multiplier * 1.,
                                       forest[node_id]['children'][False]))
                    nodes_next.append((multiplier * 0.9,
                                       forest[node_id]['children'][True]))
                else:
                    nodes_next.append((multiplier * 1.,
                                       forest[node_id]['children'][delta > 0]))

        nodes = nodes_next

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
