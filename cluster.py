from random import sample, randint, random
from math import floor, pow, fabs, sqrt
from uuid import uuid4

from numpy import argmin

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import chain
import time
from functools import reduce, partial
from scipy.spatial.distance import euclidean

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def distance(*points):
    return sqrt(reduce(lambda result, incoming: result + pow(incoming[0] - incoming[1], 2),
                       zip(*points),
                       0))

# vec(OM) = (vec(OA) + vec(OB)) / 2
def middle(points):
    return tuple(map(lambda component: (component[0] + component[1]) / 2,
                 zip(*points)))

def pd_calculator(pnormal, middle):
    d = reduce(lambda result, incoming: result + -(incoming[0] * incoming[1]),
               zip(pnormal, middle),
               0)

    return partial(_pd_calculator, pnormal=pnormal, d=d)

def _pd_calculator(point, pnormal, d):
    return ((reduce(lambda result, incoming: result + incoming[0] * incoming[1], zip(pnormal, point), 0) + d) / sqrt(reduce(lambda result, incoming: result + pow(incoming, 2), pnormal, 0)))

# AB = OA - OB
def pnormal(points):
    return tuple(map(lambda incoming: incoming[0] - incoming[1],
                    zip(*points)))

def split_points(points):
    result = []

    while(True):
        result = sample(points, 2)

        if reduce(lambda result, incoming: result or incoming[0] - incoming[1] != 0,
                  zip(*result),
                  False):
            break

    return result

def node_build(points, leaf_max=5, node_id='ROOT'):
    node, children = {}, []

    if len(points) <= leaf_max:
        node = {
            'type': 'leaf',
            'count': len(points),
            'id': node_id,
            'children': points
        }
    else:
        split = split_points(points)
        pd_calc = pd_calculator(pnormal(split), middle(split))
        negative, positive = [], []
        children = [uuid4(), uuid4()]

        for point in points:
            if pd_calc(point) > 0:
                positive.append(point)
            else:
                negative.append(point)

        node = {
            'type': 'branch',
            'func': pd_calc,
            'count': len(points),
            'id': node_id,
            'children': children
        }

        children = [partial(node_build, negative, leaf_max, children[0]),
                    partial(node_build, positive, leaf_max, children[1])]

    return (node_id, node), children

def tree(points, leaf_max=5, n_jobs=1):
    result = {}

    tasks = [partial(node_build, points, leaf_max)]
    while tasks:
        tasks_next = []
        for task in tasks:
            node, sub_tasks = task()

            tasks_next = tasks_next + sub_tasks
            result[node[0]] = node[1]

        tasks = tasks_next

    return result

    #with ThreadPoolExecutor(max_workers=n_jobs) as pool:
    #    tasks = [partial(node_build, points, leaf_max)]
    #    while True:
    #        jobs = []

    #        for task in tasks:
    #            jobs.append(pool.submit(task))

    #        tasks = []

    #        for job in jobs:
    #            node, sub_tasks = job.result()

    #            for st in sub_tasks:
    #                tasks.append(st)

    #            result[node[0]] = node[1]

    #        if len(tasks) == 0:
    #            break

    #    return result

def leaves_get(tree):
    result = []

    if tree['type'] == 'leaf':
        result.append(tree)
    else:
        result = leaves_get(tree['children'][0]) + leaves_get(tree['children'][1])

    return result

def leaves_nearest(point, tree, threshold, branch_id='ROOT', n_jobs=1):
    result = []

    branches = [branch_id]

    while branches:
        branches_next = []
        for branch_id in branches:
            if tree[branch_id]['type'] == 'leaf':
                result.append(tree[branch_id])
            else:
                delta = tree[branch_id]['func'](point)

                if threshold > 0 and -threshold < delta and delta < threshold:
                    branches_next.append(tree[branch_id]['children'][0])
                    branches_next.append(tree[branch_id]['children'][1])
                elif delta > 0:
                    branches_next.append(tree[branch_id]['children'][1])
                else:
                    branches_next.append(tree[branch_id]['children'][0])

        branches = branches_next

    return result

    #with ThreadPoolExecutor(max_workers=n_jobs) as pool:
    #    tasks = ['ROOT']
    #    while True:
    #        jobs = []
    #        for task in tasks:
    #            if tree[task]['type'] == 'leaf':
    #                result.append(tree[task])
    #            else:
    #                jobs.append((task, pool.submit(tree[task]['func'], point)))

    #        tasks = []
    #        for branch_id, job in jobs:
    #            delta = job.result()

    #            if delta > 0:
    #                tasks.append(tree[branch_id]['children'][1])
    #            elif delta > -threshold:
    #                tasks.append(tree[branch_id]['children'][0])
    #                tasks.append(tree[branch_id]['children'][1])
    #            else:
    #                tasks.append(tree[branch_id]['children'][0])

    #        if len(tasks) == 0:
    #            break

    #    return result

def distances_find(query, points):
    return [distance(query, point) for point in points]
    #with ThreadPoolExecutor(max_workers=4) as pool:
    #    jobs = []

    #    for point in points:
    #        jobs.append(pool.submit(euclidean,
    #                                query,
    #                                point))

    #    return [job.result() for job in jobs]

def search_tree(query, nleaves):
    candidates = list(chain.from_iterable([leaf['children'] for leaf in nleaves]))
    distances = distances_find(query, candidates)
    idx_min = argmin(distances)

    return (distances[idx_min], candidates[idx_min])

def search_brute(query, points):
    distances = distances_find(query, points)
    idx_min = argmin(distances)

    return (distances[idx_min], points[idx_min])

dim = 5
points = []
print('Generating Points')
for _ in range(10000):
    points.append(tuple([random() for __ in range(dim)]))

print('Building Tree')
_tree = tree(points, leaf_max=5, n_jobs=20)

print('total leaves: {}'.format(len([node for _, node in _tree.items() if node['type'] == 'leaf'])))

query = tuple([random() for __ in range(dim)])

#print('Given Query {}'.format(query))
#print('Cluster Answer')
#t0 = time.clock()
#nleaves = leaves_nearest(query, _tree, 0, n_jobs=2)
#canswer = search_tree(query, nleaves)
#print('Search took {} seconds'.format(time.clock() - t0))
#print(canswer)

print('Brutal Search Answer')
t0 = time.clock()
ganswer = search_brute(query, points)
print('Search took {} seconds'.format(time.clock() - t0))
print(ganswer)

for threshold in range(0, int(1e7) + 1, 5):
    print('Cluster Answer for threshold {}'.format(threshold / 100))
    t0 = time.clock()
    nleaves = leaves_nearest(query, _tree, threshold / 100, n_jobs=20)
    canswer = search_tree(query, nleaves)
    print('Search took {} seconds'.format(time.clock() - t0))
    print('N-leaves: {}'.format(len(nleaves)))
    print(canswer)

    if canswer[0] == ganswer[0]:
        break

#####################################
# Figure generation

#fig = plt.figure()
#ax = fig.add_subplot(111)

#nleaves_uuid = [leaf['id'] for leaf in nleaves]

#color = [random() for _ in range(3)]
#for leaf in [node for _, node in _tree.items() if node['type'] == 'leaf']:
#    leaf_color = [random() for _ in range(3)] if leaf['id'] in nleaves_uuid else color

#    for point in leaf['children']:
#        ax.plot([point[0]],
#                [point[1]],
#                #[point[2]],
#                color=leaf_color,
#                marker='x' if leaf['id'] in nleaves_uuid else '+',
#                markersize=10 if point in [canswer[1], ganswer[1]] else 5)

#query_color = [random() for _ in range(3)]
#ax.plot([query[0]],
#        [query[1]],
#        #[query[2]],
#        color=query_color,
#        marker='o')

#plt.show()
