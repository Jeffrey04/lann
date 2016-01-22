# LANN

[![NeighbourSearch](https://farm2.staticflickr.com/1615/24433467801_2f1eb1d76f_z.jpg)](https://www.flickr.com/photos/jeffrey04/24433467801/in/dateposted-public/)

LANN ~~(Lame Approximate Neighbour Search)~~ is a Python re-implementation of [Annoy](https://github.com/spotify/annoy) mainly for Jeffrey04's learning (hence the 'L' in name)/experimental purpose. It does not bring anything new to the table, and is not meant for production use for now (probably wouldn't work with large scale real-life data). Like Annoy, the library can be used to search for the nearest points for a given query point in a vector space.

It does not generate multiple trees to improve precision and recall for now. Also it does not store points in the tree. In order to use this library for searching, the points needs to be index-able, preferably in a dictionary-like structure.

## Pre-requisites

* Python 3.4+

## Example Usage

```
from lann import points_add, point_convert, forest_build, forest_query_neighbourhood, search
from uuid import uuid4
from random import random

size, dim = 25000, 5

print('generating points')
points = points_add([[random() for __ in range(dim)] for _ in range(size)],
                    dim,
                    'list',
                    [uuid4() for _ in range(size)])

print('generating query')
query = point_convert((1/3., 1/3.), 'list')

print('generating forest')
forest = forest_build(points, 25, leaf_max=5, n_jobs=4)

print('fetching neighbourhood')
neighbourhood = forest_query_neighbourhood(query, forest, threshold=0, n_jobs=20)

point_idx, distance = search(query, points, neighbourhood)[0]
```

## API

### points_add(points, dimension, ptype, identifiers=None)

* `points`: an array of points
* `dimension`: the dimension of `points`
* `ptype`: points are either `list` (a list of numeric values), or `gensim` for gensim-like corpus
* `identifiers` (optional, defaulted to `None`): an array of identifiers if applicable, otherwise a list of uuid4 is assigned to each of the point
* Returns: a dictionary that describes the points

### point_convert(vector, ptype)

* `vector`: the vector to be converted
* `ptype`: points are either `list` (a list of numeric values), or `gensim` for gensim-like corpus
* Returns: point recognized by lann


### tree_build(points, leaf_max=5)

* `points`: output of `points_add`
* `leaf_max` (optional, defaulted to `5`): maximum number of points to be stored in a leaf node
* Returns: a flattened tree structure in a dictionary form

### forest_build(points, tree_count, leaf_max=5, n_jobs=1)

* `points`: output of `points_add`
* `tree_count`: number of trees to build
* `leaf_max` (optional, defaulted to `5`): maximum number of points to be stored in a leaf node
* `n_jobs` (optional, defaulted to `1`): maximum number of processes to spawn
* Returns: a tuple of trees

### query_neighbourhood(query, tree, threshold=0, start_id='ROOT')

* `query`: the query point (converted by `point_convert`)
* `tree`: product of `tree_build`
* `threshold` (optional, defaulted to `0`): at `0`, only one leaf node is returned, the number of leaf nodes returned (hence the number of candidate points) increases as the threshold value increases.
* `start_id` (optional, defaulted to the `ROOT` node): provides the possibility to defined where to start traversing the tree.
* Returns: a list of proximity and leaf node tuples where the query point is close to (length depending on the `threshold` value)

### forest_query_neighbourhood(query, forest, threshold=0, n_jobs=1)

* `query`: the query point (converted by `point_convert`)
* `forest`: product of `forest_build`
* `threshold` (optional, defaulted to `0`): at `0`, only one leaf node is returned, the number of leaf nodes returned (hence the number of candidate points) increases as the threshold value increases.
* `n_jobs` (optional, defaulted to `1`): maximum number of threads to spawn
* Returns: a list of proximity and leaf node tuples where the query point is close to (length depending on the `threshold` value)

### search(query, points, neighbourhood)

* `query`: the query point (converted by `point_convert`)
* `points`: output of `points_add`
* `neighbourhood`: the product of `query_neighbourhood` / `forest_query_neighbourhood`
* Returns: a list of tuple, where each of them consists of the point identifier, and the corresponding distance score.

## Future plans

* Unit tests
* Further optimization (Refactor + Cython/C/C++/Golang?)
* Proper API
