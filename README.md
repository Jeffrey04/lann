# LANN

[![NeighbourSearch](https://farm2.staticflickr.com/1615/24433467801_2f1eb1d76f_z.jpg)](https://www.flickr.com/photos/jeffrey04/24433467801/in/dateposted-public/)

LANN ~~(Lame Approximate Neighbour Search)~~ is a Python re-implementation of [Annoy](https://github.com/spotify/annoy) mainly for Jeffrey04's learning (hence the 'L' in name)/experimental purpose. It does not bring anything new to the table, and is not meant for production use for now (probably wouldn't work with large scale real-life data). Like Annoy, the library can be used to search for the nearest points for a given query point in a vector space.

It does not generate multiple trees to improve precision and recall for now. Also it does not store points in the tree. In order to use this library for searching, the points needs to be index-able, preferably in a dictionary-like structure.

## Pre-requisites

* Python 3.4+
* Graphviz
* lmdb

## Example Usage

```
from lann import points_add, point_convert, forest_build, forest_query_neighbourhood, search
from uuid import uuid4
from random import gauss

size, dim = 25000, 5

print('generating points')
pmeta, points = points_add([[gauss(0, 1) for __ in range(dim)] for _ in range(size)],
                           'pointsmdb'
                           dim,
                           'list',
                           [uuid4() for _ in range(size)])

print('generating query')
query = point_convert((1/3., 1/3.), 'list')

print('generating forest')
fmeta, forest = forest_build(points, 'forestmdb', pmeta, 25, leaf_max=5, n_jobs=4)

print('search')
idx, distance = search(query, points, pmeta, forest, fmeta, 1)[0]
```

## API

### points_add(points, filename, dimension, ptype, identifiers=None)

* `points`: an array of points
* `filename`: where to save the points
* `dimension`: the dimension of `points`
* `ptype`: points are either `list` (a list of numeric values), or `gensim` for gensim-like corpus
* `identifiers` (optional, defaulted to `None`): an array of identifiers if applicable, otherwise a list of uuid4 is assigned to each of the point
* Returns: a tuple where the first being a dictionary storing the `dimension` and the second being a lmdb environment storing identifiers as key and points as values

### point_convert(vector, ptype)

* `vector`: the vector to be converted
* `ptype`: points are either `list` (a list of numeric values), or `gensim` for gensim-like corpus
* Returns: point recognized by lann

### forest_build(points, filename, tree_count, leaf_max=5, n_jobs=1, batch_size=10000)

* `points`: output of `points_add`
* `filename`: where to save the forest
* `tree_count`: number of trees to build
* `leaf_max` (optional, defaulted to `5`): maximum number of points to be stored in a leaf node
* `n_jobs` (optional, defaulted to `1`): maximum number of processes to spawn
* `batch_size` (optional, defaulted to `10000`): only usable when `n_jobs` > 1, defines how many nodes to build in a batch
* Returns: a tuple where the first being a dictionary storing the forest's `count`, `leaf_max` and `roots` (identifiers for root nodes), and the second being an lmdb environment storing identifiers as key, and nodes as values

### search(query, points, pmeta, forest, fmeta, n, threshold=None, n_jobs=1)

* `query`: the query point (converted by `point_convert`)
* `points`: second output of `points_add`
* `pmeta`: first output of `points_add`
* `forest`: second output of `forest_build`
* `fmeta`: first output of `forest_build`
* `n`: number of points to return
* `threshold` (optional, defaulted to `None` to use a heuristic value): at `0`, only one leaf node per tree is returned, the number of leaf nodes returned (as well as the number of candidate points) increases as the threshold value increases.
* `n_jobs` (optional, defaulted to `1`): maximum number of threads to spawn
* Returns: a list of tuple, where each of them consists of a point identifier, and the corresponding distance score for the point.

## Future plans

* Unit tests
* Further optimization (Refactor + Cython/C/C++/Golang?)
* Proper API
