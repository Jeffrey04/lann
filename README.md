# LANN

[![NeighbourSearch](https://farm2.staticflickr.com/1615/24433467801_2f1eb1d76f_z.jpg)](https://www.flickr.com/photos/jeffrey04/24433467801/in/dateposted-public/)

LANN ~~(Lame Approximate Neighbour Search)~~ is a Python re-implementation of [Annoy](https://github.com/spotify/annoy) mainly for Jeffrey04's learning (hence the 'L' in name)/experimental purpose. It does not bring anything new to the table, and is not meant for production use for now (probably wouldn't work with large scale real-life data). Like Annoy, the library can be used to search for the nearest points for a given query point in a vector space.

It does not generate multiple trees to improve precision and recall for now. Also it does not store points in the tree. In order to use this library for searching, the points needs to be index-able, preferably in a dictionary-like structure.

## Pre-requisites

* Scipy
* Python 3.4+

## Example Usage

```
from lann import points_add, tree_build, query_neighbourhood, search
from uuid import uuid4
from random import random

size, dim = 100, 2

points = points_add([[random() for __ in range(dim)] for _ in range(size)],
                    [uuid4() for _ in range(size)])

tree = tree_build(points, leaf_max=10)

query = [random() for __ in range(dim)]

neighbourhood = query_neighbourhood(query, tree, threshold=0.05)

result = search(query, points, neighbourhood)
print(result)
```

## API

### points_add(points, identifiers=None)

* `points`: an array of points
* `identifiers` (optional, defaulted to `None`): an array of identifiers if applicable, otherwise a list of integers (starting from 0) is assigned to each of the point
* Returns: a dictionary of points

### tree_build(points, leaf_max=5)

* `points`: a dictionary with points as values
* `leaf_max` (optional, defaulted to `5`): maximum number of points to be stored in a leaf node
* Returns: a flattened tree structure in a dictionary form

## query_neighbourhood(query, tree, threshold=0, start_id='ROOT')

* `query`: the query point
* `tree`: product of `tree_build`
* `threshold` (optional, defaulted to `0`): at `0`, only one leaf node is returned, the number of leaf nodes returned (hence the number of candidate points) increases as the threshold value increases.
* `start_id` (optional, defaulted to the `ROOT` node): provides the possibility to defined where to start traversing the tree.
* Returns: a list of leaf nodes where the query point is close to (depending on the `threshold` value)

### search(query, points, neighbourhood)

* `query`: the query point
* `points`: the dictionary of points
* `neighbourhood`: the product of `query_neighbourhood`
* Returns: a list of tuple, where each of them consists of the point identifier, and the corresponding distance score.

## Future plans

* Works with [gensim](https://radimrehurek.com/gensim/) corpus
* Forest building and querying
* Concurrency support
