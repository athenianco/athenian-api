# How to debug memory leaks

The server provides two endpoints to investigate the memory usage.

### `/memory`

Prints the summary of the top memory consuming objects aggregated by type.
You can specify `?limit=N` to set the number of rows in the returned table to `N`.

### `/objgraph`

Prints the Graphviz dot of the references graph. The roots are defined by `?type=<type qualified name>`,
e.g. `?type=pandas.DataFrame`. The type names match those found in the table from `/memory`.
The depth of the graph can be changed with `?depth=N`, where `1 <= N <= 20`. The bigger the depth,
the more references you'll see but the slower generation and the harder to comprehend.
