# Ordinary Simple Self Organizing Map

A simple implementation of [Self Organizing Map](https://en.wikipedia.org/wiki/Self-organizing_map) written in Python.

## Basic usage

Get the dataset as a [Scipy](https://www.scipy.org/) array NxM, where N is the number of samples and M is the number of features for each sample.
```python
dataset = import_data(N, M)
```

Import Som class
```python
from ossom.som import Som
```

Instantiate a new SOM, specifying the width and the height of the grid and the input size.
```python
som = Som(W, H, M)
```
Newly created SOMs can be tweaked by setting parameters like learning rate, radius of the neighborhood function etc, can be set.

Train the model, iterating through the samples for E epochs:
```python
for epoch in range(1, E):
    for x in dataset:
        som.train(x, epoch)
```

Use the trained map to classify data: given an input, find the unit of the map that better fits with it and get back its value -- or weight -- and its position within the grid.
```python
row, column, weight, distance = som.classify(x)

```

#### Full sample code
A full running sample is available [here](test-som.py).
It does use a SOM to classify randomly generated data... not very useful indeed.

## References
* [Scholarpedia article on SOM](http://www.scholarpedia.org/article/Kohonen_network)
* [Juha Vesanto, SOM-Based Data Visualization Methods](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=85500F9EEB8984A38B4A00A8A180ADF2?doi=10.1.1.26.8386&rep=rep1&type=pdf)
* [Grid distance explained](http://keekerdc.com/2011/03/hexagon-grids-coordinate-systems-and-distance-calculations/)