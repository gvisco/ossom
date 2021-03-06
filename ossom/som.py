import math

import scipy
from scipy.spatial.distance import euclidean


def similarity(x, y):
    return euclidean(x, y)


def neighborhood_kernel(radius, i, j, ii, jj):
    """Grid distance explained:
    http://keekerdc.com/2011/03/hexagon-grids-coordinate-systems-and-distance-calculations/

    """
    x1, y1, z1 = hex_3d_coordinates(i, j)
    x2, y2, z2 = hex_3d_coordinates(ii, jj)
    grid_distance = max(abs(x1 - x2), abs(y1 - y2), abs(z1 - z2))
    return math.exp(- grid_distance ** 2 / (2 * radius ** 2))


def hex_3d_coordinates(i, j):
    hex_i = - i / 2
    hex_j = j
    return hex_i, hex_j, -(hex_i + hex_j)


def exponential_decay(initial_value, decay, time):
    return initial_value * math.exp(- float(time) / decay)


class Som(object):
    def __init__(self, width, height, input_size, learning_rate=1.0, radius=2.0, learning_rate_decay=1.0,
                 radius_decay=1.0):
        self.lattice = [[scipy.random.uniform(-1.0, 1.0, input_size) for _ in range(width)] for _ in range(height)]
        self.lattice_width = width
        self.lattice_height = height
        self.learning_rate = learning_rate
        self.update_radius = radius
        self.learning_rate_decay = learning_rate_decay
        self.radius_decay = radius_decay

    def train(self, x, epoch):
        winner_row, winner_column, winner_weight, winner_distance = self.__compete(x)
        self.__cooperate(epoch, winner_row, winner_column, x)
        return winner_row, winner_column, winner_weight, winner_distance

    def classify(self, x):
        return self.__compete(x)

    def __compete(self, x):
        winner_row, winner_column, winner_distance = (-1, -1, float('inf'))
        for row in range(self.lattice_height):
            for column in range(self.lattice_width):
                candidate = self.lattice[row][column]
                distance = similarity(candidate, x)
                if distance < winner_distance:
                    winner_row, winner_column, winner_distance = (row, column, distance)
        return winner_row, winner_column, self.lattice[winner_row][winner_column], winner_distance

    def __cooperate(self, epoch, winner_row, winner_column, x):
        for row in range(self.lattice_height):
            for column in range(self.lattice_width):
                learning = exponential_decay(self.learning_rate, self.learning_rate_decay, epoch)
                radius = exponential_decay(self.update_radius, self.radius_decay, epoch)
                neighborhood = neighborhood_kernel(radius, row, column, winner_row, winner_column)

                neuron = self.lattice[row][column]
                weight_increment = learning * neighborhood * (x - neuron)
                neuron += weight_increment

    def __str__(self):
        return '\n'.join([''.join([str(neuron).ljust(10) for neuron in row]) for row in self.lattice])
