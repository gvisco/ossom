from ossom.som import Som
import scipy

if __name__ == '__main__':

    dataset = scipy.random.uniform(-.9, .9, (50, 2))

    epochs = 10

    l_rate = 0.5
    rad = 1.0
    l_rate_decay = 10.0
    r_decay = 10.0

    som = Som(20, 20, 2, learning_rate=l_rate, radius=rad,
              learning_rate_decay=l_rate_decay,
              radius_decay=r_decay)

    print(som)

    # train
    for epoch in range(1, epochs):
        print('--- Epoch %d ---' % epoch)
        for x in dataset:
            som.train(x, epoch)

    # classify
    clusters = {}
    cells = {}
    for i, x in enumerate(dataset):
        winner_row, winner_column, winner_weight, winner_distance = som.classify(x)
        if (winner_row, winner_column) not in clusters:
            clusters[(winner_row, winner_column)] = []
            cells[(winner_row, winner_column)] = winner_weight
        clusters[(winner_row, winner_column)].append([i, winner_distance])

    # print result
    print('===============')
    for k, v in clusters.items():
        print('Cell %s: %s' % (str(k), (str(cells[k]))))
        for value in v:
            print('\titem: %i - distance: %f' % tuple(value))
