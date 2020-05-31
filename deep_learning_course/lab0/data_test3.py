import numpy as np
import data
import matplotlib.pyplot as plt

if __name__ == "__main__":
    np.random.seed(100)

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(2, 100)

    # get the class predictions
    Y = data.myDummyDecision(X) > 0.5

    # graph the data points
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(data.myDummyDecision, bbox, offset=0.4)
    data.graph_data(X, Y_, Y)


    # show the results
    plt.show()