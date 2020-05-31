from sklearn.svm import SVC
import numpy as np
import data as data
import matplotlib.pyplot as plt


class KSVMwrap:

    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.svm = SVC(C=param_svm_c, gamma=param_svm_gamma, probability=True)
        self.svm.fit(X, Y_)

    def predict(self, X):
        return self.svm.predict(X)

    def get_scores(self, X):
        return self.svm.predict_proba(X)

    def support(self):
        return self.svm.support_


if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(69)

    # instanciraj podatke X i labele Yoh_
    X, Y_ = data.sample_gmm_2d(6, 2, 10)

    # definiraj model:
    model = KSVMwrap(X, Y_)

    # dohvati vjerojatnosti na skupu za učenje
    probs = model.get_scores(X)
    Y = model.predict(X)

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    print(accuracy, recall, precision)

    # iscrtaj rezultate, decizijsku plohu
    decfun = lambda X: model.get_scores(X)[:, 1]
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    data.graph_data(X, Y_, Y, special=[model.support()])

    plt.show()
