import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from torchvision.datasets import MNIST
from sklearn.decomposition import PCA


class EM_GMM:
    def __init__(self, k, n_iter, tol, seed):
        self.k = k
        self.n_iter = n_iter
        self.tol = tol
        self.seed = seed

    def fit(self, X):
        r, c = X.shape
        self.resp = np.zeros((r, self.k)) # responsibility vector

        # initialize mu, pi
        np.random.seed(self.seed)
        chosen = np.random.choice(r, self.k, replace=False)
        self.means = X[chosen]
        self.weights = np.full(self.k, 1/self.k)

        shape = self.k, c, c
        self.covs = np.full(shape, np.cov(X, rowvar=False))

        log_likelihood = 0
        self.converged = False
        self.log_likelihood_trace = []

        for i in range(self.n_iter):
            log_likelihood_new = self.estep(X)
            self.mstep(X)

            if abs(log_likelihood_new-log_likelihood) <= self.tol * log_likelihood_new:
                self.converged = True
                break

            log_likelihood = log_likelihood_new
            self.log_likelihood_trace.append(log_likelihood)
        return self

    def estep(self, X):
        self.compute_log_likelihood(X)
        log_likelihood = -np.sum(np.log(np.sum(self.resp, axis=1)))

        self.resp = self.resp/self.resp.sum(axis=1, keepdims=1)
        return log_likelihood

    def compute_log_likelihood(self, X):
        for ki in range(self.k):
            prior = self.weights[ki]
            likelihood = multivariate_normal(self.means[ki], self.covs[ki]).pdf(X)
            self.resp[:, ki] = prior*likelihood
        return self

    def mstep(self, X):
        """M-steps: update the parameters"""

        resp_weights = self.resp.sum(axis=0)

        # weights
        self.weights = resp_weights/X.shape[0]

        # means
        weighted_sum = np.dot(self.resp.T, X)
        self.means = weighted_sum/resp_weights.reshape(-1,1)

        # covariance
        for kj in range(self.k):
            diff = (X - self.means[kj]).T
            weighted_sum = np.dot(self.resp[:, kj] * diff, diff.T)
            self.covs[kj] = weighted_sum/resp_weights[kj]
        return self


def predict(test_data, models, labelMap):
    label = list()
    for i in range(test_data.shape[0]):
        prob_sum = []
        for model in models:
            sum = 0
            for ki in range(model.k):
                pi = model.weights[ki]
                pdf = multivariate_normal(model.means[ki], model.covs[ki]).pdf(test_data[i])
                sum += pi*pdf
            prob_sum.append(sum)
        label.append(labelMap[prob_sum.index(max(prob_sum))])

    return label


def MNIST_pipe(k):
    train_dataset = MNIST(root='../data', train=True, download=False)
    val_dataset = MNIST(root='../data', train=False, download=False)

    train_data = train_dataset.data
    pca = PCA(n_components=20)

    # prepare train data using pca
    train_data = pca.fit_transform(train_data.reshape(train_data.shape[0], -1))

    # prepare the label
    labels = train_dataset.targets
    labels = labels.cpu().detach().numpy()
    label_set = set(labels)

    # train the GMM as classifier
    models = list()
    for i,label in enumerate(label_set):
        x_subset = train_data[np.in1d(labels, label)]
        models.append(EM_GMM(k=k, n_iter=500, tol=1e-5, seed=24))
        print(i)
        models[i].fit(x_subset)

    # test the GMM classifier, get labels
    test_data = val_dataset.data
    test_data = pca.transform(test_data.reshape(test_data.shape[0], -1))
    test_labels = val_dataset.targets
    test_labels = test_labels.cpu().detach().numpy()

    pred = predict(test_data, models, list(label_set))

    corrects = 0
    for i in range(len(pred)):
        if pred[i] == test_labels[i]:
            corrects += 1

    return corrects/len(test_labels)


# Load data
X = np.loadtxt('gmm_dataset.csv', delimiter=",")

k = np.arange(1, 11)
ll_final = list()
for ki in k:
    gmm = EM_GMM(k=ki, n_iter=500, tol=1e-5, seed=24)
    gmm.fit(X)
    ll_final.append(gmm.log_likelihood_trace[-1])

plt.plot(k, ll_final)
plt.xlabel('k')
plt.ylabel('Negative log-likelihood')
plt.savefig('ex1_2.png')
plt.show()

gmm_5 = EM_GMM(k=5, n_iter=500, tol=1e-5, seed=24)
gmm_5.fit(X)

mixing_weights = gmm_5.weights
mean_vectors = gmm_5.means
diagnoals_vector = gmm_5.covs

sort_index = mixing_weights.argsort()
mixing_weights = mixing_weights[sort_index]
mean_vectors = mean_vectors[sort_index]
diagnoals_vector = diagnoals_vector[sort_index]

# MNIST
print(MNIST_pipe(5))

