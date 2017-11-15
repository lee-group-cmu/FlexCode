import numpy as np

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import sklearn.ensemble
    import sklearn.neighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class FlexCodeRegression(object):
    def __init__(self, max_basis):
        self.max_basis = max_basis

    def fit(self, x_train, z_basis):
        pass

    def predict(self, x_new):
        pass


class NN(FlexCodeRegression):
    def __init__(self, max_basis, params):
        if not SKLEARN_AVAILABLE:
            raise Exception("NN requires sklearn to be installed")

        super().__init__(max_basis)

        self.k = params.get("k", 5)
        self.nn = sklearn.neighbors.NearestNeighbors()

    def fit(self, x_train, z_basis):
        self.nn.fit(x_train)
        self.z_basis = z_basis

    def predict(self, x_test):
        n_obs = x_test.shape[0]
        coefs = np.empty((n_obs, self.max_basis))
        neighbors = self.nn.kneighbors(x_test, self.k, False)
        for ii in range(n_obs):
            coefs[ii, :] = np.mean(self.z_basis[neighbors[ii], :], 0)
        return coefs

class RandomForest(FlexCodeRegression):
    def __init__(self, max_basis, params):
        if not SKLEARN_AVAILABLE:
            raise Exception("RandomForest requires sklearn to be installed")

        super().__init__(max_basis)
        self.models = [sklearn.ensemble.RandomForestRegressor()
                       for ii in range(self.max_basis)]

    def fit(self, x_train, z_basis):
        for ii in range(self.max_basis):
            self.models[ii].fit(x_train, z_basis[:, ii])

    def predict(self, x_test):
        n_obs = x_test.shape[0]
        coefs = np.empty((n_obs, self.max_basis))
        for ii in range(self.max_basis):
            coefs[:, ii] = self.models[ii].predict(x_test)
        return coefs

class XGBoost(FlexCodeRegression):
    def __init__(self, max_basis, params):
        super().__init__(max_basis)

        self.params = {'max_depth' : params.get("max_depth", 6),
                       'eta' : params.get("eta", 0.3),
                       'silent' : params.get("silent", 1),
                       'objective' : 'reg:linear'}
        self.num_round = params.get("num_round", 500)

    def fit(self, x_train, z_basis):
        self.models = []
        for ii in range(self.max_basis):
            dtrain = xgb.DMatrix(x_train, label=z_basis[:,ii])
            self.models.append(xgb.train(self.params, dtrain, self.num_round))

    def predict(self, x_test):
        n_obs = x_test.shape[0]
        coefs = np.empty((n_obs, self.max_basis))
        dtest = xgb.DMatrix(x_test)
        for ii in range(self.max_basis):
            coefs[:, ii] = self.models[ii].predict(dtest)
        return coefs
