import numpy as np

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import sklearn.ensemble
    import sklearn.neighbors
    import sklearn.multioutput
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class FlexCodeRegression(object):
    def __init__(self, max_basis):
        self.max_basis = max_basis

    def fit(self, x_train, z_basis, weight):
        pass

    def predict(self, x_new):
        pass


class NN(FlexCodeRegression):
    def __init__(self, max_basis, params):
        if not SKLEARN_AVAILABLE:
            raise Exception("NN requires sklearn to be installed")

        super(NN, self).__init__(max_basis)

        self.k = params.get("k", 5)
        self.nn = sklearn.neighbors.NearestNeighbors()

    def fit(self, x_train, z_basis, weight):
        if weight is not None:
            raise Exception("Weights not implemented for NN")
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

        super(RandomForest, self).__init__(max_basis)
        self.params = {
            'n_estimators': params.get("n_estimators", 10)
        }
        self.models = sklearn.multioutput.MultiOutputRegressor(
            sklearn.ensemble.RandomForestRegressor(**self.params), n_jobs=-1
        )

    def fit(self, x_train, z_basis, weight=None):
        self.models.fit(x_train, z_basis, sample_weight=weight)

    def predict(self, x_test):
        coefs = self.models.predict(x_test)
        return coefs


class XGBoost(FlexCodeRegression):
    def __init__(self, max_basis, params):
        if not XGBOOST_AVAILABLE:
            raise Exception("XGBoost requires xgboost to be installed")
        super(XGBoost, self).__init__(max_basis)

        self.params = {
            'max_depth': params.get("max_depth", 6),
            'learning_rate': params.get("eta", 0.3),
            'silent': params.get("silent", 1),
            'objective': params.get("objective", 'reg:linear')
        }
        self.models = sklearn.multioutput.MultiOutputRegressor(
            xgb.XGBRegressor(**self.params), n_jobs=-1
        )

    def fit(self, x_train, z_basis, weight):
        self.models.fit(x_train, z_basis, sample_weight=weight)

    def predict(self, x_test):
        coefs = self.models.predict(x_test)
        return coefs

