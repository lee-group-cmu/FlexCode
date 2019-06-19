import numpy as np
from .helpers import params_dict_optim_decision, params_name_format

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import sklearn.ensemble
    import sklearn.neighbors
    import sklearn.multioutput
    import sklearn.model_selection
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

        # Historically, we have used 'k' to indicate the number of neighbors, so
        # this just puts the right notation for KNeighborsRegressor
        if 'k' in params:
            params['n_neighbors'] = params['k']
            del params['k']
        params_opt, opt_flag = params_dict_optim_decision(params, multi_output=True)
        self.params = params_opt
        self.models = None if opt_flag else sklearn.multioutput.MultiOutputRegressor(
            sklearn.neighbors.KNeighborsRegressor(**self.params), n_jobs=-1
        )

    def fit(self, x_train, z_basis, weight):
        if weight is not None:
            raise Exception("Weights not implemented for NN")

        if self.models is None:
            self.cv_optim(x_train, z_basis)

        self.models.fit(x_train, z_basis)

    def cv_optim(self, x_train, z_basis):
        nn_obj = sklearn.multioutput.MultiOutputRegressor(
            sklearn.neighbors.KNeighborsRegressor(), n_jobs=-1
        )
        clf = sklearn.model_selection.GridSearchCV(
            nn_obj, self.params, cv=5, scoring='neg_mean_squared_error', verbose=2
        )
        clf.fit(x_train, z_basis)

        self.params = params_name_format(clf.best_params_, str_rem='estimator__')
        self.models = sklearn.multioutput.MultiOutputRegressor(
            sklearn.neighbors.KNeighborsRegressor(**self.params), n_jobs=-1
        )

    def predict(self, x_test):
        coefs = self.models.predict(x_test)
        return coefs


class RandomForest(FlexCodeRegression):
    def __init__(self, max_basis, params):
        if not SKLEARN_AVAILABLE:
            raise Exception("RandomForest requires sklearn to be installed")

        super(RandomForest, self).__init__(max_basis)

        params_opt, opt_flag = params_dict_optim_decision(params, multi_output=True)
        self.params = params_opt
        self.models = None if opt_flag else sklearn.multioutput.MultiOutputRegressor(
            sklearn.ensemble.RandomForestRegressor(**self.params), n_jobs=-1
        )

    def fit(self, x_train, z_basis, weight=None):
        if self.models is None:
            self.cv_optim(x_train, z_basis, weight)

        self.models.fit(x_train, z_basis, sample_weight=weight)

    def cv_optim(self, x_train, z_basis, weight=None):
        rf_obj = sklearn.multioutput.MultiOutputRegressor(
            sklearn.ensemble.RandomForestRegressor(), n_jobs=-1
        )
        clf = sklearn.model_selection.GridSearchCV(
            rf_obj, self.params, cv=5, scoring='neg_mean_squared_error', verbose=2,
            fit_params={'sample_weight': weight}
        )
        clf.fit(x_train, z_basis)

        self.params = params_name_format(clf.best_params_, str_rem='estimator__')
        self.models = sklearn.multioutput.MultiOutputRegressor(
            sklearn.ensemble.RandomForestRegressor(**self.params), n_jobs=-1
        )

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

