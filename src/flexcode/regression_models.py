import numpy as np

from .helpers import params_dict_optim_decision, params_name_format

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import sklearn.ensemble
    import sklearn.linear_model
    import sklearn.model_selection
    import sklearn.multioutput
    import sklearn.neighbors

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
    def __init__(self, max_basis, params, *args, **kwargs):
        if not SKLEARN_AVAILABLE:
            raise Exception("NN requires scikit-learn to be installed")

        super(NN, self).__init__(max_basis)

        # Historically, we have used 'k' to indicate the number of neighbors, so
        # this just puts the right notation for KNeighborsRegressor
        if "k" in params:
            params["n_neighbors"] = params["k"]
            del params["k"]
        params_opt, opt_flag = params_dict_optim_decision(params, multi_output=True)
        self.params = params_opt
        self.models = (
            None
            if opt_flag
            else sklearn.multioutput.MultiOutputRegressor(
                sklearn.neighbors.KNeighborsRegressor(**self.params), n_jobs=-1
            )
        )

    def fit(self, x_train, z_basis, weight):
        if weight is not None:
            raise Exception("Weights not implemented for NN")

        if self.models is None:
            self.cv_optim(x_train, z_basis)

        self.models.fit(x_train, z_basis)

    def cv_optim(self, x_train, z_basis):
        nn_obj = sklearn.multioutput.MultiOutputRegressor(sklearn.neighbors.KNeighborsRegressor(), n_jobs=-1)
        clf = sklearn.model_selection.GridSearchCV(
            nn_obj, self.params, cv=5, scoring="neg_mean_squared_error", verbose=2
        )
        clf.fit(x_train, z_basis)

        self.params = params_name_format(clf.best_params_, str_rem="estimator__")
        self.models = sklearn.multioutput.MultiOutputRegressor(
            sklearn.neighbors.KNeighborsRegressor(**self.params), n_jobs=-1
        )

    def predict(self, x_test):
        coefs = self.models.predict(x_test)
        return coefs


class RandomForest(FlexCodeRegression):
    def __init__(self, max_basis, params, *args, **kwargs):
        if not SKLEARN_AVAILABLE:
            raise Exception("RandomForest requires scikit-learn to be installed")

        super(RandomForest, self).__init__(max_basis)

        params_opt, opt_flag = params_dict_optim_decision(params, multi_output=True)
        self.params = params_opt
        self.models = (
            None
            if opt_flag
            else sklearn.multioutput.MultiOutputRegressor(
                sklearn.ensemble.RandomForestRegressor(**self.params), n_jobs=-1
            )
        )

    def fit(self, x_train, z_basis, weight=None):
        if self.models is None:
            self.cv_optim(x_train, z_basis, weight)

        self.models.fit(x_train, z_basis, sample_weight=weight)

    def cv_optim(self, x_train, z_basis, weight=None):
        rf_obj = sklearn.multioutput.MultiOutputRegressor(sklearn.ensemble.RandomForestRegressor(), n_jobs=-1)
        clf = sklearn.model_selection.GridSearchCV(
            rf_obj, self.params, cv=5, scoring="neg_mean_squared_error", verbose=2
        )
        clf.fit(x_train, z_basis, sample_weight=weight)

        self.params = params_name_format(clf.best_params_, str_rem="estimator__")
        self.models = sklearn.multioutput.MultiOutputRegressor(
            sklearn.ensemble.RandomForestRegressor(**self.params), n_jobs=-1
        )

    def predict(self, x_test):
        coefs = self.models.predict(x_test)
        return coefs


class XGBoost(FlexCodeRegression):
    def __init__(self, max_basis, params, *args, **kwargs):
        if not XGBOOST_AVAILABLE:
            raise Exception("XGBoost requires xgboost to be installed")
        super(XGBoost, self).__init__(max_basis)

        # Historically, people have used `eta` for `learning_rate` - taking that
        # into account
        if "eta" in params:
            params["learning_rate"] = params["eta"]
            del params["eta"]

        # Also, set the default values if not passed
        params["max_depth"] = params.get("max_depth", 6)
        params["learning_rate"] = params.get("learning_rate", 0.3)
        params["silent"] = params.get("silent", 1)
        params["objective"] = params.get("objective", "reg:linear")

        params_opt, opt_flag = params_dict_optim_decision(params, multi_output=True)
        self.params = params_opt
        self.models = (
            None
            if opt_flag
            else sklearn.multioutput.MultiOutputRegressor(xgb.XGBRegressor(**self.params), n_jobs=-1)
        )

    def fit(self, x_train, z_basis, weight=None):
        if self.models is None:
            self.cv_optim(x_train, z_basis, weight)

        self.models.fit(x_train, z_basis, sample_weight=weight)

    def cv_optim(self, x_train, z_basis, weight=None):
        xgb_obj = sklearn.multioutput.MultiOutputRegressor(xgb.XGBRegressor(), n_jobs=-1)
        clf = sklearn.model_selection.GridSearchCV(
            xgb_obj, self.params, cv=5, scoring="neg_mean_squared_error", verbose=2
        )
        clf.fit(x_train, z_basis, sample_weight=weight)

        self.params = params_name_format(clf.best_params_, str_rem="estimator__")
        self.models = sklearn.multioutput.MultiOutputRegressor(xgb.XGBRegressor(**self.params), n_jobs=-1)

    def predict(self, x_test):
        coefs = self.models.predict(x_test)
        return coefs


class Lasso(FlexCodeRegression):
    def __init__(self, max_basis, params, *args, **kwargs):
        if not SKLEARN_AVAILABLE:
            raise Exception("Lasso requires scikit-learn to be installed")
        super(Lasso, self).__init__(max_basis)

        # Also, set the default values if not passed
        params["alpha"] = params.get("alpha", 1.0)
        params["l1_ratio"] = params.get("l1_ratio", 1.0)

        params_opt, opt_flag = params_dict_optim_decision(params, multi_output=True)
        self.params = params_opt
        self.models = (
            None
            if opt_flag
            else sklearn.multioutput.MultiOutputRegressor(
                sklearn.linear_model.ElasticNet(**self.params), n_jobs=-1
            )
        )

    def fit(self, x_train, z_basis, weight=None):
        if weight is not None:
            raise ValueError(
                "Weights are not supported in the ElasticNet/Lasso " "implementation in scikit-learn."
            )

        if self.models is None:
            self.cv_optim(x_train, z_basis)

        self.models.fit(x_train, z_basis)

    def cv_optim(self, x_train, z_basis):
        lasso_obj = sklearn.multioutput.MultiOutputRegressor(sklearn.linear_model.ElasticNet(), n_jobs=-1)
        clf = sklearn.model_selection.GridSearchCV(
            lasso_obj, self.params, cv=5, scoring="neg_mean_squared_error", verbose=2
        )
        clf.fit(x_train, z_basis)

        self.params = params_name_format(clf.best_params_, str_rem="estimator__")
        self.models = sklearn.multioutput.MultiOutputRegressor(
            sklearn.linear_model.ElasticNet(**self.params), n_jobs=-1
        )

    def predict(self, x_test):
        coefs = self.models.predict(x_test)
        return coefs


class CustomModel(FlexCodeRegression):
    def __init__(self, max_basis, params, custom_model, *args, **kwargs):
        if not SKLEARN_AVAILABLE:
            raise Exception("Custom class requires scikit-learn to be installed")
        super(CustomModel, self).__init__(max_basis)

        params_opt, opt_flag = params_dict_optim_decision(params, multi_output=True)
        self.params = params_opt
        self.base_model = custom_model
        self.models = (
            None
            if opt_flag
            else sklearn.multioutput.MultiOutputRegressor(self.base_model(**self.params), n_jobs=-1)
        )

    def fit(self, x_train, z_basis, weight=None):
        # Given it's a custom class, work would need to be done
        # for sample weights - for now this is not implemented.
        if weight:
            raise NotImplementedError("Weights for custom class not implemented.")

        if self.models is None:
            self.cv_optim(x_train, z_basis)

        self.models.fit(x_train, z_basis)

    def cv_optim(self, x_train, z_basis):
        custom_obj = sklearn.multioutput.MultiOutputRegressor(self.base_model(), n_jobs=-1)
        clf = sklearn.model_selection.GridSearchCV(
            custom_obj, self.params, cv=5, scoring="neg_mean_squared_error", verbose=2
        )
        clf.fit(x_train, z_basis)

        self.params = params_name_format(clf.best_params_, str_rem="estimator__")
        self.models = sklearn.multioutput.MultiOutputRegressor(self.base_model(**self.params), n_jobs=-1)

    def predict(self, x_test):
        coefs = self.models.predict(x_test)
        return coefs
