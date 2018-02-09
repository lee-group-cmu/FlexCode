Implementation of Flexible Conditional Density Estimator (FlexCode) in Python. See Izbicki, R.; Lee, A.B. [Converting High-Dimensional Regression to High-Dimensional Conditional Density Estimation](https://projecteuclid.org/euclid.ejs/1499133755). Electronic Journal of Statistics, 2017 for details. Port of the original [R package](https://github.com/rizbicki/FlexCoDE).


# FlexCode

FlexCode is a general-purpose method for converting any conditional mean point estimator of \(z\) to a conditional {\em density} estimator \(f(z \vert x)\), where \(x\) represents the covariates. The key idea is to expand the unknown function \(f(z \vert x)\) in an orthonormal basis \(\{\phi_i(z)\}_{i}\):

\[ f(z|x)=\sum_{i}\beta_{i }(x)\phi_i(z) \]

By the orthogonality property, the expansion coefficients are just conditional means

\[ \beta_{i }(x) = \mathbb{E}\left[\phi_i(z)|x\right] \equiv \int f(z|x) \phi_i(z) dz \]

where the coefficients are estimated from data by an appropriate regression method.


# Installation

```shell
git clone https://github.com/tpospisi/FlexCode.git
pip install FlexCode[all]
```

Flexcode handles a number of regression models; if you wish to avoid installing all dependencies you can specify your desired regression methods using the optional requires in brackets. Targets include

-   xgboost
-   sklearn (for nearest neighbor regression, random forests)


# A simple example

```python
import numpy as np
import scipy.stats
import flexcode
from flexcode.regression_models import NN
import matplotlib.pyplot as plt

# Generate data p(z | x) = N(x, 1)
def generate_data(n_draws):
    x = np.random.normal(0, 1, n_draws)
    z = np.random.normal(x, 1, n_draws)
    return x.reshape((len(x), 1)), z

x_train, z_train = generate_data(10000)
x_validation, z_validation = generate_data(10000)
x_test, z_test = generate_data(10000)

# Parameterize model
model = flexcode.FlexCodeModel(NN, max_basis=31, basis_system="cosine",
                               regression_params={"k":20})

# Fit and tune model
model.fit(x_train, z_train)
model.tune(x_validation, z_validation)

# Estimate CDE loss
print(model.estimate_error(x_test, z_test))

# Calculate conditional density estimates
cdes, z_grid = model.predict(x_test, n_grid=200)

for ii in range(10):
    true_density = scipy.stats.norm.pdf(z_grid, x_test[ii], 1)
    plt.plot(z_grid, cdes[ii, :])
    plt.plot(z_grid, true_density, color = "green")
    plt.axvline(x=z_test[ii], color="red")
    plt.show()

```


# FlexZBoost Buzzard Data

One particular realization of the FlexCode algorithm is FlexZBoost which uses XGBoost as the regression method. We apply this method to photo-z estimation in the LSST DESC DC-1. For members of the LSST DESC, you can find information on obtaining the data [here](https://confluence.slac.stanford.edu/pages/viewpage.action?spaceKey=LSSTDESC&title=DC1+resources).

```python
import numpy as np
import pandas as pd
import flexcode
from flexcode.regression_models import XGBoost

# Read in data
def process_data(feature_file, has_z=False):
    """Processes buzzard data"""
    df = pd.read_table(feature_file, sep=" ")
    df["ug"] = df["u"] - df["g"]

    df.assign(ug = df.u - df.g,
              gr = df.g - df.r,
              ri = df.r - df.i,
              iz = df.i - df.z,
              zy = df.z - df.y,
              ug_err = np.sqrt(df['u.err'] ** 2 + df['g.err'] ** 2),
              gr_err = np.sqrt(df['g.err'] ** 2 + df['r.err'] ** 2),
              ri_err = np.sqrt(df['r.err'] ** 2 + df['i.err'] ** 2),
              iz_err = np.sqrt(df['i.err'] ** 2 + df['z.err'] ** 2),
              zy_err = np.sqrt(df['z.err'] ** 2 + df['y.err'] ** 2))

    if has_z:
        z = df.redshift.as_matrix()
        df.drop('redshift', axis=1, inplace=True)
    else:
        z = None

    return df.as_matrix(), z

x_data, z_data = process_data('buzzard_spec_witherrors_mass.txt', has_z=True)
x_test, _ = process_data('buzzard_phot_witherrors_mass.txt', has_z=False)

n_obs = x_data.shape[0]
n_train = round(n_obs * 0.8)
n_validation = n_obs - n_train

perm = np.random.permutation(n_obs)
x_train = x_data[perm[:n_train], :]
z_train = z_data[perm[:n_train]]
x_validation = x_data[perm[n_train:]]
z_validation = z_data[perm[n_train:]]

# Fit the model
model = flexcode.FlexCodeModel(XGBoost, max_basis=40, basis_system='cosine',
                               regression_params={"num_round":2000})
model.fit(x_train, z_train)
model.tune(x_validation, z_validation)

# Make predictions
cdes, z_grid = model.predict(x_test, n_grid=200)

```