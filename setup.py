from setuptools import setup

setup(name = "flexcode",
      version = "0.2",
      license="GPL",
      description="Fits Flexible Conditional Density Estimator (FlexCode)",
      author="Taylor Pospisil, Nic Dalmasso",
      maintainer="Taylor Pospisil",
      author_email="tpospisi@andrew.cmu.edu",
      url="http://github.com/tpospisi/Flexcode",
      package_dir={"":"src"},
      packages=["flexcode"],
      install_requires=["numpy", "pywavelets"],
      setup_requires=["pytest-runner"],
      tests_require=["pytest", "scikit-learn", "xgboost"],
      zip_safe=True,
      extras_require={
          "xgboost" : ["xgboost"],
          "scikit-learn" : ["scikit-learn>=0.18"],
        "all" : ["scikit-learn", "xgboost"],
      },
)
