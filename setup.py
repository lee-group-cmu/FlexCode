from setuptools import setup

setup(name = "flexcode",
      version = "0.1.3",
      license="GPL",
      description="Fits Flexible Conditional Density Estimator (FlexCode)",
      author="Taylor Pospisil",
      maintainer="Taylor Pospisil",
      author_email="tpospisi@andrew.cmu.edu",
      url="http://github.com/tpospisi/Flexcode",
      package_dir={"":"src"},
      packages=["flexcode"],
      install_requires=["numpy", "pywavelets"],
      setup_requires=["pytest-runner"],
      tests_require=["pytest", "sklearn"],
      zip_safe=True,
      extras_require={
          "xgboost" : ["xgboost"],
          "sklearn" : ["sklearn>=0.18"],
        "all" : ["sklearn", "xgboost"],
      },
)
