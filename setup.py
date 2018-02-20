from setuptools import setup

setup(name = "flexcode",
      version = "0.1.1",
      license="GPL",
      description="Fits Flexible Conditional Density Estimator (FlexCode)",
      author="Taylor Pospisil",
      author_email="tpospisi@andrew.cmu.edu",
      maintainer="tpospisi@andrew.cmu.edu",
      url="http://github.com/tpospisi/Flexcode",
      package_dir={"":"src"},
      packages=["flexcode"],
      install_requires=["numpy", "pywavelets"],
      setup_requires=["pytest-runner"],
      tests_require=["pytest"],
      zip_safe=True,
      extras_require={
          "xgboost" : ["xgboost"],
          "sklearn" : ["sklearn"],
        "all" : ["sklearn", "xgboost"],
      },
)
