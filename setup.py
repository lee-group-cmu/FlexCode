from setuptools import setup

setup(
    name = "FlexCode",
    version = "0.1",
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=[
        "numpy",
    ],
    extras_require={
        "xgboost" : ["xgboost"],
        "sklearn" : ["sklearn"],
        "all" : ["sklearn", "xgboost"],
    },
    author="Taylor Pospisil",
    author_email="tpospisi@andrew.cmu.edu",
    description="Fits Flexible Conditional Density Estimator (FlexCode)",
    url="http://github.com/tpospisi/Flexcode",
)
