from setuptools import setup, find_packages

setup(
    name="market_data",
    version="0.2.13",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "numba",
        "google-cloud-storage",
        "google-cloud-bigquery",
        "google-cloud-bigquery[pandas]",
        "pytz",
    ],
    extras_require={
        'docs': [
            'sphinx>=7.2.6',
            'sphinx-rtd-theme>=2.0.0',
            'sphinx-autodoc-typehints>=1.25.2',
            'sphinx-copybutton>=0.5.2',
            'sphinx-toolbox>=3.5.0',
        ],
    },
    author="HyungJun Lim",
    author_email="sculd3@gmail.com",
    description="A package for market, feature, and machine learning data generation",
    keywords="market data, finance, feature engineering, machine learning",
    python_requires=">=3.7",
) 
