from setuptools import setup, find_packages

setup(
    name="bulk",
    version="0.3.0",
    packages=find_packages(),
    install_requires=["radicli>=0.0.8,<0.1.0", "bokeh", "pandas>=1.0.0", "wasabi>=0.9.1", "pyarrow", "tqdm"],
    extras_require={
        "dev": ["pytest-playwright==0.3.0"],
    },
    entry_points={
        'console_scripts': [
            'bulk = bulk.__main__:cli.run',
        ],
    },
)
