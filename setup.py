import os
import setuptools

import cytomata


with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name=cytomata.__name__,
    version=cytomata.__version__,
    description=cytomata.__description__,
    long_description=long_description,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
    keywords=cytomata.__keywords__,
    author=cytomata.__author__,
    author_email=cytomata.__email__,
    maintainer=cytomata.__author__,
    maintainer_email=cytomata.__email__,
    url=cytomata.__website__,
    license=cytomata.__license__,
    packages=setuptools.find_packages(exclude=['tests']),
    install_requires=[
        'numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn', 'scikit-image', 'joblib',
        'opencv-python', 'natsort', 'tqdm', 'lmfit', 'pycromanager', 'filterpy'
    ]
)
