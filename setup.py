import os
import setuptools

import cytomata


def read(*names):
    values = {}
    extensions = ['.txt', '.rst', '.md']
    for name in names:
        value = ''
        for extension in extensions:
            filename = name + extension
            if os.path.isfile(filename):
                value = open(filename).read()
                break
        values[name] = value
    return values


setuptools.setup(
    name=cytomata.__name__,
    version=cytomata.__version__,
    description=cytomata.__description__,
    long_description="""%(README)s""" % read('README'),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.8",
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
        'numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn', 'scikit-image',
        'opencv-python', 'natsort', 'tqdm', 'lmfit', 'pycromanager', 'filterpy'
    ]
)
