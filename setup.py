import setuptools

setuptools.setup(
    author="Victor Kulikov, Victor Yurchenko, Victor Lempitsky",
    author_email="v.kulikov@skoltech.ru",
    install_requires=[
        "numpy",
        "scikit-image",
        "matplotlib",
        "torch"
    ],
    packages=setuptools.find_packages(
        exclude=["cvppp"]
    ),
    license="GPL-v3",
    name="deepcoloring",
    url="https://github.com/kulikovv/DeepColoring",
    version="0.0.2"
)