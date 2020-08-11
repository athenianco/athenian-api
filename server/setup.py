from importlib.machinery import SourceFileLoader
import os

from setuptools import find_packages, setup
# The following import has to stay after imports from `setuptools`:
# - https://stackoverflow.com/questions/21594925/
#     error-each-element-of-ext-modules-option-must-be-an-extension-instance-or-2-t
from Cython.Build import cythonize  # noqa: I100

version = SourceFileLoader("version", "athenian/api/metadata.py").load_module()

with open(os.path.join(os.path.dirname(__file__), "../README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name=version.__package__,
    description=version.__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=version.__version__,
    license="Proprietary",
    author="Athenian",
    author_email="vadim@athenian.co",
    url="https://github.com/athenian/athenian-api",
    download_url="https://github.com/athenian/athenian-api",
    packages=find_packages(exclude=["tests"]),
    ext_modules=cythonize("athenian/api/controllers/miners/github/release_accelerated.pyx"),
    namespace_packages=["athenian"],
    keywords=[],
    install_requires=[],
    tests_require=[],
    package_data={"": ["requirements.txt"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
