from importlib.machinery import SourceFileLoader
import os
from pathlib import Path

from setuptools import find_packages, setup
# The following import has to stay after imports from `setuptools`:
# - https://stackoverflow.com/questions/21594925/
#     error-each-element-of-ext-modules-option-must-be-an-extension-instance-or-2-t
from Cython.Build import cythonize  # noqa: I100

project_root = Path(__file__).parent
code_root = project_root / "athenian" / "api"
os.chdir(str(project_root))
version = SourceFileLoader("version", str(code_root / "metadata.py")).load_module()

with open(project_root.parent / "README.md", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name=version.__package__.replace(".", "-"),
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
    ext_modules=cythonize(str(
        code_root / "controllers" / "miners" / "github" / "dag_accelerated.pyx")),
    namespace_packages=["athenian"],
    keywords=[],
    install_requires=[],
    tests_require=[],
    package_data={
        "": ["*.md", "*.jinja2", "*.mako"],
        "athenian": ["../requirements.txt"],
        "athenian.api": ["openapi"],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
