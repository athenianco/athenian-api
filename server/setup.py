from importlib.machinery import SourceFileLoader
import os
from pathlib import Path
import site

site.ENABLE_USER_SITE = True  # workaround https://github.com/pypa/pip/issues/7953

# The following import has to stay after imports from `setuptools`:
# - https://stackoverflow.com/questions/21594925/
#     error-each-element-of-ext-modules-option-must-be-an-extension-instance-or-2-t
from Cython.Build import cythonize  # noqa: I100, E402
import numpy as np  # noqa: I100, E402
from setuptools import find_packages, setup  # noqa: E402

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
    ext_modules=cythonize(
        [
            str(path)
            # fmt: off
            for path in (
                code_root / "internal" / "miners" / "github" / "dag_accelerated.pyx",
                code_root / "internal" / "miners" / "github" / "check_run_accelerated.pyx",
                code_root / "internal" / "miners" / "github" / "deployment_accelerated.pyx",
                code_root / "internal" / "miners" / "github" / "precomputed_prs" / "utils_accelerated.pyx",  # noqa
                code_root / "internal" / "miners" / "types_accelerated.pyx",
                code_root / "internal" / "features" / "github" / "check_run_metrics_accelerated.pyx",  # noqa
                code_root / "internal" / "features" / "github" / "pull_request_filter_accelerated.pyx",  # noqa
                code_root / "internal" / "logical_accelerated.pyx",
                code_root / "sorted_intersection.pyx",
                code_root / "to_object_arrays.pyx",
                code_root / "unordered_unique.pyx",
                code_root / "pandas_io.pyx",
                code_root / "sentry_native.pyx",
                code_root / "models" / "sql_builders.pyx",
                code_root / "models" / "web_model_io.pyx",
                code_root / "native" / "mi_heap_destroy_stl_allocator.pyx",
            )
            # fmt: on
        ],
    ),
    include_dirs=[np.get_include()],
    namespace_packages=["athenian"],
    keywords=[],
    install_requires=[],  # see requirements.txt
    tests_require=[],  # see requirements-test.txt
    package_data={
        "": ["*.md", "*.jinja2", "*.mako"],
        "athenian": ["../requirements.txt"],
        "athenian.api": ["openapi"],
        "athenian.api.align": ["spec.gql"],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
