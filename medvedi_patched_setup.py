from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

# Define the project root (repository root)
project_root = Path(__file__).parent.resolve()

# Load version and metadata from medvedi/metadata.py
version = SourceFileLoader("version", str(project_root / "medvedi" / "metadata.py")).load_module()

# Explicitly define the Cython extensions with linking info for mimalloc
extensions = [
    Extension(
        name="medvedi.accelerators",
        sources=["medvedi/accelerators.pyx"],
        include_dirs=[np.get_include(), "medvedi"],
        libraries=["mimalloc"],
        library_dirs=[str(project_root / "medvedi")],
        extra_compile_args=[
            "-mavx2",                  # Enable AVX2 instructions for better performance
            "-ftree-vectorize",        # Enable vectorization optimizations
            "-std=c++17",              # Use C++17 standard
            "-fno-strict-aliasing",    # Relax aliasing rules which helps with template resolution
            "-Wno-strict-aliasing",    # Suppress aliasing-related warnings
            "-Wno-unused-function"     # Suppress warnings about unused template instantiations
        ],
        language="c++",
    ),
    Extension(
        name="medvedi.io",
        sources=["medvedi/io.pyx"],
        include_dirs=[np.get_include(), "medvedi"],
        libraries=["mimalloc"],
        library_dirs=[str(project_root / "medvedi")],
        extra_compile_args=[
            "-fno-strict-aliasing",    # Add these flags even for non-C++ extensions
            "-Wno-strict-aliasing"     # to maintain consistent behavior
        ],
    ),
    Extension(
        name="medvedi.native.mi_heap_destroy_stl_allocator",
        sources=["medvedi/native/mi_heap_destroy_stl_allocator.pyx"],
        include_dirs=[np.get_include(), "medvedi"],
        libraries=["mimalloc"],
        library_dirs=[str(project_root / "medvedi")],
        extra_compile_args=[
            "-std=c++17",              # C++17 for modern template features
            "-fno-strict-aliasing",    # Relax aliasing rules
            "-Wno-strict-aliasing",    # Suppress aliasing warnings
            "-Wno-unused-function"     # Suppress template-related warnings
        ],
        language="c++",
    ),
]

# Use cythonize to process the extensions
ext_modules = cythonize(
    extensions,
    force=True,
    compiler_directives={"language_level": "3"},
)

# Read the long description from the README file
with open(project_root / "README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name=version.__package__.replace(".", "-"),
    version=version.__version__,
    description=version.__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Athenian",
    author_email="vadim@athenian.co",
    url="https://github.com/athenianco/medvedi",
    download_url="https://github.com/athenianco/medvedi",
    packages=find_packages(exclude=["*tests"]),
    ext_modules=ext_modules,
    include_dirs=[np.get_include(), "medvedi"],
    install_requires=["numpy>=1.23,<1.24"],
    extras_require={
        "arrow": ["pyarrow"],
    },
    package_data={
        "": ["*.md"],
        "medvedi": ["../requirements.txt", "libmimalloc.so*", "mimalloc.h", "*.pyx"],
        "medvedi.native": ["*.pyx", "*.pxd", "*.h"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
