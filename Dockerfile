FROM ubuntu:20.04

ENV BROWSER=/browser \
    LC_ALL=en_US.UTF-8 \
    SETUPTOOLS_USE_DISTUTILS=stdlib \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Madrid \
    PYTHON_TARGET_VERSION="3.8.5 1~20.04.3"

RUN echo '#!/bin/bash\n\
\n\
echo\n\
echo "  $@"\n\
echo\n' > /browser && \
    chmod +x /browser

# matches our production except -march=haswell, we have to downgrade -march because of GHA
ENV OPT="-fno-semantic-interposition -march=haswell -mabm -maes -mno-pku -mno-sgx --param l1-cache-line-size=64 --param l1-cache-size=32 --param l2-cache-size=33792"

# runtime environment
RUN echo 'deb-src http://archive.ubuntu.com/ubuntu focal-updates main' >>/etc/apt/sources.list && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install --no-install-suggests --no-install-recommends -y \
      ca-certificates apt-utils wget locales python3-dev python3-distutils dpkg-dev devscripts && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    mkdir /cpython && \
    cd /cpython && \
    apt-get source python3.8 && \
    apt-get -s build-dep python3.8 | grep "Inst " | cut -d" " -f2 | sort | tr '\n' ' ' >build_bloat && \
    DEBIAN_FRONTEND="noninteractive" TZ="Europe/Madrid" apt-get build-dep -y python3.8 && \
    wget -O - https://bootstrap.pypa.io/get-pip.py | python3 && \
    cd python3.8* && \
    sed -i 's/__main__/__skip__/g' Tools/scripts/run_tests.py && \
    dch --bin-nmu -Dunstable "Optimized build" && \
    DEB_CFLAGS_SET="$OPT" DEB_LDFLAGS_SET="$OPT" dpkg-buildpackage -uc -b -j2 && \
    cd .. && \
    apt-get remove -y $(cat build_bloat) && \
    rm -f libpython3.8-testsuite* python3.8-examples* python3.8-doc* idle-python3.8* python3.8-venv* python3.8-full* && \
    apt-get remove -y dpkg-dev devscripts && \
    apt-get autoremove -y && \
    dpkg -i *.deb && \
    cd / && \
    rm -rf /cpython && \
    apt-mark hold python3.8 python3.8-minimal libpython3.8 libpython3.8-minimal && \
    pip3 install --no-cache-dir cython && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD Makefile /

RUN apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends curl binutils elfutils make && \
    make prodfiler-symbols && \
    apt-get purge -y curl binutils elfutils make && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm Makefile

ENV MKL=2020.0-166

# Intel MKL
RUN apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends gnupg && \
    echo "deb https://apt.repos.intel.com/mkl all main" > /etc/apt/sources.list.d/intel-mkl.list && \
    wget -O - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB | \
    apt-key add - && \
    apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends intel-mkl-common-c-$MKL intel-mkl-gnu-rt-$MKL intel-mkl-f95-$MKL && \
    rm -rf /opt/intel/documentation_2019 /opt/intel/compilers_and_libraries_*/linux/mkl/{bin,tools,examples} && \
    ln -s /opt/intel/compilers_and_libraries_*/linux/mkl /opt/intel/mkl && \
    printf '/opt/intel/mkl/lib/intel64_lin' >> /etc/ld.so.conf.d/mkl.conf && \
    ldconfig && \
    update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so     \
                    libblas.so-x86_64-linux-gnu      /opt/intel/mkl/lib/intel64/libmkl_rt.so 50 && \
    update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so.3   \
                    libblas.so.3-x86_64-linux-gnu    /opt/intel/mkl/lib/intel64/libmkl_rt.so 50 && \
    update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so   \
                    liblapack.so-x86_64-linux-gnu    /opt/intel/mkl/lib/intel64/libmkl_rt.so 50 && \
    update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so.3 \
                    liblapack.so.3-x86_64-linux-gnu  /opt/intel/mkl/lib/intel64/libmkl_rt.so 50 && \
    apt-get clean && \
    apt-get purge -y gnupg && \
    apt-get autoremove -y --purge && \
    rm -rf /var/lib/apt/lists/*

# numpy
RUN echo '[ALL]\n\
extra_compile_args = -O3 -fopenmp -flto -ftree-vectorize OPT\n\
extra_link_args = -O3 -fopenmp -flto -ftree-vectorize OPT\n\
\n\
[fftw]\n\
libraries = fftw3\n\
\n\
[mkl]\n\
library_dirs = /opt/intel/mkl/lib/intel64_lin\n\
include_dirs = /opt/intel/mkl/include\n\
mkl_libs = mkl_rt\n\
lapack_libs = mkl_lapack95_lp64' >/root/.numpy-site.cfg && \
    sed -i "s/OPT/$OPT/g" /root/.numpy-site.cfg && \
    cat /root/.numpy-site.cfg && \
    apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends \
      libfftw3-3 libfftw3-dev gfortran libgfortran5 gcc g++ && \
    export NPY_NUM_BUILD_JOBS=$(getconf _NPROCESSORS_ONLN) && \
    echo $NPY_NUM_BUILD_JOBS && \
    pip3 $VERBOSE install --no-cache-dir scipy==1.7.3 numpy==1.21.4 --no-binary numpy && \
    apt-get purge -y libfftw3-dev gfortran gcc g++ && \
    apt-get autoremove -y --purge && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /usr/share/doc/*

ADD server/requirements.txt /server/requirements.txt
ADD patches /patches
ARG GKWILLIE_TOKEN
RUN apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends gcc g++ patch git && \
    sed -i "s/git+ssh:\/\/git@/git+https:\/\/gkwillie:$GKWILLIE_TOKEN@/g" server/requirements.txt && \
    pip3 install --no-cache-dir -r /server/requirements.txt && \
    sed -i "s/git+https:\/\/gkwillie:$GKWILLIE_TOKEN@/git+ssh:\/\/git@/g" server/requirements.txt && \
    pip3 uninstall -y flask && \
    patch /usr/local/lib/python*/dist-packages/prometheus_client/exposition.py /patches/prometheus_client.patch && \
    patch /usr/local/lib/python*/dist-packages/aiomcache/client.py /patches/aiomcache_version.patch && \
    apt-get purge -y patch git gcc g++ && \
    apt-get autoremove -y --purge && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD server /server
ADD README.md /
RUN apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends gcc g++ && \
    echo "Installing Python packages" && \
    pip3 install -e /server && \
    apt-get purge -y gcc g++ && \
    apt-get autoremove -y --purge && \
    apt-get upgrade -y && \
    apt-get clean
ARG COMMIT
RUN echo "__commit__ = \"$COMMIT\"" >>/server/athenian/api/metadata.py && \
    echo "__date__ = \"$(date -u +'%Y-%m-%dT%H:%M:%SZ')\"" >>/server/athenian/api/metadata.py

ENTRYPOINT ["python3", "-m", "athenian.api"]
