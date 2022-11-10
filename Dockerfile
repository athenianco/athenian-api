FROM ubuntu:22.04

ENV BROWSER=/browser \
    LC_ALL=en_US.UTF-8 \
    SETUPTOOLS_USE_DISTUTILS=stdlib \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Madrid \
    PYTHON_TARGET_VERSION="3.11.0-1+jammy1" \
    PYTHON_VERSION=3.11 \
    MKL=2020.4-304

RUN echo '#!/bin/bash\n\
\n\
echo\n\
echo "  $@"\n\
echo\n' > /browser && \
    chmod +x /browser

# Intel MKL
RUN apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends gnupg gcc ca-certificates wget && \
    echo "deb https://apt.repos.intel.com/mkl all main" > /etc/apt/sources.list.d/intel-mkl.list && \
    wget -O - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB | \
    apt-key add - && \
    apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends intel-mkl-common-c-$MKL intel-mkl-gnu-rt-$MKL intel-mkl-f95-$MKL && \
    rm -rf \
        /opt/intel/documentation_* \
        /opt/intel/compilers_and_libraries_*/linux/mkl/bin \
        /opt/intel/compilers_and_libraries_*/linux/mkl/tools \
        /opt/intel/compilers_and_libraries_*/linux/mkl/examples && \
    find /opt/intel/compilers_and_libraries_*/linux/mkl -name '*.a' -delete && \
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
    apt-get purge -y gnupg gcc && \
    apt-get autoremove -y --purge && \
    rm -rf /var/lib/apt/lists/*

# matches our production except -march=haswell, we have to downgrade -march because of GHA
ENV OPT="-fno-semantic-interposition -Wl,--emit-relocs -march=haswell -mabm -maes -mno-pku -mno-sgx --param l1-cache-line-size=64 --param l1-cache-size=32 --param l2-cache-size=33792"
# Bolt: -fno-reorder-blocks-and-partition

# runtime environment
RUN echo 'deb-src http://archive.ubuntu.com/ubuntu/ jammy main restricted' >>/etc/apt/sources.list && \
    echo 'deb-src http://archive.ubuntu.com/ubuntu/ jammy-updates main restricted' >>/etc/apt/sources.list && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install --no-install-suggests --no-install-recommends -y \
      apt-utils git locales dpkg-dev devscripts software-properties-common \
      libexpat1-dev tzdata mime-support libsqlite3-0 libreadline8 \
      python3-distutils html2text libjs-sphinxdoc && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    add-apt-repository -s ppa:deadsnakes/ppa && \
    mkdir /cpython && \
    cd /cpython && \
    apt-get source python$PYTHON_VERSION && \
    apt-get -s build-dep python$PYTHON_VERSION | grep "Inst " | cut -d" " -f2 | sort | tr '\n' ' ' >build_bloat && \
    DEBIAN_FRONTEND="noninteractive" TZ="Europe/Madrid" apt-get build-dep -y python$PYTHON_VERSION && \
    rm /etc/apt/sources.list.d/deadsnakes* && \
    cd python$PYTHON_VERSION* && \
    sed -i 's/__main__/__skip__/g' Tools/scripts/run_tests.py && \
    dch --bin-nmu -Dunstable "Optimized build" && \
    echo 11 >debian/compat && \
    sed -i 's/debhelper (>= 9)/debhelper (>= 11)/g' debian/control.in && \
    DEB_CFLAGS_SET="$OPT" DEB_LDFLAGS_SET="$OPT" dpkg-buildpackage -uc -b -j$(getconf _NPROCESSORS_ONLN) && \
    cd .. && \
	apt-get source python3 && \
    cd python3-defaults-*/debian && \
    for f in debian_defaults control control.in rules; do sed -i -e 's/3.10.6/3.11.0/g' -e 's/3.10/3.11/g' $f; done && \
    sed -i 's/python3.11:any (>= 3.11.0/python3.10:any (>= 3.10.0/g' control && \
    sed -i 's/python3-distutils (>= @STDLIBVER@/python3-distutils (>= 3.10.6-1~/g' control.in && \
    cd /cpython/python3-defaults-* && \
    dch -v "3.11.0-1+jammy1+b1" -Dunstable "3.11" && \
    dpkg-buildpackage -uc -b && \
    cd .. && \
    rm -f \
      2to3* \
      idle* \
      libpython3-all* \
      python3-all* \
      python3-examples* \
      python3-doc* \
      python3-full* \
      python3-nopie* \
      python3-venv* \
      libpython$PYTHON_VERSION-testsuite* \
      python$PYTHON_VERSION-examples* \
      python$PYTHON_VERSION-doc* \
      idle-python$PYTHON_VERSION* \
      python$PYTHON_VERSION-venv* \
      python$PYTHON_VERSION-full* && \
    echo "========" && ls && \
    apt-get purge -y dpkg-dev devscripts software-properties-common html2text $(cat build_bloat) && \
    apt-get autoremove -y && \
    dpkg -i *python3.11*.deb && \
    dpkg -i python3-minimal*.deb libpython3-stdlib*.deb && \
    rm -f python3-minimal*.deb libpython3-stdlib*.deb && \
    dpkg -i python3_*.deb && \
    dpkg -i *python3-*.deb && \
    apt-get purge -y python3.10* libpython3.10* && \
    apt-get autoremove -y && \
    echo "========" && python3 --version && ls /etc/python3* && \
    cd / && \
    rm -rf /cpython && \
    apt-mark hold python$PYTHON_VERSION python$PYTHON_VERSION-minimal libpython$PYTHON_VERSION libpython$PYTHON_VERSION-minimal && \
    wget -O - https://bootstrap.pypa.io/get-pip.py | python3 && \
    python3 -m pip install --no-cache-dir 'cython>=0.29.30' && \
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
    pip3 install 'setuptools<60.0.0' && \
    export NPY_NUM_BUILD_JOBS=$(getconf _NPROCESSORS_ONLN) && \
    echo $NPY_NUM_BUILD_JOBS && \
    pip3 $VERBOSE install --no-cache-dir numpy==1.23.4 --no-binary numpy && \
    apt-get purge -y libfftw3-dev gfortran gcc g++ && \
    apt-get autoremove -y --purge && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /usr/share/doc/*


ARG ROOT_ENC_PASSWD=
RUN if [ -n "${ROOT_ENC_PASSWD}" ]; then \
    echo root:"${ROOT_ENC_PASSWD}" | chpasswd -e; \
    fi

ARG UID=1984
ARG GID=1984

RUN groupadd -g "${GID}" worker && useradd -N -m -s /bin/bash -u "${UID}" -g "${GID}" worker

ADD server/requirements.txt /server/requirements.txt

ADD patches /patches
ARG GKWILLIE_TOKEN
RUN apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends gcc g++ patch && \
    sed -i "s/git+ssh:\/\/git@/git+https:\/\/gkwillie:$GKWILLIE_TOKEN@/g" server/requirements.txt && \
    echo "Installing Python packages" && \
    pip3 install --no-cache-dir -r /server/requirements.txt && \
    sed -i "s/git+https:\/\/gkwillie:$GKWILLIE_TOKEN@/git+ssh:\/\/git@/g" server/requirements.txt && \
    pip3 uninstall -y flask && \
    patch /usr/local/lib/python*/dist-packages/prometheus_client/exposition.py /patches/prometheus_client.patch && \
    apt-get purge -y patch gcc g++ && \
    apt-get autoremove -y --purge && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD server /server
ADD README.md /

RUN apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends gcc g++ cmake make libcurl4 libcurl4-openssl-dev libssl-dev zlib1g-dev && \
    echo "Building native libraries" && \
    make -C /server install-native-user && \
    make -C /server clean-native && \
    rm -rf /usr/local/lib/cmake && \
    pip3 install --no-deps -e /server && \
    apt-get purge -y gcc g++ cmake make libcurl4-openssl-dev libssl-dev zlib1g-dev && \
    apt-get autoremove -y --purge && \
    apt-get upgrade -y && \
    apt-get clean
ARG COMMIT
RUN echo "__commit__ = \"$COMMIT\"" >>/server/athenian/api/metadata.py && \
    echo "__date__ = \"$(date -u +'%Y-%m-%dT%H:%M:%SZ')\"" >>/server/athenian/api/metadata.py

USER worker

WORKDIR /home/worker

ENTRYPOINT ["python3", "-m", "athenian.api"]
