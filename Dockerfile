FROM ubuntu:20.04

ENV BROWSER=/browser \
    LC_ALL=en_US.UTF-8

RUN echo '#!/bin/bash\n\
\n\
echo\n\
echo "  $@"\n\
echo\n' > /browser && \
    chmod +x /browser

# runtime environment
RUN apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends \
      apt-utils ca-certificates gnupg2 locales curl python3 python3-distutils && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    curl -L https://bootstrap.pypa.io/get-pip.py | python3 && \
    pip3 install --no-cache-dir cython && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV MKL=2020.0-166

# Intel MKL
RUN echo "deb https://apt.repos.intel.com/mkl all main" > /etc/apt/sources.list.d/intel-mkl.list && \
    curl -L https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB | \
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
    rm -rf /var/lib/apt/lists/*

# numpy
RUN echo '[ALL]\n\
extra_compile_args = -O3 -fopenmp -flto -ftree-vectorize -march=haswell -mmmx -msse -msse2 -msse3 -mssse3 -mcx16 -msahf -mmovbe -maes -mpclmul -mpopcnt -mabm -mfma -mbmi -mbmi2 -mavx -mavx2 -msse4.2 -msse4.1 -mlzcnt -mrdrnd -mf16c -mfsgsbase -mfxsr -mxsave -mxsaveopt --param l1-cache-size=32 --param l1-cache-line-size=64 --param l2-cache-size=46080 -mtune=haswell\n\
extra_link_args = -O3 -fopenmp -flto -ftree-vectorize -march=haswell -mmmx -msse -msse2 -msse3 -mssse3 -mcx16 -msahf -mmovbe -maes -mpclmul -mpopcnt -mabm -mfma -mbmi -mbmi2 -mavx -mavx2 -msse4.2 -msse4.1 -mlzcnt -mrdrnd -mf16c -mfsgsbase -mfxsr -mxsave -mxsaveopt --param l1-cache-size=32 --param l1-cache-line-size=64 --param l2-cache-size=46080 -mtune=haswell \n\
\n\
[fftw]\n\
libraries = fftw3\n\
\n\
[mkl]\n\
library_dirs = /opt/intel/mkl/lib/intel64_lin\n\
include_dirs = /opt/intel/mkl/include\n\
mkl_libs = mkl_rt\n\
lapack_libs = mkl_lapack95_lp64' > /root/.numpy-site.cfg && \
    apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends \
      libfftw3-3 libfftw3-dev gfortran libgfortran5 python3-dev gcc g++ && \
    export NPY_NUM_BUILD_JOBS=$(getconf _NPROCESSORS_ONLN) && \
    echo $NPY_NUM_BUILD_JOBS && \
    pip3 $VERBOSE install --no-cache-dir scipy==1.2.1 numpy==1.18.3 --no-binary numpy && \
    apt-get remove -y libfftw3-dev gfortran python3-dev gcc g++ && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /usr/share/doc/*

ADD server/requirements.txt /server/requirements.txt
ADD databases.patch /databases.patch
RUN apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends python3-dev gcc g++ patch && \
    pip3 install --no-cache-dir -r /server/requirements.txt && \
    pip3 uninstall -y typing && \
    patch /usr/local/lib/python*/dist-packages/databases/backends/postgres.py /databases.patch && \
    rm /databases.patch && \
    apt-get remove -y patch && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD server /server
ADD README.md /
RUN pip3 install -e /server
ARG COMMIT
RUN echo "__commit__ = \"$COMMIT\"" >>/server/athenian/api/metadata.py && \
    echo "__date__ = \"$(date -u +'%Y-%m-%dT%H:%M:%SZ')\"" >>/server/athenian/api/metadata.py

ENTRYPOINT ["python3", "-m", "athenian.api"]
