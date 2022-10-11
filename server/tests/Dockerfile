ARG base
FROM $base
USER root
RUN pip3 install coverage && \
    echo "[run]" >/server/.coveragerc && \
    echo "parallel = True" >>/server/.coveragerc && \
    echo "source = /server/athenian/api" >>/server/.coveragerc && \
    echo "import coverage; coverage.process_startup()" >>$(echo /etc/python3.*/sitecustomize.py)
USER worker
ENV COVERAGE_PROCESS_START=/server/.coveragerc
