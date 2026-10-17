# Build through scripts/container.py: the context contains only audited inputs.
# Trixie supplies glibc >= 2.38, required by the locked ARM64 Z3 wheel.
FROM python:3.12-slim-trixie@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

ARG VERSION
ARG REVISION
LABEL org.opencontainers.image.title="pybingraph" \
      org.opencontainers.image.source="https://github.com/manelfx/pybingraph" \
      org.opencontainers.image.licenses="BSD-2-Clause" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    BINGRAPH_ROOT=/binaries XDG_CACHE_HOME=/tmp/.cache
RUN apt-get update \
    && apt-get install -y --no-install-recommends graphviz libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 bingraph \
    && mkdir /binaries

# Hash-verified locked runtime dependencies, then our already validated wheel.
# BuildKit mounts keep wheels and requirements out of the final image layers.
RUN --mount=type=bind,source=.,target=/install \
    pip install --no-cache-dir --require-hashes -r /install/requirements.txt \
    && pip install --no-cache-dir --no-deps /install/*.whl \
    && pip check
COPY LICENSE NOTICE /usr/share/doc/pybingraph/

USER 10001:10001
WORKDIR /home/bingraph
EXPOSE 8000
ENTRYPOINT ["bingraph"]
CMD ["server", "--host", "0.0.0.0", "--port", "8000"]
