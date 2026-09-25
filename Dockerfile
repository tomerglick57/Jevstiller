# Jevstiller drop-in Jev proxy.
#   docker build -t jevstiller .                                   # CPU, ONNX Runtime, bge-small preloaded
#   docker build -t jevstiller --build-arg PRELOAD_ENCODER= .      # no preload (downloads on first start)
#   docker run -p 8080:8080 -v jevstiller-data:/data jevstiller
# GPU: see docs/deploy.md (EXTRAS=server,gpu on a CUDA base image).
ARG PYTHON=3.12

FROM python:${PYTHON}-slim AS build
ARG EXTRAS=server,onnx
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY jevstiller ./jevstiller
RUN pip wheel --no-cache-dir --wheel-dir /wheels ".[${EXTRAS}]"

FROM python:${PYTHON}-slim
ARG PRELOAD_ENCODER=small
RUN useradd --system --uid 10001 --home-dir /data --shell /usr/sbin/nologin jevstiller \
 && mkdir -p /data /models && chown jevstiller:jevstiller /data && chmod 700 /data
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
ENV JEVSTILLER_DATA_DIR=/data \
    HF_HOME=/models \
    PYTHONUNBUFFERED=1
# Bake the encoder into the image so the container needs no network except to Jev.
RUN if [ -n "$PRELOAD_ENCODER" ]; then \
      python -c "from jevstiller import load_encoder; load_encoder('${PRELOAD_ENCODER}', backend='onnx', device='cpu')"; \
    fi && chmod -R a+rX /models
# With a baked-in encoder, never contact the model hub at runtime (empty, i.e. off, when nothing was preloaded:
# then the default encoder downloads on first start).
ENV JEVSTILLER_ENCODER=${PRELOAD_ENCODER:-small} \
    HF_HUB_OFFLINE=${PRELOAD_ENCODER:+1}
USER jevstiller
VOLUME /data
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2)" || exit 1
ENTRYPOINT ["jevstiller"]
CMD ["serve"]
