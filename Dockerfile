# Inference-only image: small, fast build. Python 3.11 slim.
FROM python:3.11-slim AS builder
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1
# Install deps only (no dev); copy dependency spec first for layer cache.
COPY pyproject.toml ./
RUN pip install --no-deps wheel \
    && pip wheel --no-deps --wheel-dir /wheels numpy scikit-learn joblib fastapi "uvicorn[standard]" pydantic

FROM python:3.11-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1
# Copy wheels and install (no build in final stage).
COPY --from=builder /wheels /wheels
RUN pip install /wheels/*.whl && rm -rf /wheels
# Application code.
COPY src ./src
# Model can be mounted at runtime via MODEL_PATH or copied in at build time.
ENV MODEL_PATH= PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
