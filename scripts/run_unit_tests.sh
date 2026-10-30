XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true python -m pytest --cov=pancax -q --disable-warnings
# XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true pytest test --cov=pancax -q --disable-warnings
# JAX_ENABLE_X64=true \
# XLA_PYTHON_CLIENT_PREALLOCATE=false \
# python -m pytest --cov=pancax -q --disable-warnings