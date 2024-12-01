for i in *.py; do
    echo "$i"
    JAX_ENABLE_X64=true python "$i"
done
