mkdir book/api
pydoc-markdown -I . -p pancax --render-toc > book/api/pancax.md
mdbook build
# mdbook serve -o