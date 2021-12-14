Anaconda environment YAML
=========================

- `numpy-only.yml`: bare minimum environment. Only supports MPI + NumPy.
- `numpy-cupy.yml`: regular solver environment. Supports MPI + CuPy and MPI + NumPy.
- `all.yml`: on top of `numpy-cupy.yml`, it adds optional
  dependencies required for creating plots in the provided cases/examples.
- `development.yml`: this environments add packages for development, such as
  pytest, flake8, etc.
