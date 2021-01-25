## Release 0.1.2 (2021-01-25)

* Add a new `compiler_gym.views.ObservationView.add_derviced_space(...)` API
  for constructing derived observation spaces.
* Added default reward and observation values for `env.step()` in case of
  service failure.
* Extended the public `compiler_gym.datasets` API for managing datasets.
* [llvm] Adds `-Norm`-suffixed rewards that are normalized to unoptimized cost.
* Extended documentation and example codes.
* Numerous bug fixes and improvements.

## Release 0.1.1 (2020-12-28)

* Expose the package version through `compiler_gym.__version__`, and
  the compiler version through `CompilerEnv.compiler_version`.
* Add a [notebook
  version](https://colab.research.google.com/github/facebookresearch/CompilerGym/blob/development/examples/getting-started.ipynb)
  of the "Getting Started" guide that can be run in colab.
* [llvm] Reformulate reward signals to be cumulative.
* [llvm] Add a new reward signal based on the size of the `.text`
  section of compiled object files.
* [llvm] Add a `LlvmEnv.make_benchmark()` API for easily constructing
  custom benchmarks for use in environments.
* Numerous bug fixes and improvements.

## Release 0.1.0 (2020-12-21)

Initial release.
