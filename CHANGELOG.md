## Release 0.1.6 (2021-03-22)

This release focuses on hardening the LLVM environments, providing improved
semantics validation, and improving the datasets. Many thanks to @JD-at-work,
@bwasti, and @mostafaelhoushi for code contributions.

- [llvm] Added a new `cBench-v1` dataset which changes the function attributes
  of the IR to permit inlining. `cBench-v0` is deprecated and will be removed no
  earlier than v0.1.6.
- [llvm] Removed 15 passes from the LLVM action space: `-bounds-checking`,
  `-chr`, `-extract-blocks`, `-gvn-sink`, `-loop-extract-single`,
  `-loop-extract`, `-objc-arc-apelim`, `-objc-arc-contract`, `-objc-arc-expand`,
  `-objc-arc`, `-place-safepoints`, `-rewrite-symbols`,
  `-strip-dead-debug-info`, `-strip-nonlinetable-debuginfo`, `-structurizecfg`.
  Passes are removed if they are: irrelevant (e.g. used only debugging), if they
  change the program semantics (e.g. inserting runtimes bound checking), or if
  they have been found to have nondeterministic behavior between runs.
- Extended `env.step()` so that it can take a list of actions that are all
  performed in a single batch. This improve efficiency.
- Added default reward spaces for `CompilerEnv` that are derived from scalar
  observations (thanks @bwasti!)
- Added a new Q learning example (thanks @JD-at-work!).
- *Deprecation:* The next release v0.1.5 will introduce a new datasets API that
  is easier to use and more flexible. In preparation for this, the `Dataset`
  class has been renamed to `LegacyDataset`, the following dataset operations
  have been marked deprecated: `activate()`, `deactivate()`, and `delete()`. The
  `GetBenchmarks()` RPC interface method has also been marked deprecated..
- [llvm] Improved semantics validation using LLVM's memory, thread, address, and
  undefined behavior sanitizers.
- Numerous bug fixes and improvements.

## Release 0.1.3 (2021-02-25)

This release adds numerous enhancements aimed at improving ease-of-use. Thanks
to @broune, @hughleat, and @JD-ETH for contributions.

* Added a new `env.validate()` API for validating the state of an environment.
  Added semantics validation for some LLVM benchmarks.
* Added a `env.fork()` method to efficiently duplicate an environment state.
* The `manual_env` environment has been improved with new features such as hill
  climbing search and tab completion.
* Ease of use improvements for string observation space and reward space names:
  Added new getter methods such as `env.observation.Autophase()` and generated
  constants such as `llvm.observation_spaces.autophase`.
* *Breaking change*: Calculation of environment reward has been moved to Python.
  Reward functions have been removed from backend service implementations and
  replaced with equivalent Python classes.
* Various bug fixes and improvements.

## Release 0.1.2 (2021-01-25)

* Add a new `compiler_gym.views.ObservationView.add_derived_space(...)` API
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
