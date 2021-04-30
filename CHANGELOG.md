## Release 0.1.8 (2021-04-30)

This release introduces some significant changes to the way that benchmarks are
managed, introducing a new dataset API. This enabled us to add support for
millions of new benchmarks and a more efficient implementation for the LLVM
environment, but this will require some migrating of old code to the new
interfaces (see "Migration Checklist" below). Some of the key changes of this
release are:

- **[Core API change]** We have added a Python
  [Benchmark](https://facebookresearch.github.io/CompilerGym/compiler_gym/datasets.html#compiler_gym.datasets.Benchmark)
  class ([#190](https://github.com/facebookresearch/CompilerGym/pull/190)). The
  `env.benchmark` attribute is now an instance of this class rather than a
  string ([#222](https://github.com/facebookresearch/CompilerGym/pull/222)).
- **[Core behavior change]** Environments will no longer select benchmarks
  randomly. Now `env.reset()` will now always select the last-used benchmark,
  unless the `benchmark` argument is provided or `env.benchmark` has been set.
  If no benchmark is specified, a default is used.
- **[API deprecations]** We have added a new
  [Dataset](https://facebookresearch.github.io/CompilerGym/compiler_gym/datasets.html#compiler_gym.datasets.Dataset)
  class hierarchy
  ([#191](https://github.com/facebookresearch/CompilerGym/pull/191),
  [#192](https://github.com/facebookresearch/CompilerGym/pull/192)). All
  datasets are now available without needing to be downloaded first, and a new
  [Datasets](https://facebookresearch.github.io/CompilerGym/compiler_gym/datasets.html#compiler_gym.datasets.Datasets)
  class can be used to iterate over them
  ([#200](https://github.com/facebookresearch/CompilerGym/pull/200)). We have
  deprecated the old dataset management operations, the
  `compiler_gym.bin.datasets` script, and removed the `--dataset` and
  `--ls_benchmark` flags from the command line tools.
- **[RPC interface change]** The `StartSession` RPC endpoint now accepts a list
  of initial observations to compute. This removes the need for an immediate
  call to `Step`, reducing environment reset time by 15-21%
  ([#189](https://github.com/facebookresearch/CompilerGym/pull/189)).
- [LLVM] We have added several new datasets of benchmarks, including the Csmith
  and llvm-stress program generators
  ([#207](https://github.com/facebookresearch/CompilerGym/pull/207)), a dataset
  of OpenCL kernels
  ([#208](https://github.com/facebookresearch/CompilerGym/pull/208)), and a
  dataset of compilable C functions
  ([#210](https://github.com/facebookresearch/CompilerGym/pull/210)). See [the
  docs](https://facebookresearch.github.io/CompilerGym/llvm/index.html#datasets)
  for an overview.
- `CompilerEnv` now takes an optional `Logger` instance at construction time for
  fine-grained control over logging output
  ([#187](https://github.com/facebookresearch/CompilerGym/pull/187)).
- [LLVM] The ModuleID and source_filename of LLVM-IR modules are now anonymized
  to prevent unintentional overfitting to benchmarks by name
  ([#171](https://github.com/facebookresearch/CompilerGym/pull/171)).
- [docs] We have added a [Feature
  Stability](https://facebookresearch.github.io/CompilerGym/about.html#feature-stability)
  section to the documentation
  ([#196](https://github.com/facebookresearch/CompilerGym/pull/196)).
- Numerous bug fixes and improvements.

Please use this checklist when updating code for the previous CompilerGym release:

* Review code that accesses the `env.benchmark` property and update to
  `env.benchmark.uri` if a string name is required. Setting this attribute by
  string (`env.benchmark = "benchmark://a-v0/b"`) and comparison to string types
  (`env.benchmark == "benchmark://a-v0/b"`) still work.
* Review code that calls `env.reset()` without first setting a benchmark.
  Previously, calling `env.reset()` would select a random benchmark. Now,
  `env.reset()` always selects the last used benchmark, or a predetermined
  default if none is specified.
* Review code that relies on `env.benchmark` being `None` to select benchmarks
  randomly. Now, `env.benchmark` is always set to the previously used benchmark,
  or a predetermined default benchmark if none has been specified. Setting
  `env.benchmark = None` will raise an error. Select a benchmark randomly by
  sampling from the `env.datasets.benchmark_uris()` iterator.
* Remove calls to `env.require_dataset()` and related operations. These are no
  longer required.
* Remove accesses to `env.benchmarks`. An iterator over available benchmark URIs
  is now available at `env.datasets.benchmark_uris()`, but the list of URIs
  cannot be relied on to be fully enumerable (the LLVM environments have over
  2^32 URIs).
* Review code that accesses `env.observation_space` and update to
  `env.observation_space_spec` where necessary
  ([#228](https://github.com/facebookresearch/CompilerGym/pull/228)).
* Update compiler service implementations to support the updated RPC interface
  by removing the deprecated `GetBenchmarks` RPC endpoint and replacing it with
  `Dataset` classes. See the [example
  service](https://github.com/facebookresearch/CompilerGym/tree/development/examples/example_compiler_gym_service)
  for details.
* [LLVM] Update references to the `poj104-v0` dataset to `poj104-v1`.
* [LLVM] Update references to the `cBench-v1` dataset to `cbench-v1`.

## Release 0.1.7 (2021-04-01)

This release introduces [public
leaderboards](https://github.com/facebookresearch/CompilerGym#leaderboards) to
track the performance of user-submitted algorithms on compiler optimization
tasks.

- Added a new `compiler_gym.leaderboard` package which contains utilities for
  preparing leaderboard submissions
  [(#161)](https://github.com/facebookresearch/CompilerGym/pull/161).
- Added a LLVM instruction count leaderboard and seeded it with a random search
  baseline [(#117)](https://github.com/facebookresearch/CompilerGym/pull/117).
- Added support for Python 3.9, extending the set of supported python versions to
  3.6, 3.7, 3.8, and 3.9
  [(#160)](https://github.com/facebookresearch/CompilerGym/pull/160).
- [llvm] Added a new `InstCount` observation space that contains the counts of
  each type of instruction
  [(#159)](https://github.com/facebookresearch/CompilerGym/pull/159).

**Build dependencies update notice:** If you are building from source and
upgrading from an older version of CompilerGym, your build environment will need
to be updated. The easiest way to do that is to remove your existing conda
environment using `conda remove --name compiler_gym --all` and to repeat the
steps in [building from
source](https://github.com/facebookresearch/CompilerGym#building-from-source).

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
- *Deprecation:* The v0.1.8 release will introduce a new datasets API that is
  easier to use and more flexible. In preparation for this, the `Dataset` class
  has been renamed to `LegacyDataset`, the following dataset operations have
  been marked deprecated: `activate()`, `deactivate()`, and `delete()`. The
  `GetBenchmarks()` RPC interface method has also been marked deprecated.
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
