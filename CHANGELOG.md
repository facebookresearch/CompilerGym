## Release 0.2.2 (2022-01-19)

Amongst the highlights of this release are support for building with CMake and a
new compiler environment based on loop unrolling. Many thanks to @sogartar,
@mostafaelhoushi, @KyleHerndon, and @yqtianust for code contributions!

- Added support for building CompilerGym from source on Linux using **CMake**
  ([#498](https://github.com/facebookresearch/CompilerGym/pull/498),
  [#478](https://github.com/facebookresearch/CompilerGym/pull/478)). The new
  build system coexists with the bazel build and enables customization over the
  CMake configuration used to build the LLVM environment. See
  [INSTALL.md](https://github.com/facebookresearch/CompilerGym/blob/development/INSTALL.md#building-from-source-with-cmake)
  for details. Credit: @sogartar, @KyleHerndon.
- Added an environment for loop optimizations in LLVM
  ([#530](https://github.com/facebookresearch/CompilerGym/pull/530),
  [#529](https://github.com/facebookresearch/CompilerGym/pull/529),
  [#517](https://github.com/facebookresearch/CompilerGym/pull/517)). This new
  example environment provides control over loop unrolling factors and
  demonstrates how to build a standalone LLVM binary using the new CMake build
  system. Credit: @mostafaelhoushi.
- Added a new `BenchmarkUri` class and API for parsing URIs
  ([#525](https://github.com/facebookresearch/CompilerGym/pull/525)). This
  enables benchmarks to have optional parameters that can be used by the backend
  services to modify their behavior.
- **[llvm]** Enabled runtime reward to be calculated on systems where `/dev/shm`
  does not permit executables
  ([#510](https://github.com/facebookresearch/CompilerGym/pull/510)).
- **[llvm]** Added a new `benchmark://mibench-v1` dataset and deprecated
  `benchmark://mibench-v0`
  ([#511](https://github.com/facebookresearch/CompilerGym/pull/511)). If you are
  using `mibench-v0`, please update to the new version.
- **[llvm]** Enabled all 20 of the cBench runtime datasets to be used by the
  `benchmark://cbench-v1` dataset
  ([#525](https://github.com/facebookresearch/CompilerGym/pull/525)).
- Made the `site_data_base` argument of the `Dataset` class constructor optional
  ([#518](https://github.com/facebookresearch/CompilerGym/pull/518)).
- Added support for building CompilerGym from source on macOS Monterey
  ([#494](https://github.com/facebookresearch/CompilerGym/issues/494)).
- Removed the legacy dataset scripts and APIs that were deprecated in v0.1.8.
  Please use the [new dataset
  API](https://compilergym.com/compiler_gym/datasets.html#datasets). The
  following has been removed:
    - The `compiler_gym.bin.datasets` script.
    - The properties: `CompilerEnv.available_datasets`, and
      `CompilerEnv.benchmarks`.
    - The `CompilerEnv.require_dataset()`, `CompilerEnv.require_datasets()`,
      `CompilerEnv.register_dataset()`, and
      `CompilerEnv.get_benchmark_validation_callback()` methods.
- Numerous other bug fixes and improvements.

**Full Change Log**:
[v0.2.1...v0.2.2](https://github.com/facebookresearch/CompilerGym/compare/v0.2.1...v0.2.2)


## Release 0.2.1 (2021-11-17)

Highlights of this release include:

- **[Complex and composite action spaces]** Added a new schema for describing
  action spaces
  ([#369](https://github.com/facebookresearch/CompilerGym/pull/369)). This
  complete overhaul enables a much richer set of actions to be exposed, such as
  composite actions spaces, dictionaries, and continuous actions.
- **[State Transition Dataset]** We have released the first iteration of the
  state transition dataset, a large collection of (state,action,reward) tuples
  for the LLVM environments, suitable for large-scale supervised learning. We
  have added an example learned cost model using a graph neural network in
  `examples/gnn_cost_model`
  ([#484](https://github.com/facebookresearch/CompilerGym/pull/484), thanks
  @bcui19!).
- **[New examples]** We have added several new examples to the `examples/`
  directory, including a new loop unrolling demo based on LLVM
  ([#477](https://github.com/facebookresearch/CompilerGym/pull/477), thanks
  @mostafaelhoushi!), a loop tool demo
  ([#457](https://github.com/facebookresearch/CompilerGym/pull/457), thanks
  @bwasti!), micro-benchmarks for operations, and example reinforcement learning
  scripts ([#484](https://github.com/facebookresearch/CompilerGym/pull/484)).
  See `examples/README.md` for details. We also overhauled the example compiler
  gym service
  ([#467](https://github.com/facebookresearch/CompilerGym/pull/467)).
- **[New logo]** Thanks Christy for designing a great new logo for CompilerGym!
  ([#471](https://github.com/facebookresearch/CompilerGym/pull/471))
- **[llvm]** Added a new `Bitcode` observation space
  ([#442](https://github.com/facebookresearch/CompilerGym/pull/442)).
- Numerous bug fixes and improvements.

Deprecations and breaking changes:

- **[Backend API change]** Out-of-tree compiler services will require updating
  to the new action space API
  ([#369](https://github.com/facebookresearch/CompilerGym/pull/369)).
- The `env.observation.add_derived_space()` method has been deprecated and will
  be removed in a future release. Please use the new
  `derived_observation_spaces` argument to the `CompilerEnv` constructor
  ([#463](https://github.com/facebookresearch/CompilerGym/pull/463)).
- The `compiler_gym.utils.logs` module has been deprecated. Use
  `compiler_gym.utils.runfiles_path` instead
  ([#453](https://github.com/facebookresearch/CompilerGym/pull/453)).
- The `compiler_gym.replay_search` module has been deprecated and merged into
  the `compiler_gym.random_search`
  ([#453](https://github.com/facebookresearch/CompilerGym/pull/453)).


## Release 0.2.0 (2021-09-28)

This release adds two new compiler optimization problems to CompilerGym: GCC
command line flag optimization and CUDA loop nest optimization.

- **[GCC]** A new `gcc-v0` environment, authored by @hughleat, exposes the
  command line flags of [GCC](https://gcc.gnu.org/) as a reinforcement learning
  environment. GCC is a production-grade compiler for C and C++ used throughout
  industry. The environment provides several datasets and a large, high
  dimensional action space that works on several GCC versions. For further
  details check out the [reference
  documentation](https://facebookresearch.github.io/CompilerGym/envs/gcc.html).
- **[loop_tool]** A new `loop_tool-v0` environment, authored by @bwasti,
  provides an experimental intermediate representation of *n*-dimensional data
  computation that can be lowered to both CPU and GPU backends. This provides a
  reinforcement learning environment for manipulating nests of loop computations
  to maximize throughput. For further details check out the [reference
  documentation](https://facebookresearch.github.io/CompilerGym/envs/loop_tool.html).

Other highlights of this release include:

- **[Docker]** Published a
  [chriscummins/compiler_gym](https://hub.docker.com/repository/docker/chriscummins/compiler_gym)
  docker image that can be used to run CompilerGym services in standalone
  isolated containers
  ([#424](https://github.com/facebookresearch/CompilerGym/pull/424)).
- **[LLVM]** Fixed a bug in the experimental `Runtime` observation space that
  caused observations to slow down over time
  ([#398](https://github.com/facebookresearch/CompilerGym/pull/398)).
- **[LLVM]** Added a new utility module to compute observations from bitcodes
  ([#405](https://github.com/facebookresearch/CompilerGym/pull/405)).
- Overhauled the continuous integration services to reduce computational
  requirements by 59.4% while increasing test coverage
  ([#392](https://github.com/facebookresearch/CompilerGym/pull/392)).
- Improved error reporting if computing an observation fails
  ([#380](https://github.com/facebookresearch/CompilerGym/pull/380)).
- Changed the return type of `compiler_gym.random_search()` to a `CompilerEnv`
  ([#387](https://github.com/facebookresearch/CompilerGym/pull/387)).
- Numerous other bug fixes and improvements.

Many thanks to code contributors: @thecoblack, @bwasti, @hughleat, and
@sahirgomez1!


## Release 0.1.10 (2021-09-08)

This release lays the foundation for several new exciting additions to
CompilerGym:

- [LLVM] Added experimental support for **optimizing for runtime** and **compile
  time** ([#307](https://github.com/facebookresearch/CompilerGym/pull/307)).
  This is still proof of concept and is not yet stable. For now, only the
  `benchmark://cbench-v1` and `generator://csmith-v0` datasets are supported.
- [CompilerGym Explorer] Started development of a **web frontend** for the
  LLVM environments. The work-in-progress Flask API and React website can be
  found in the `www` directory.
- [New Backend API] Added a mechanism for sending arbitrary data payloads to the
  compiler service backends
  ([#313](https://github.com/facebookresearch/CompilerGym/pull/313)). This
  allows ad-hoc parameters that do not conform to the usual action space to be
  set for the duration of an episode. Add support for these parameters in the
  backend by implementing the optional
  [handle_session_parameter()](https://github.com/facebookresearch/CompilerGym/blob/63ee95a34157856ca21c392c49d35234e065fa8d/compiler_gym/service/compilation_session.py#L94-L112)
  method, and then send parameters using the
  [send_params()](https://github.com/facebookresearch/CompilerGym/blob/63ee95a34157856ca21c392c49d35234e065fa8d/compiler_gym/envs/compiler_env.py#L1317-L1338)
  method.

Other highlights of this release include:

- [LLVM] The Csmith program generator is now shipped as part of the CompilerGym
  binary release, removing the need to compile it locally
  ([#348](https://github.com/facebookresearch/CompilerGym/pull/348)).
- [LLVM] A new `ProgramlJson` observation space provides the JSON node-link data
  of a ProGraML graph without parsing to a `nx.MultiDiGraph`
  ([#332](https://github.com/facebookresearch/CompilerGym/pull/332)).
- [LLVM] Added a leaderboard submission for a DQN agent
  ([#292](https://github.com/facebookresearch/CompilerGym/pull/292), thanks
  @phesse001!).
- [Backend API Update] The `Reward.reset()` method now receives an observation
  view that can be used to compute initial states
  ([#341](https://github.com/facebookresearch/CompilerGym/pull/341), thanks
  @bwasti!).
- [Datasets API] The size of infinite datasets has been changed from
  `float("inf")` to `0`
  ([#347](https://github.com/facebookresearch/CompilerGym/pull/347)). This is a
  compatibility fix for `__len__()` which requires integers values.
- Prevent excessive growth of in-memory caches
  ([#299](https://github.com/facebookresearch/CompilerGym/pull/299)).
- Multiple compatibility fixes for `compiler_gym.wrappers`.
- Numerous other bug fixes and improvements.

## Release 0.1.9 (2021-06-03)

This release of CompilerGym focuses on **backend extensibility** and adds a
bunch of new features to make it easier to add support for new compilers:

- Adds a new `CompilationSession` class encapsulates a single incremental
  compilation session
  ([#261](https://github.com/facebookresearch/CompilerGym/pull/261)).
- Adds a common runtime for CompilerGym services that takes a
  `CompilationSession` subclass and handles all the RPC wrangling for you
  ([#270](https://github.com/facebookresearch/CompilerGym/pull/270)).
- Ports the LLVM service and example services to the new runtime
  ([#277](https://github.com/facebookresearch/CompilerGym/pull/277)). This
  provides a net performance win with fewer lines of code.

Other highlights of this release include:

- [Core API] Adds a new `compiler_gym.wrappers` module that makes it easy to
  apply modular transformations to CompilerGym environments without modifying
  the environment code
  ([#272](https://github.com/facebookresearch/CompilerGym/pull/272)).
- [Core API] Adds a new `Datasets.random_benchmark()` method for selecting a
  uniform random benchmark from one or more datasets
  ([#247](https://github.com/facebookresearch/CompilerGym/pull/247)).
- [Core API] Adds a new `compiler_gym.make()` function, equivalent to
  `gym.make()`
  ([#257](https://github.com/facebookresearch/CompilerGym/pull/257)).
- [LLVM] Adds a new `IrSha1` observation space that uses a fast, service-side
  C++ implementation to compute a checksum of the environment state
  ([#267](https://github.com/facebookresearch/CompilerGym/pull/267)).
- [LLVM] Adds 12 new C programs from the CHStone benchmark suite
  ([#284](https://github.com/facebookresearch/CompilerGym/pull/284)).
- [LLVM] Adds the `anghabench-v1` dataset and deprecated `anghabench-v0`
  ([#242](https://github.com/facebookresearch/CompilerGym/pull/242)).
- Numerous bug fixes and improvements.

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
