# Contributing  <!-- omit in toc -->

**Table of Contents**

- [How to Contribute](#how-to-contribute)
- [Pull Requests](#pull-requests)
- [Leaderboard Submissions](#leaderboard-submissions)
- [Code Style](#code-style)
- [Contributor License Agreement ("CLA")](#contributor-license-agreement-cla)

---

## How to Contribute

We want to make contributing to CompilerGym as easy and transparent
as possible. The most helpful ways to contribute are:

1. Provide feedback.
   * [Report bugs](https://github.com/facebookresearch/CompilerGym/issues). In
     particular, it’s important to report any crash or correctness bug. We use
     GitHub issues to track public bugs. Please ensure your description is clear
     and has sufficient instructions to be able to reproduce the issue.
   * Report issues when the documentation is incomplete or unclear, or an error
     message could be improved.
   * Make feature requests. Let us know if you have a use case that is not well
     supported, including as much detail as possible.
1. Contribute to the CompilerGym ecosystem.
   * Pull requests. Please see below for details. The easiest way to get stuck
     is to grab an [unassigned "Good first issue"
     ticket](https://github.com/facebookresearch/CompilerGym/issues?q=is%3Aopen+is%3Aissue+no%3Aassignee+label%3A%22Good+first+issue%22)!
   * Add new features not on [the
     roadmap](https://facebookresearch.github.io/CompilerGym/about.html#roadmap).
     Examples could include adding support for new compilers, producing research
     results using CompilerGym, etc.


## Pull Requests

We actively welcome your pull requests.

1. Fork [the repo](https://github.com/facebookresearch/CompilerGym) and create
   your branch from `development`.
2. Follow the instructions for
   [building from source](https://github.com/facebookresearch/CompilerGym/blob/development/INSTALL.md)
   to set up your environment.
3. If you've added code that should be tested, add tests.
4. If you've changed APIs, update the [documentation](/docs/source).
5. Ensure the `make test` suite passes.
6. Make sure your code lints (see [Code Style](#code-style) below).
7. If you haven't already, complete the [Contributor License Agreement
   ("CLA")](#contributor-license-agreement-cla).


## Leaderboard Submissions

To add a new result to the leaderboard, add a new entry to the leaderboard table
and file a [Pull Request](#pull-requests). Please include:

1. A list of all authors.
2. A CSV file of your results. The
   [compiler_gym.leaderboard](https://facebookresearch.github.io/CompilerGym/compiler_gym/leaderboard.html)
   package provides utilities to help generate results using your agent.
3. A write-up of your approach. You may use the
   [submission template](/leaderboard/SUBMISSION_TEMPLATE.md) as a guide.

Please make sure to update to the latest CompilerGym release prior to
submission. We do not require that you submit the source code for your approach,
though we encourage that you make it publicly available. Once you submit your
pull request we will validate your results CSV files and may ask clarifying
questions if we feel that those would be useful to improve reproducibility.
[Take a look here](https://github.com/facebookresearch/CompilerGym/pull/117) for
an example of a well-formed pull request submission.


## Code Style

We want to ease the burden of code formatting using tools. Our code style
is simple:

* Python:
  [black](https://github.com/psf/black/blob/master/docs/the_black_code_style.md)
  and [isort](https://pypi.org/project/isort/).
* C++: [Google C++
  style](https://google.github.io/styleguide/cppguide.html) with 100
  character line length and `camelCaseFunctionNames()`.

We use [pre-commit](https://pre-commit.com/) to ensure that code is formatted
prior to committing. Before submitting pull requests, please run pre-commit. See
the [config file](/.pre-commit-config.yaml) for installation and usage
instructions.

Other common sense rules we encourage are:

* Prefer descriptive names over short ones.
* Split complex code into small units.
* When writing new features, add tests.
* Make tests deterministic.
* Prefer easy-to-use code over easy-to-read, and easy-to-read code over
  easy-to-write.


## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You
only need to do this once to work on any of Facebook's open source
projects.

Complete your CLA here: <https://code.facebook.com/cla>
