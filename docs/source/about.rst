About
=====

CompilerGym is a toolkit for exposing compiler optimization problems
for reinforcement learning. It allows AI researchers to experiment
with program optimization techniques without requiring any experience
in compilers, and provides a framework for compiler developers to
expose new optimization problems for AI.

.. contents:: Overview:
    :local:

Motivation
-----------

The optimization decisions that compilers make are critical for the
performance and efficiency of compiled software. However, tuning an
optimizing compiler is hard. There are many decisions to make, a near
infinite number of possible input programs, and most programs respond
differently to optimizations. Interest in applying AI to this problem
is increasing rapidly, but compilers are complex, always changing, and
comprise millions of lines of low-level systems code.  Further, most
compilers don’t expose the optimization decisions in a discoverable
manner, and those that do often don’t provide an accessible platform
for experimentation. This creates a high barrier to entry. As a
result, progress is slow.

Our Vision
-----------

We aim to lower the barrier to entry to compiler AI research by
building a playground that allows anyone to experiment with program
optimizations, without needing to write a single line of C++. We have
four goals:

#. Build the “ImageNet for compilers”: a high-quality, open-source
   OpenAI gym environment for experimenting with compiler
   optimizations using real-world software and datasets and get it
   into the hands of AI researchers. The environment takes care of the
   substantial engineering legwork that is required to get started in
   compiler research.

#. Improve the fairness and reproducibility of compiler research by
   allowing direct comparison of different techniques using a common
   experimental framework.

#. Allow users to control every single decision that a compiler
   makes. To begin with, we will focus on exposing high-level
   optimization decisions for which there are existing hand-tuned
   heuristics. We will gradually relax this restriction to accommodate
   every optimization decision, eventually, even introducing new
   decisions, using the compiler as a platform for arbitrary code
   manipulation (see :ref:`roadmap` for details).

#. Make it easy to deploy research findings to production. Often,
   there is no clear path for a research prototype to have impact on
   real-world infrastructure. Using our environment we can build out
   the tools so that the best of the research results can be directly
   applied to improving production use cases, by integrating with
   existing build infrastructure.

.. _roadmap:

Roadmap
-------

* v0.1.0 software release.

  * Release the LLVM codesize problem set for one-line install on
    Linux / macOS.

  * Provide baseline agent implementations and results on benchmarks.

* v0.1.x releases.

  * Add support for optimizing LLVM for performance and compiled
    codesize.

  * Add support for distributed agent/environments during training.

  * Add a distributed environment state cache to make training faster.

* v1.x.x releases.

  * Take what we learned from building environments for compilers and
    generalize it to other systems domains, e.g. OS level
    optimizations, scheduling problems, etc.

Enabling increasingly-granular control over optimization decisions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most compilers optimize programs by running code through a fixed
pipeline of optimization passes. An optimization pass is a set of code
rewrite rules, for example a :code:`LoopUnroll` pass duplicates the
body so that fewer loop iterations are required. Many of these rules
are parameterized, e.g., loop unrolling can be applied with different
factors (how many times to unroll a loop). A pattern matcher
determines where the transform should be applied, e.g. estimating
which loops would benefit from unrolling.

.. image:: /_static/img/compiler_pipeline.png

Our plan is to extend the granularity of control over the compiler
that we expose in four phases:

.. list-table::
   :widths: 15 25 60
   :header-rows: 1

   * - Phase
     - Granularity
     - Example decision
   * - 1
     - Optimization Pass
     - Which optimization pass should I run next?
   * - 2
     - Pattern Matcher
     - Which of the 10 loops in the program sould I unroll?
   * - 3
     - Parametrized Transform
     - How many times should I unroll this loop?
   * - 4
     - Code Rewrite Rule
     - What instruction should I replace this code with to make it
       better?
