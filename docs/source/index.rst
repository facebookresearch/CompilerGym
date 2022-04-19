.. image:: /_static/img/logo.png

|pypi| |downloads| |stars| |forks|

.. |pypi| image:: https://badge.fury.io/py/compiler-gym.svg
   :target: https://pypi.org/project/compiler-gym/
   :height: 20px

.. |downloads| image:: https://pepy.tech/badge/compiler-gym
   :target: https://pypi.org/project/compiler-gym/
   :height: 20px

.. |stars| image:: https://img.shields.io/github/stars/facebookresearch/CompilerGym?style=social
   :target: https://github.com/facebookresearch/CompilerGym
   :height: 20px

.. |forks| image:: https://img.shields.io/github/forks/facebookresearch/CompilerGym?style=social
   :target: https://github.com/facebookresearch/CompilerGym
   :height: 20px


`CompilerGym <https://github.com/facebookresearch/CompilerGym>`_  is a library
of easy to use and performant reinforcement learning environments for compiler
tasks. It allows ML researchers to interact with important compiler optimization
problems in a language and vocabulary with which they are comfortable, and
provides a toolkit for systems developers to expose new compiler tasks for ML
research. We aim to act as a catalyst for making compilers faster using ML.

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   getting_started
   about
   cli
   rpc
   changelog
   contributing
   faq

.. toctree::
   :maxdepth: 2
   :caption: Environments

   llvm/index
   envs/gcc
   envs/loop_tool
   envs/mlir

..
    .. toctree::
       :maxdepth: 1
       :caption: Tutorials
       tutorial/makefile_integration
       tutorial/reinforcement_learning
       tutorial/example_service

.. toctree::
   :maxdepth: 3
   :caption: Python API Reference

   compiler_gym/compiler_gym
   compiler_gym/datasets
   compiler_gym/envs
   compiler_gym/envs/gcc
   llvm/api
   compiler_gym/leaderboard
   compiler_gym/service
   compiler_gym/service/runtime
   compiler_gym/spaces
   compiler_gym/views
   compiler_gym/wrappers

.. toctree::
   :maxdepth: 3
   :caption: C++ API Reference

   cc/compiler_gym/envs/llvm/service
   cc/compiler_gym/service
   cc/compiler_gym/service/runtime
   cc/compiler_gym/util


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
