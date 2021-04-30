compiler_gym.datasets
=====================

An instance of a CompilerGym environment uses a :class:`Benchmark
<compiler_gym.datasets.Benchmark>` as the program being optimized. A
:class:`Dataset <compiler_gym.datasets.Dataset>` is collection of benchmarks
that can be installed and made available for use.

.. contents::
  :local:

.. currentmodule:: compiler_gym.datasets


Benchmark
---------

.. autoclass:: Benchmark
  :members:

.. autoclass:: BenchmarkSource
  :members:

.. autoclass:: BenchmarkInitError

Dataset
-------

.. autoclass:: Dataset
 :members:

 .. automethod:: __init__

 .. automethod:: __len__

 .. automethod:: __getitem__

 .. automethod:: __iter__

.. autoclass:: DatasetInitError

FilesDataset
-------------

.. autoclass:: FilesDataset
  :members:

  .. automethod:: __init__


TarDataset
----------

.. autoclass:: TarDataset
  :members:

  .. automethod:: __init__


TarDatasetWithManifest
----------------------

.. autoclass:: TarDatasetWithManifest
  :members:

  .. automethod:: __init__


Datasets
--------

 .. autoclass:: Datasets
  :members:

  .. automethod:: __len__

  .. automethod:: __getitem__

  .. automethod:: __setitem__

  .. automethod:: __delitem__

  .. automethod:: __contains__

  .. automethod:: __iter__
