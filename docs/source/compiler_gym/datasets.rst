compiler_gym.datasets
=====================

An instance of a CompilerGym environment uses a :class:`Benchmark
<compiler_gym.datasets.Benchmark>` as the program being optimized. Collections
of benchmarks are packaged into a :class:`Dataset <compiler_gym.datasets.Dataset>`, storing additional metadata such as the
license.

.. contents::
  :local:

.. currentmodule:: compiler_gym.datasets


.. autofunction:: require

.. autofunction:: activate

.. autofunction:: deactivate

.. autofunction:: delete


Benchmark
---------

.. autoclass:: Benchmark
 :members:


Dataset
-------

.. autoclass:: Dataset
 :members:

 .. automethod:: __init__


FilesDataset
-------------

.. autoclass:: FilesDataset
  :members:

  .. automethod:: __init__


TarDataset
----------

.. autoclass:: Dataset
  :members:

  .. automethod:: __init__


TarDatasetWithManifest
----------------------

.. autoclass:: Dataset
  :members:

  .. automethod:: __init__


Datasets
--------

 .. autoclass:: Datasets
  :members:
