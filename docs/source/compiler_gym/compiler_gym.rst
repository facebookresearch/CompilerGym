compiler_gym
============

.. contents::
    :local:

.. currentmodule:: compiler_gym


CompilerEnvState
----------------

.. autoclass:: CompilerEnvState
   :members:

.. autoclass:: CompilerEnvStateWriter
   :members:

   .. automethod:: __init__

.. autoclass:: CompilerEnvStateReader
   :members:

   .. automethod:: __init__

   .. automethod:: __iter__


Validation
----------

.. autoclass:: ValidationResult
   :members:

.. autoclass:: ValidationError
   :members:

.. autofunction:: validate_states

Filesystem Paths
----------------

.. autofunction:: cache_path

.. autofunction:: site_data_path

.. autofunction:: transient_cache_path

.. autofunction:: download

Debugging
---------

.. autofunction:: get_debug_level

.. autofunction:: get_logging_level

.. autofunction:: set_debug_level
