compiler_gym.spaces
===================

CompilerGym extends the builtin `gym spaces
<https://gym.openai.com/docs/#spaces>`_ to better describe the
observation spaces available to compilers.

.. contents:: Additional spaces:
    :local:

.. currentmodule:: compiler_gym.spaces


Scalar
------

.. autoclass:: Scalar
   :members:

   .. automethod:: __init__


Sequence
--------

.. autoclass:: Sequence
   :members:

   .. automethod:: __init__


NamedDiscrete
-------------

.. autoclass:: NamedDiscrete
   :members:

   .. automethod:: __init__

   .. automethod:: __getitem__


Commandline
-----------

.. autoclass:: Commandline
   :members:

   .. automethod:: __init__

.. autoclass:: CommandlineFlag
   :members:


Reward
------

.. autoclass:: Reward
   :members:

   .. automethod:: __init__
