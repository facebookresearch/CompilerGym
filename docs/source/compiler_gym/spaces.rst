compiler_gym.spaces
===================

CompilerGym extends the builtin `gym spaces
<https://gym.openai.com/docs/#spaces>`_ to better describe the
observation spaces available to compilers.

.. contents:: Additional spaces:
    :local:

.. currentmodule:: compiler_gym.spaces


Commandline
-----------

.. autoclass:: Commandline
   :members:

   .. automethod:: __init__

.. autoclass:: CommandlineFlag
   :members:


Dict
----

.. autoclass:: Dict
   :members:

   .. automethod:: __init__


Discrete
--------

.. autoclass:: Discrete
   :members:

   .. automethod:: __init__


NamedDiscrete
-------------

.. autoclass:: NamedDiscrete
   :members:

   .. automethod:: __init__

   .. automethod:: __getitem__


Permutation
--------

.. autoclass:: Permutation
   :members:

   .. automethod:: __init__


Reward
------

.. autoclass:: Reward
   :members:

   .. automethod:: __init__


Scalar
------

.. autoclass:: Scalar
   :members:

   .. automethod:: __init__


SpaceSequence
------

.. autoclass:: SpaceSequence
   :members:

   .. automethod:: __init__


Sequence
--------

.. autoclass:: Sequence
   :members:

   .. automethod:: __init__


Tuple
-----

.. autoclass:: Tuple
   :members:

   .. automethod:: __init__
