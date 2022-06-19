# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CGRACompileSettings = {
    # When this is set, the scheduler will take schedules
    # that don't account for delays appropriately, and try
    # to stretch them out to account for delays correctly.
    # When this is false, the compiler will just reject
    # such invalid schedules.
    # (when set, something like x + (y * z), scheduled
    # as +: PE0 on cycle 0, *: PE1 on cycle 0 is valid).
    "IntroduceRequiredDelays": False,

    # How much buffering to assume each PE has. (i.e.,
    # how many operands can be waiting at that node?)
    # Set to 0 for inifinite buffering, which seems
    # to be a fairly common assumption in literature,
    # although is obviously bogus in real life.
    "BufferLimits": 0,

    # The relative placement algorithm relies on an
    # initial placement of the DFG ndoes.
    # There are several options:
    #   random: uses truly random node placement.  This
    #       seems to perform poorly sometimes as it relies
    #       on the agent to de-fuck the placement without
    #       intermediate rewards --- it is a challenging
    #       enfironment for an agent, although it seems to work
    #       OK for a GA approach.
    #   first_avail:  This uses the first valid slot
    #       for every node ordering.  Under the current
    #       scheme, this is guaranteed to work (I think)
    #       as we have support for infinite buffering --- however
    #       with infinite buffering disabled, this can walk itself
    #       into a hole.
    "InitialPlacementMode": 'first_avail',

    # if there is more gap between operations than required to transmit
    # the operands, should we buffer before or after the transmission?
    # buffering after is more intuitive, but can lead to (rare) situations where
    # the nth_avail assignment runs itself into a hole and can't generate
    # an assignmnet.
    # optionsare before_transmit and after_transmit.
    "BufferingMode": "before_transmit",

    # These are debug flags.  Done this way because various differnet
    # frontends use this, and redefining all the flags seems
    # like a pain in the ass.
    "DebugGetInitializationInterval": True, # Debug the Schedule:get_InitializationInterval() function.
    "DebugGetValidSlots": True, # Debug the InternalSchedule:get_valid_slots function
    "DebugShortestPath": True # Debug the DictNOC:shortest_avaibale_path function
}

# These are some settings for the relative placement algorith.
RelativePlacementSettings = {
    # Allow swaps that cause invalid states.  This may allow a clever agent
    # to perform well, but for less clever agents can result in a lot
    # of failed compilations.  (e.g., for a GA agent, this should
    # probably be true, as that can handle lots of wrong compilations)
    'AllowInvalidIntermediateSchedules': False,

    # Number of times to iterate over the placement algorithm.
    # We do Iterations * DFG Nodes iterations.
    'Iterations': 100,
}