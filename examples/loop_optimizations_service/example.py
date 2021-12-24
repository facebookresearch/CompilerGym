import compiler_gym
import examples.loop_optimizations_service as loop_optimizations_service  # noqa Register environments.

env = compiler_gym.make(
    "loops-opt-py-v0",
    benchmark="loops-opt-v0/add",
    observation_space="ir",
    reward_space="runtime",
    # loop="outer_to_inner",#"inner_to_outer",#"all_loops",
    # function="",
    # call_site="",
)
compiler_gym.set_debug_level(4)  # TODO: check why this has no effect

observation = env.reset()
print("observation: ", observation)

print()

# loops_config = env.describe_loops()

# env.set_loop(loops_config(i))

# get_loop(1.3)

# observations:
#   - for ProGraML: add an attribute to statement nodes for which loop index they belong to (e.g, loop 1.3)
#   - for AutoPhase: ask for features of a loop, or a loop and its children

# TODO: these methods are not working:
#    - env.step(env.action_space.sample())
#    - env.step({"unroll": 0, "vectorize": 2})

# for loop in loops_config.loops().flatten():
#     while !done:
#         observation, reward, done, info = env.step()
#         # you can read observation , rewards, etc. every env.step() OR every env.next_loop() OR change all loops then step
#         print("observation: ", observation)
#         print("reward: ", reward)
#         print("done: ", done)
#         print("info: ", info)
#
#     env.next_loop()
#
# for loop in loops_config.loops()
#     for loop_1 in loop:
#         ...
#         # or use recursion

print()

observation, reward, done, info = env.step(env.action_space.sample())
print("observation: ", observation)
print("reward: ", reward)
print("done: ", done)
print("info: ", info)

env.close()

# TODO: implement write_bitcode(..) or write_ir(..)
# env.write_bitcode("/tmp/output.bc")
