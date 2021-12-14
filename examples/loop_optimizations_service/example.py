import compiler_gym
import examples.loop_optimizations_service as unrolling_service  # noqa Register environments.

env = compiler_gym.make(
    "unrolling-py-v0",
    benchmark="unrolling-v0/add",
    observation_space="ir",
    reward_space="runtime",
)
compiler_gym.set_debug_level(4)  # TODO: check why this has no effect

observation = env.reset()
print("observation: ", observation)

print()

# TODO: these methods are not working:
#    - env.step(env.action_space.sample())
#    - env.step({"unroll": 0, "vectorize": 2})
observation, reward, done, info = env.step(7)
print("observation: ", observation)
print("reward: ", reward)
print("done: ", done)
print("info: ", info)

print()

observation, reward, done, info = env.step(env.action_space.sample())
print("observation: ", observation)
print("reward: ", reward)
print("done: ", done)
print("info: ", info)

env.close()

# TODO: implement write_bitcode(..) or write_ir(..)
# env.write_bitcode("/tmp/output.bc")
