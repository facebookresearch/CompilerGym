import compiler_gym
import examples.loop_optimizations_service as unrolling_service  # noqa Register environments.

env = compiler_gym.make(
    "unrolling-py-v0",
    benchmark="unrolling-v0/offsets1",
    observation_space="features",
    reward_space="runtime",
)
compiler_gym.set_debug_level(4)  # TODO: check why this has no effect

observation = env.reset()
print("observation: ", observation)

print()

observation, reward, done, info = env.step(0, 32)
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

# TODO: implement write_bitcode(..) or write_ir(..)
# env.write_bitcode("/tmp/output.bc")
