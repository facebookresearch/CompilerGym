import compiler_gym
import examples.example_unrolling_service as unrolling_service  # noqa Register environments.

env = compiler_gym.make(
    "unrolling-py-v0",
    benchmark="unrolling-v0/conv2d",
    observation_space="features",
    reward_space="runtime",
)

observation = env.reset()
print("observation: ", observation)

observation, reward, done, info = env.step(env.action_space.sample())
print("observation: ", observation)
print("reward: ", reward)
print("done: ", done)
print("info: ", info)

# TODO: implement write_bitcode(..)
# env.write_bitcode("/tmp/output.bc")
