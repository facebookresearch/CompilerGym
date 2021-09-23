import compiler_gym
import examples.example_unrolling_service as unrolling_service

env = compiler_gym.make(
    "unrolling-py-v0",
    benchmark="unrolling-v0/foo",
    observation_space="ir",
    reward_space="runtime",
)

print(unrolling_service.UNROLLING_PY_SERVICE_BINARY)

observation = env.reset()

observation, reward, done, info = env.step(env.action_space.sample())

print("observation: ", observation)
print("reward: ", reward)
print("done: ", done)
print("info: ", info)

# env.write_bitcode("/tmp/output.bc")
