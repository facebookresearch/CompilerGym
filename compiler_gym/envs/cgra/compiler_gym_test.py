import compiler_gym

from compiler_gym.service.proto import (
    ActionSpace,
    Benchmark,
    DoubleRange,
    Event,
    Int64Box,
    Int64Tensor,
    NamedDiscreteSpace,
    ObservationSpace,
    Space,
    StringSpace
)

env.reset(benchmark="benchmark://npb-v0/50")
episode_reward = 0

while len(nodes_to_schedule) > 0:

    observation, reward, done, info = env.step(env.action_space.sample())
    if done:
        break

    episode_reward += reward

    print(f"Ste {i}, quality={episode_reward:.2%}")

with compiler_gym.make("llvm-autophase-ic-v0") as env:
    env.reset()