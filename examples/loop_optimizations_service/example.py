import compiler_gym
import examples.loop_optimizations_service as loop_optimizations_service  # noqa Register environments.

with compiler_gym.make(
    "loops-opt-py-v0",
    benchmark="loops-opt-v0/add",
    observation_space="ir",
    reward_space="runtime",
) as env:
    compiler_gym.set_debug_level(4)  # TODO: check why this has no effect

    observation = env.reset()
    print("observation: ", observation)

    print()

    observation, reward, done, info = env.step(env.action_space.sample())
    print("observation: ", observation)
    print("reward: ", reward)
    print("done: ", done)
    print("info: ", info)

    env.close()

    # TODO: implement write_bitcode(..) or write_ir(..)
    # env.write_bitcode("/tmp/output.bc")
