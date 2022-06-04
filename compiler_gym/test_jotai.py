import gym
import compiler_gym

env = compiler_gym.make(
    "llvm-v0",
    benchmark="jotai-v1/extr_anypixelfirmwarecontrollersrcfifo.c_FIFO_available_Final",
    observation_space="Autophase",       # selects the observation space
    reward_space="IrInstructionCountOz", # selects the optimization target
)

env.reset()  
#env.render() 

#env1 = compiler_gym.make(                 # creates a new environment (same as gym.make)
#    "llvm-v0",                           # selects the compiler to use
#    benchmark="cbench-v1/qsort",         # selects the program to compile
#    observation_space="Autophase",       # selects the observation space
#    reward_space="IrInstructionCountOz", # selects the optimization target
#)

#for dataset in env.datasets:
#    print(dataset.name)

#env.reset(benchmark="benchmark://jotai-v1/extr_anypixelfirmwarecontrollersrcfifo.c_FIFO_available_Final")

#info = env.step(env.action_space.sample())
#print(info)

episode_reward = 0
for i in range(1, 101):
    observation, reward, done, info = env.step(env.action_space.sample())
    if done:
        break
    episode_reward += reward
    print(f"Step {i}, quality={episode_reward:.3%}")
env.close() 