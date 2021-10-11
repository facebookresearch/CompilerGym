import sys

import compiler_gym

names = ["toggle_mode", "up", "down", "toggle_thread"]
actions = [3, 0, 1, 3, 0]
base = 1024 * 512
K = int(sys.argv[1]) * base

vectorize = int(sys.argv[2])
run_log = int(sys.argv[3])
env = compiler_gym.make("loop_tool-v0")
env.reset(
    benchmark=env.datasets.benchmark(uri="benchmark://loop_tool-v0/{}".format(K)),
    action_space="simple",
)
if vectorize - 1:
    vs = [1] * (vectorize - 1)
    actions += vs + [0, 1, 0] + vs + [0, 2, 0]
for a in actions:
    o = env.step(a)

if run_log:
    env.observation_space = "action_state"
    inner = 1
    step = 512
    for i in range(1, step):
        o = env.step(1)
        inner += 1
    while inner * vectorize < K:
        env.observation_space = "loop_tree"
        for i in range(step):
            if i == step - 1:
                env.observation_space = "flops"
            o = env.step(1)
            inner += 1
        print(f"{K}, {inner}, {vectorize}: {o[0]}", flush=True)
        step *= 2
else:
    for i in range(K // (vectorize * 1024)):
        env.observation_space = "action_state"
        for j in range(1022 if i == 0 else 1023):
            o = env.step(1)
        env.observation_space = "flops"
        o = env.step(1)
        print(f"{K}, {(i + 1) * 1024}, {vectorize}: {o[0]}", flush=True)
