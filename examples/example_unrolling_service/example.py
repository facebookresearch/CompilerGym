import compiler_gym
import examples.example_unrolling_service as unrolling_service  # noqa Register environments.

# temporary hack to avoid "isort" pre-hook error
# TODO: remove this line
print(unrolling_service.UNROLLING_PY_SERVICE_BINARY)

# TODO: avoid using hard-coded paths for benchmark files
# benchmark = compiler_gym.envs.llvm.llvm_env.make_benchmark("/Users/melhoushi/CompilerGym-Playground/dataset/offsets1.c")
# benchmark = compiler_gym.datasets.benchmark.Benchmark.from_file("unrolling-example-for-now", "/Users/melhoushi/CompilerGym-Playground/dataset/offsets1.c")
# benchmark = compiler_gym.datasets.benchmark.BenchmarkWithSource.create("unrolling-example-for-now", "/Users/melhoushi/CompilerGym-Playground/dataset/offsets1.c")
benchmark = compiler_gym.envs.llvm.llvm_benchmark.make_benchmark(
    "/Users/melhoushi/CompilerGym-Playground/dataset/offsets1.c"
)

env = compiler_gym.make(
    "unrolling-py-v0",
    benchmark=benchmark,  # "unrolling-v0/foo"
    observation_space="ir",
    reward_space="runtime",
)

observation = env.reset()

observation, reward, done, info = env.step(env.action_space.sample())

print("observation: ", observation)
print("reward: ", reward)
print("done: ", done)
print("info: ", info)

# TODO: implement write_bitcode(..)
# env.write_bitcode("/tmp/output.bc")
