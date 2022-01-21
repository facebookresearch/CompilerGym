LLVM's opt does not always enforce the unrolling or vectorization options passed as cli arguments. Hence, we created our own exeutable with custom unrolling pass in examples/loop_optimizations_service/loop_unroller that enforces the unrolling factors passed in its cli.

To run the custom unroller:
```
bazel run //examples/loop_optimizations_service/opt_loops:opt_loops -- <input>.ll --funroll-count=<num> --force-vector-width=<num> -S -o <output>.ll
```
