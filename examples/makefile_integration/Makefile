CXX := clang++-11
LLVM_LINK := llvm-link-11

this_dir := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))

# The directory containing the source code of the program we are compiling.
SRCDIR := third_party/NPB-CPP/NPB-SER/CG

# The list of binary targets defined in this file.
BINARIES := cg cg-Oz cg-CompilerGym

all: $(BINARIES)
.DEFAULT_GOAL := all

# "Vanilla" build
# ---------------

%.o: %.cpp
	$(CXX) -Oz -c $< -o $@

cg: $(SRCDIR)/cg.o $(SRCDIR)/c_print_results.o $(SRCDIR)/c_randdp.o $(SRCDIR)/c_timers.o $(SRCDIR)/wtime.o
	$(CXX) -Oz $^ -o $@

# Break into LLVM
# ---------------

%.bc: %.cpp
	$(CXX) -emit-llvm -c -O0 -Xclang -disable-O0-optnone -Xclang -disable-llvm-passes $< -o $@

cg.bc: $(SRCDIR)/cg.bc $(SRCDIR)/c_print_results.bc $(SRCDIR)/c_randdp.bc $(SRCDIR)/c_timers.bc $(SRCDIR)/wtime.bc
	$(LLVM_LINK) $^ -o $@

cg-Oz: cg.bc
	$(CXX) -Oz $< -o $@

# Optimize through random search using CompilerGym
# ------------------------------------------------

SEARCH_TIME ?= 60

cg-CompilerGym.bc: cg.bc
	@echo "Running CompilerGym ..."
	@mkdir -p logs
	@touch logs/random_search.json  # file must exist
	@if python -m compiler_gym.bin.random_search --env=llvm-v0 \
		    --benchmark=file:///$(this_dir)/$< --runtime=$(SEARCH_TIME) \
			--reward=IrInstructionCountOz \
		    --output_dir=$(this_dir)/logs --fail_threshold=1; then \
        cp -v logs/optimized.bc $@; \
	else \
	    echo "Random search did not beat the compiler ..."; \
        cp -v $< $@; \
	fi

cg-CompilerGym: cg-CompilerGym.bc
	$(CXX) -Xlinker=--nomagic -v $< -o $@

.PHONY: cg-CompilerGym.bc

# Tests
# -----

test: cg cg-Oz cg-CompilerGym
	test -x cg
	test -x cg-Oz
	test -x cg-CompilerGym

# Tidy up
# -------

.PHONY: clean
clean:
	rm -rf *.bc $(SRCDIR)/*.bc $(SRCDIR)/*.o logs $(BINARIES)

.PHONY: clean
