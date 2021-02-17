CXX ?= clang++
LLVM_LINK ?= llvm-link

this_dir := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))

# The directory containing the source code of the program we are compiling.
SRCDIR := third_party/NPB-CPP/NPB-SER/CG

# The list of binary targets defined in this file.
BINARIES := cg cg-unoptimized cg-optimized cg-gym-optimized

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

$(SRCDIR)/unoptimized.bc: $(SRCDIR)/cg.bc $(SRCDIR)/c_print_results.bc $(SRCDIR)/c_randdp.bc $(SRCDIR)/c_timers.bc $(SRCDIR)/wtime.bc
	$(LLVM_LINK) $^ -o $@

cg-unoptimized: $(SRCDIR)/unoptimized.bc
	$(CXX) $< -o $@

cg-optimized: $(SRCDIR)/unoptimized.bc
	$(CXX) -Oz $< -o $@

# Optimize through random search using CompilerGym
# ------------------------------------------------

SEARCH_TIME ?= 300

$(SRCDIR)/gym-optimized.bc: $(SRCDIR)/unoptimized.bc
	@echo "Running CompilerGym ..."
	@mkdir -p compiler_gym_logs logs
	@touch logs/random_search.json  # file must exist
	@if python -m compiler_gym.bin.random_search --env=llvm-ic-v0 \
		    --benchmark=file:///$(this_dir)/$< --runtime=$(SEARCH_TIME) \
		    --output_dir=$(this_dir)/logs --fail_threshold=1; then \
        cp -v logs/optimized.bc $@; \
	else \
	    echo "Random search did not beat the compiler ..."; \
        cp -v $< $@; \
	fi

cg-gym-optimized: $(SRCDIR)/gym-optimized.bc
	$(CXX) -Xlinker=--nomagic -v $< -o $@

# Tests
# -----

test: cg cg-unoptimized cg-gym-optimized
	test -x cg
	test -x cg-unoptimized
	test -x cg-gym-optimized

# Tidy up
# -------

.PHONY: clean
clean:
	rm -rf $(SRCDIR)/*.bc $(SRCDIR)/*.o logs $(BINARIES)

.PHONY: clean
