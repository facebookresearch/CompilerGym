tool: ../loop_unroller

build/bin/loop_unroller: ../loop_unroller.cc CMakeLists.txt
	mkdir -p build
	cmake -DLT_LLVM_INSTALL_DIR=${LLVM_DIR} -S . -B build
	(cd build; make)

../loop_unroller: build/bin/loop_unroller
	cp build/bin/loop_unroller ../

clean:
	rm -rf build
