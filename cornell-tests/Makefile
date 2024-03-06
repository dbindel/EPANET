run:
	../build/bin/runepanet NW_Model_EPS.inp results.out

run_prof:
	rm -rf run.trace
	xcrun xctrace record --output run.trace --launch \
		../xc-build/bin/RelWithDebInfo/runepanet NW_Model_EPS.inp results.out

build:
	( cd ../xc-build ; \
	 cmake --build . --config RelWithDebInfo ; \
	 pushd bin/RelWithDebInfo ; \
	 rm -rf runepanet.dSYM ; \
	 dsymutil runepanet ; \
	 popd ; \
	 pushd lib/RelWithDebInfo ; \
	 rm -rf libepanet2.dylib.dSYM ; \
	 dsymutil libepanet2.dylib )
