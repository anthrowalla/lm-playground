# GPU-free (CPU-only) llama.cpp analyst-assist server distribution.
# Target host: x86_64 Linux, RedHat-compatible (RHEL / Rocky / Alma).
# Build deps:  ./dependencies.sh  (git, make, cmake; gcc-toolset on RHEL 8)
#
# `make` fetches the pinned llama.cpp source, builds the CPU-only
# llama-server, and installs it into ./runtime.

LLAMA_URL ?= https://github.com/ggml-org/llama.cpp
# pinned to the commit the models were validated against
LLAMA_REF ?= 8172e6577ac2b35de1ec1e5d1c0aaad6c4a2129f
PREFIX ?= $(CURDIR)/runtime
NPROC := $(shell nproc)

# RHEL 8's system GCC 8.5 cannot build the pinned llama.cpp (missing
# std::filesystem link, CTAD and <iomanip> errors). When a gcc-toolset
# >= 9 is installed, activate it for every recipe regardless of how make
# was invoked; compiler-check drops any llama.cpp/build configured by a
# different compiler (CMake caches CMAKE_CXX_COMPILER at first configure
# and ignores later environment changes).
TOOLSET := $(firstword $(foreach v,14 13 12 11 10 9,$(wildcard /opt/rh/gcc-toolset-$(v)/enable)))
ifneq ($(TOOLSET),)
ENABLE = . $(TOOLSET) &&
endif

.PHONY: all fetch build install serve clean compiler-check

all: install

fetch: llama.cpp

llama.cpp:
	git init -q $@
	cd $@ && git remote add origin $(LLAMA_URL)
	cd $@ && git fetch -q --depth 1 origin $(LLAMA_REF) && git checkout -q FETCH_HEAD

# CMAKE_INSTALL_LIBDIR pinned to lib: RedHat's GNUInstallDirs defaults it
# to lib64, which would break serve.sh and the $ORIGIN/../lib RUNPATH
CONFIG_FLAGS = -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF \
               -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
               -DCMAKE_INSTALL_LIBDIR=lib \
               -DCMAKE_INSTALL_PREFIX=$(PREFIX) \
               -DCMAKE_INSTALL_RPATH='$$ORIGIN/../lib'

compiler-check:
	@if [ -f llama.cpp/build/CMakeCache.txt ]; then \
	  cached=`sed -n 's/^CMAKE_CXX_COMPILER:FILEPATH=//p' llama.cpp/build/CMakeCache.txt | head -1`; \
	  want=`$(ENABLE)command -v c++ 2>/dev/null`; \
	  if [ -n "$$cached" ] && [ "$$cached" != "$$want" ]; then \
	    echo "compiler changed ($$cached -> $${want:-none}) - dropping stale llama.cpp/build"; \
	    rm -rf llama.cpp/build; \
	  fi; \
	fi
	@if [ -z "$(TOOLSET)" ]; then \
	  major=`gcc -dumpversion 2>/dev/null | cut -d. -f1`; \
	  if [ -z "$$major" ] || [ "$$major" -lt 9 ]; then \
	    echo "error: system GCC ($${major:-unknown}) is too old for the pinned llama.cpp" >&2; \
	    echo "       and no /opt/rh/gcc-toolset-* is installed - run ./dependencies.sh" >&2; \
	    exit 1; \
	  fi; \
	fi

build: compiler-check | llama.cpp
	$(ENABLE)cmake -S llama.cpp -B llama.cpp/build $(CONFIG_FLAGS)
	$(ENABLE)cmake --build llama.cpp/build --target llama-server -j$(NPROC)

# install needs the full install manifest (libs + tools), so build it here
install: compiler-check | llama.cpp
	$(ENABLE)cmake -S llama.cpp -B llama.cpp/build $(CONFIG_FLAGS)
	$(ENABLE)cmake --build llama.cpp/build --target install -j$(NPROC)
	mkdir -p models
	@test -f "$(PREFIX)/lib/libllama-server-impl.so" \
	  || { echo "ERROR: install incomplete - libs did not land in $(PREFIX)/lib" >&2; exit 1; }
	@echo
	@echo "Install OK: $(PREFIX)/bin/llama-server"
	@echo "Next: put a .gguf model in models/, then run ./serve.sh"

serve: install
	./serve.sh

clean:
	rm -rf llama.cpp runtime
