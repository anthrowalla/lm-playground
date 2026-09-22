# GPU-free (CPU-only) llama.cpp analyst-assist server distribution.
# Target host: x86_64 Linux, RedHat-compatible (RHEL / Rocky / Alma).
# Build deps:  sudo dnf install -y gcc gcc-c++ make cmake git
#
# `make` fetches the pinned llama.cpp source, builds the CPU-only
# llama-server, and installs it into ./runtime.

LLAMA_URL ?= https://github.com/ggml-org/llama.cpp
# pinned to the commit the models were validated against
LLAMA_REF ?= 8172e6577ac2b35de1ec1e5d1c0aaad6c4a2129f
PREFIX ?= $(CURDIR)/runtime
NPROC := $(shell nproc)

.PHONY: all fetch build install serve clean

all: install

fetch: llama.cpp

llama.cpp:
	git init -q $@
	cd $@ && git remote add origin $(LLAMA_URL)
	cd $@ && git fetch -q --depth 1 origin $(LLAMA_REF) && git checkout -q FETCH_HEAD

CONFIG_FLAGS = -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF \
               -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
               -DCMAKE_INSTALL_PREFIX=$(PREFIX) \
               -DCMAKE_INSTALL_RPATH='$$ORIGIN/../lib'

build: | llama.cpp
	cmake -S llama.cpp -B llama.cpp/build $(CONFIG_FLAGS)
	cmake --build llama.cpp/build --target llama-server -j$(NPROC)

# install needs the full install manifest (libs + tools), so build it here
install: | llama.cpp
	cmake -S llama.cpp -B llama.cpp/build $(CONFIG_FLAGS)
	cmake --build llama.cpp/build --target install -j$(NPROC)
	mkdir -p models
	@echo
	@echo "Install OK: $(PREFIX)/bin/llama-server"
	@echo "Next: put a .gguf model in models/, then run ./serve.sh"

serve: install
	./serve.sh

clean:
	rm -rf llama.cpp runtime
