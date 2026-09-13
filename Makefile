PYTHON := .venv/bin/python
LLAMA_CPP ?= llama.cpp
MODEL ?= checkpoints/tiny_shakespeare/tiny-q8_0.gguf
PORT ?= 8080
THREADS ?= 10

.PHONY: deps prepare train gguf quantize serve clean

deps:
	uv venv -q
	uv pip install -q -p $(PYTHON) torch --index-url https://download.pytorch.org/whl/cu130
	uv pip install -q -p $(PYTHON) numpy tokenizers safetensors transformers gguf sentencepiece
	test -d $(LLAMA_CPP) || git clone --depth 1 https://github.com/ggml-org/llama.cpp

data/tiny_shakespeare.txt:
	mkdir -p data
	curl -sL -o $@ https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

prepare: data/tiny_shakespeare.txt
	$(PYTHON) prepare.py --input data/tiny_shakespeare.txt --out data/ts --vocab-size 8000

train:
	$(PYTHON) train.py --config configs/tiny_shakespeare.toml

$(LLAMA_CPP)/build/bin/llama-server:
	cmake -S $(LLAMA_CPP) -B $(LLAMA_CPP)/build -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
	cmake --build $(LLAMA_CPP)/build --target llama-server llama-cli llama-quantize -j

gguf:
	$(PYTHON) convert.py checkpoints/tiny_shakespeare \
		--outfile checkpoints/tiny_shakespeare/tiny-f16.gguf --outtype f16

quantize: $(LLAMA_CPP)/build/bin/llama-server
	$(LLAMA_CPP)/build/bin/llama-quantize checkpoints/tiny_shakespeare/tiny-f16.gguf \
		checkpoints/tiny_shakespeare/tiny-q8_0.gguf Q8_0

serve: $(LLAMA_CPP)/build/bin/llama-server
	$(LLAMA_CPP)/build/bin/llama-server -m $(MODEL) \
		--host 127.0.0.1 --port $(PORT) -t $(THREADS)

clean:
	rm -rf data/ts checkpoints/tiny_shakespeare
