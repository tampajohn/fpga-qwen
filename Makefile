PYTHON ?= python3
VERILATOR ?= verilator
FAST_BINARY = build/variants/scalar-O3-lto-unchecked-masked-activation-unsigned-indices-fast-attention-parallel-attention-batched-prefill-block-unroll-compact-projection-group-projections/full/Vqwen_behavioral
GROUP_REFERENCE_BINARY = build/variants/scalar-O3-lto-unchecked-masked-activation-unsigned-indices-fast-attention-parallel-attention-batched-prefill-block-unroll-compact-projection/full/Vqwen_behavioral
BLOCK_REFERENCE_BINARY = build/variants/scalar-O3-lto-unchecked-masked-activation-unsigned-indices-fast-attention-parallel-attention-compact-projection/full/Vqwen_behavioral
ATTENTION_REFERENCE_BINARY = build/variants/scalar-O3-lto-unchecked-masked-activation-unsigned-indices-fast-attention-compact-projection/full/Vqwen_behavioral

.PHONY: all prepare test report clean block-test kv-test suffix-test setup check chat chat-test
all: test

check: test kv-test suffix-test full-test chat-test quality-test

.PHONY: quality-test
quality-test:
	uv run --locked python -m unittest tests.test_projection_quality tests.test_matched_prefill -v

prepare:
	$(PYTHON) tools/experiment.py prepare $(if $(MODEL),--model "$(MODEL)",)

build/obj/Vq4_matvec: rtl/q4_matvec.sv sim/main.cpp
	mkdir -p build/obj
	$(VERILATOR) --cc --exe --build -j 4 --top-module q4_matvec \
		--Mdir build/obj -Wall -CFLAGS "-std=c++17 -O2" \
		rtl/q4_matvec.sv $(CURDIR)/sim/main.cpp

test: prepare build/obj/Vq4_matvec
	build/obj/Vq4_matvec build/fixtures > build/results.csv
	$(PYTHON) tools/experiment.py report

report:
	$(PYTHON) tools/experiment.py report

setup:
	uv sync --locked --python python3.12

chat: $(FAST_BINARY)
	uv run --locked tools/chat_server.py $(if $(MODEL),--model "$(MODEL)",) $(if $(PORT),--port $(PORT),)

chat-test: build/full/Vqwen_behavioral
	uv run --locked python -m unittest tests.test_chat tests.test_rtl_chat -v

build/linear/Vq4_matvec: rtl/q4_matvec.sv sim/linear.cpp
	mkdir -p build/linear
	$(VERILATOR) --cc --exe --build -j 4 --top-module q4_matvec \
		--Mdir build/linear -Wall -CFLAGS "-std=c++17 -O2" \
		rtl/q4_matvec.sv $(CURDIR)/sim/linear.cpp

build/kv/Vkv_cursor: rtl/kv_cursor.sv sim/kv_server.cpp
	mkdir -p build/kv
	$(VERILATOR) --cc --exe --build -j 4 --top-module kv_cursor \
		--Mdir build/kv -Wall -CFLAGS "-std=c++17 -O2" \
		rtl/kv_cursor.sv $(CURDIR)/sim/kv_server.cpp

kv-test: build/kv/Vkv_cursor
	$(PYTHON) -m unittest discover -s tests -p 'test_kv.py' -v

suffix-test: build/kv/Vkv_cursor
	$(PYTHON) -m unittest discover -s tests -p 'test_suffix.py' -v

reference-suffix-test: build/kv/Vkv_cursor
	uv run --locked --extra reference tools/suffix_decode.py $(if $(MODEL),--model "$(MODEL)",)

block-test: build/linear/Vq4_matvec build/kv/Vkv_cursor
	uv run --locked --extra reference tools/block_cosim.py $(if $(MODEL),--model "$(MODEL)",)

clean:
	rm -rf build

build/full/Vqwen_behavioral: rtl/qwen_behavioral.sv rtl/sampling.svh sim/full_model.cpp sim/projection_pool.h sim/projection_descriptors.h build/projection/Vprojection_worker__ALL.a
	mkdir -p build/full
	$(VERILATOR) --cc --exe --build -j 4 --top-module qwen_behavioral \
		--Mdir build/full -MAKEFLAGS "OPT_FAST=-O3" -Irtl -Wno-fatal --public-flat-rw \
		-CFLAGS "-std=c++17 -O3 -I$(CURDIR)/build/projection" -O3 \
		rtl/qwen_behavioral.sv $(CURDIR)/sim/full_model.cpp $(CURDIR)/build/projection/Vprojection_worker__ALL.a

.PHONY: full-test
full-test: build/full/Vqwen_behavioral
	uv run --locked python -m unittest tests.test_full_model tests.test_parallel_model tests.test_sampling -v

build/projection/Vprojection_worker__ALL.a: rtl/projection_worker.sv
	mkdir -p build/projection
	$(VERILATOR) --cc --build -j 4 --top-module projection_worker 		--Mdir build/projection -MAKEFLAGS "OPT_FAST=-O3" -Wall -Wno-BLKSEQ -CFLAGS "-std=c++17 -O3" -O3 rtl/projection_worker.sv

.PHONY: conversation-test
conversation-test: build/full/Vqwen_behavioral
	uv run --locked python -m unittest tests.test_conversation -v

.PHONY: chat-reference fast-test
chat-reference: build/full/Vqwen_behavioral
	uv run --locked tools/chat_server.py --simulator reference $(if $(MODEL),--model "$(MODEL)",) $(if $(PORT),--port $(PORT),)

$(FAST_BINARY): rtl/qwen_behavioral.sv rtl/sampling.svh rtl/projection_worker.sv sim/full_model.cpp sim/projection_pool.h sim/projection_descriptors.h tools/build_projection_variant.py
	uv run --locked tools/build_projection_variant.py --storage scalar --opt O3 --lto --unchecked --masked-activation --unsigned-indices --fast-attention --parallel-attention --batched-prefill --block-unroll --compact-projection --group-projections

fast-test: $(FAST_BINARY)
	test -f "$(GROUP_REFERENCE_BINARY)" || uv run --locked tools/build_projection_variant.py --storage scalar --opt O3 --lto --unchecked --masked-activation --unsigned-indices --fast-attention --parallel-attention --batched-prefill --block-unroll --compact-projection
	test -f "$(BLOCK_REFERENCE_BINARY)" || uv run --locked tools/build_projection_variant.py --storage scalar --opt O3 --lto --unchecked --masked-activation --unsigned-indices --fast-attention --parallel-attention --compact-projection
	test -f "$(ATTENTION_REFERENCE_BINARY)" || uv run --locked tools/build_projection_variant.py --storage scalar --opt O3 --lto --unchecked --masked-activation --unsigned-indices --fast-attention --compact-projection
	SV_GROUP_BINARY="$(CURDIR)/$(FAST_BINARY)" SV_GROUP_REFERENCE="$(CURDIR)/$(GROUP_REFERENCE_BINARY)" SV_TEST_BINARY="$(CURDIR)/$(FAST_BINARY)" SV_TEST_LANES=8 SV_TEST_BLOCK_SIZE=4 SV_BLOCK_BINARY="$(CURDIR)/$(FAST_BINARY)" SV_BLOCK_REFERENCE="$(CURDIR)/$(BLOCK_REFERENCE_BINARY)" SV_ATTENTION_BINARY="$(CURDIR)/$(FAST_BINARY)" SV_ATTENTION_REFERENCE="$(CURDIR)/$(ATTENTION_REFERENCE_BINARY)" uv run --locked python -m unittest tests.test_fast_model tests.test_projection_span tests.test_sampling tests.test_projection_scheduler tests.test_conversation tests.test_parallel_attention tests.test_suffix_chat tests.test_suffix_real tests.test_block_model tests.test_grouped_projections -v
