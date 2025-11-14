.PHONY:
train:
	uv run python -m mentor_finder.train

.PHONY:
install_deps_cpu:
	uv sync
	uv pip install pyg_lib -f https://data.pyg.org/whl/torch-2.8.0+cpu.html

.PHONY:
install_deps_cu128:
	uv sync
	uv pip install pyg_lib -f https://data.pyg.org/whl/torch-2.8.0+cu128.html