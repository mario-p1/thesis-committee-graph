.PHONY:
train:
	uv run python -m thesis_graph.train

.PHONY:
mlflow_ui:
	uv run python -m mlflow ui --backend-store-uri sqlite:///mlflow.db

.PHONY:
install_deps_cpu:
	uv sync
	uv pip install pyg_lib -f https://data.pyg.org/whl/torch-2.8.0+cpu.html

.PHONY:
install_deps_cu128:
	uv sync
	uv pip install pyg_lib -f https://data.pyg.org/whl/torch-2.8.0+cu128.html