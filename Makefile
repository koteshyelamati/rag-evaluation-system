install:
	pip install -r requirements.txt

run:
	uvicorn app.main:app --reload --port 8000

test:
	pytest tests/ -v

eval:
	curl -X POST http://localhost:8000/api/evaluate

lint:
	python -m py_compile app/config.py app/document_loader.py app/rag_pipeline.py app/evaluator.py app/main.py
