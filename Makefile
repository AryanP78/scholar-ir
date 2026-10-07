# Scholar-IR build targets. Run from the repo root inside your virtualenv.
PY ?= python3
export PYTHONPATH := $(CURDIR)/src:$(PYTHONPATH)

.PHONY: setup data citations-test citations index tune pool split judge app merge eval demo test notebooks report all

setup:            ## install dependencies
	$(PY) -m pip install -r requirements.txt

data:             ## Phase 1: stream arXiv metadata -> data/processed/papers.parquet (needs internet)
	$(PY) -m scholar_ir.data_loader

citations-test:   ## Phase 1b smoke test: look up 5 known papers in OpenAlex
	$(PY) -m scholar_ir.citations --test

citations:        ## Phase 1b: Semantic Scholar counts + OpenAlex graph -> data/processed/citations.parquet (resumable)
	$(PY) -m scholar_ir.citations_s2
	$(PY) -m scholar_ir.citations

index:            ## Phase 2: build positional / zone / parametric indexes + authority scores
	$(PY) -m scholar_ir.build

tune:             ## tune BM25/BM25F/lambda on the DEV split only (needs eval/qrels.csv)
	$(PY) -m scholar_ir.experiments tune

pool:             ## export the blind judging pool for eval/queries.csv
	$(PY) -m scholar_ir.experiments pool

split:            ## split the judging pool between people: make split NAMES="aryan rahul priya"
	$(PY) scripts/split_judging.py $(NAMES)

judge:            ## interactive relevance judging: make judge NAME=yourname [QUERIES=F01,R03]
	$(PY) -m scholar_ir.judge --annotator $(NAME) $(if $(QUERIES),--queries $(QUERIES),)

merge:            ## merge eval/judgments/*.csv into eval/qrels.csv (+ inter-annotator kappa)
	$(PY) -m scholar_ir.experiments merge

eval:             ## regenerate every table and figure in results/ (judged parts need eval/qrels.csv)
	$(PY) -m scholar_ir.experiments all

app:              ## open the point-and-click app (Jupyter + ipywidgets)
	$(PY) -m jupyter notebook notebooks/07_app.ipynb

demo:             ## execute the demo notebook end to end
	$(PY) scripts/run_notebooks.py notebooks/06_demo.ipynb

notebooks:        ## execute all notebooks
	$(PY) scripts/run_notebooks.py

report:           ## render report/report_draft.md from report/report_template.md + results/
	$(PY) scripts/build_report.py

test:             ## unit tests (Gate checks)
	$(PY) -m pytest -q tests

all: index eval
