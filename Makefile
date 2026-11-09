.PHONY: verify test reproduce-cpu paper

verify:
	python scripts/verify_release.py

test:
	pytest

reproduce-cpu: verify
	bash scripts/reproduce_cpu.sh

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error arXiv_v1.tex
