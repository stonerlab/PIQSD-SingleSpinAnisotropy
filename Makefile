CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

all: figures/figure.pdf

figures/figure%.pdf: python/figure%.py
	$(CONDA_ACTIVATE) quantum_spin_dynamics && python $< >> $(@:.pdf=.log) 2>&1

clean:
	-rm -rf figures/*
	-rm -rf python/__pycache__

paper:
	@cd paper && latexmk --synctex=1 -xelatex -pdf main

.PHONY: all clean paper