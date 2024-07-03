CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

all: figures/figure_a.pdf \
		figures/figure_b.pdf \
		figures/figure_c.pdf \
		figures/figure_d.pdf \
		figures/figure2_a.pdf \
		figures/figure2_b.pdf \
		figures/figure2_c.pdf \
		figures/figure2_d.pdf \
		figures/figure3_a.pdf \
		figures/figure3_b.pdf \
		figures/figure3_c.pdf \
		figures/figure3_d.pdf

figures/figure%.pdf: python/figure%.py
	$(CONDA_ACTIVATE) quantum_spin_dynamics && python $< >> $(@:.pdf=.log) 2>&1

clean:
	-rm -rf figures/*
	-rm -rf python/__pycache__

.PHONY: all clean