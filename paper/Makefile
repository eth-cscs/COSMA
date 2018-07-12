PAPER = mmm-ppopp
TEX = $(wildcard *.tex)
BIB = $(PAPER).bib
FIGS = $(wildcard figures/*.gif figures/*.vsdx figures/*.pdf figures/*.eps)

.PHONY: all clean

$(PAPER).pdf: $(TEX) $(BIB) $(FIGS) 
	echo $(FIGS)
	pdflatex $(PAPER)
	bibtex $(PAPER)
	pdflatex $(PAPER)
	pdflatex $(PAPER)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out $(PAPER).pdf 

