# Build the lecture deck.  Needs a TeX distribution with latexmk and beamer.
TEX = Agentic_Programming_Munich.tex
PDF = $(TEX:.tex=.pdf)

all: $(PDF)

$(PDF): $(TEX) figures/*
	latexmk -pdf -interaction=nonstopmode $(TEX)

clean:
	latexmk -c $(TEX)
	rm -f *.nav *.snm *.vrb build.log

.PHONY: all clean
