.PHONY: all clean pdf

pdf: clean
	xelatex equivalence-of-neural-field-dynamics-with-different-embedding-dimensionality.tex
	bibtex equivalence-of-neural-field-dynamics-with-different-embedding-dimensionality.aux
	xelatex equivalence-of-neural-field-dynamics-with-different-embedding-dimensionality.tex
	xelatex equivalence-of-neural-field-dynamics-with-different-embedding-dimensionality.tex

clean:
	rm -f *.aux *.log *.out
