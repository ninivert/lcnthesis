.PHONY: all clean pdf

TEXC=pdflatex

pdf: clean
	mkdir tmp
	cd tmp && \
		cp ../{manuscript.tex,references.bib,plos2015.bst} . && \
	    gpp -U "\\" "" "{" "}{" "}" "{" "}" "#" "" manuscript.tex -o manuscript-tmp.tex && \
	    $(TEXC) manuscript-tmp.tex && \
		bibtex manuscript-tmp.aux && \
	    $(TEXC) manuscript-tmp.tex && \
	    $(TEXC) manuscript-tmp.tex && \
		gpp -U "\\" "" "{" "}{" "}" "{" "}" "#" "" -DSUBMISSION manuscript.tex -o manuscript-submission.tex && \
		$(TEXC) manuscript-submission.tex && \
		$(TEXC) manuscript-submission.tex

clean:
	mkdir -p tmp && rm -r tmp
	# rm -f *.aux *.log *.out
