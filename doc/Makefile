PDF = description.pdf presentation.pdf devoir1.pdf

all: $(PDF)

%.pdf : %.tex
	grep ^%SCRIPT $< | cut -c9- | sh
	cat $< | ./pdflatexfilter > $@

clean :
	$(RM) $(PDF) *.png
