PDF = description.pdf presentation.pdf devoir1.pdf marimba.pdf

all: $(PDF)

%.pdf : %.tex
	grep ^%SCRIPT $< | cut -c9- | sh
	@./.process_verbatims $<
	cat $< | ./pdflatexfilter > $@

clean :
	$(RM) $(PDF) *.png
