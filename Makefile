SOURCEDIR = ../compressai/docs

build:
	@make -C "$(SOURCEDIR)" html
	$(eval BUILDDIR = $(shell sed -n -e 's/^BUILDDIR *= \(.*\)/\1/p' "$(SOURCEDIR)/Makefile"))
	@cp -r $(wildcard $(SOURCEDIR)/$(BUILDDIR)/html/*) .


push:
	touch .nojekyll
	git add .
	git commit -a -m "Update docs $(shell date)"
	git push gitlab-ri gh-pages

.PHONY: build push
