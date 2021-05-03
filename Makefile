SOURCEDIR = ../compressai/docs

build:
	@rm -rf "$(SOURCEDIR)/_build/"
	@make -C "$(SOURCEDIR)" html
	$(eval BUILDDIR = $(shell sed -n -e 's/^BUILDDIR *= \(.*\)/\1/p' "$(SOURCEDIR)/Makefile"))
	@cp -r $(wildcard $(SOURCEDIR)/$(BUILDDIR)/html/*) .


push:
	touch .nojekyll
	git add .
	git commit -a -m "Update docs $(shell date)"
	git push gitlab-ri gh-pages

watch:
	@find $(SOURCEDIR) ../compressai/compressai/ -type f \
		-name "*.py" \
		-o -name "*.md" -o -name "*.rst" \
		| entr make build

.PHONY: build push watch
