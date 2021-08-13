SOURCEDIR = ../compressai/docs

.PHONY: help
help: ## Show this message
	@echo "Usage: make COMMAND\n\nCommands:"
	@grep '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' | cat

build: ## Build doc
	@rm -rf "$(SOURCEDIR)/_build/"
	@make -C "$(SOURCEDIR)" html
	$(eval BUILDDIR = $(shell sed -n -e 's/^BUILDDIR *= \(.*\)/\1/p' "$(SOURCEDIR)/Makefile"))
	@cp -r $(wildcard $(SOURCEDIR)/$(BUILDDIR)/html/*) .


push: ## Push to gitlab
	touch .nojekyll
	git add .
	git commit -a -m "Update docs $(shell date)"
	git push gitlab-ri gh-pages

watch: ## Watch for changes and rebuild
	@find $(SOURCEDIR) ../compressai/compressai/ -type f \
		-name "*.py" \
		-o -name "*.md" -o -name "*.rst" \
		| entr make build

.PHONY: build push watch
