# edit this Makefile as necessary for your project

# Options passed to decktape. Ubuntu 23.10+ (AppArmor) blocks the sandbox of the
# Chrome that Puppeteer downloads, so on Linux Chrome runs without it: export
# only slides you trust. Setting DECKTAPE_ARGS replaces this default, e.g.
#   make DECKTAPE_ARGS=--chrome-path=/usr/bin/google-chrome-stable
ifeq ($(shell uname -s),Linux)
DECKTAPE_ARGS ?= --chrome-arg=--no-sandbox
endif

.PHONY: slides-jp

slides-jp: ## Export index.html to slides-jp.pdf with decktape
	decktape reveal $(DECKTAPE_ARGS) --size 1280x720 index.html slides-jp.pdf
