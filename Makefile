.PHONY: all deps compile test package vsix tui clean install ovsx-namespace publish-ovsx

# Build both the VS Code .vsix and the relocatable TUI executable under dist/.
all: package

deps:
	npm install

compile: deps
	npm run compile

test: deps
	npm test

# Full release artifacts in dist/ (ordered: do not parallelize).
package: compile test
	node scripts/package-vsix.mjs
	node scripts/package-tui.mjs

vsix: compile test
	node scripts/package-vsix.mjs

tui: compile
	node scripts/package-tui.mjs

install: vsix
	code --install-extension $$(ls -1t dist/*.vsix | head -n1) --force

# Open VSX (code-server, VSCodium, Gitpod, ...). ovsx reads the token from
# OVSX_PAT, so it never appears on the command line. One-time setup:
#   OVSX_PAT=... make ovsx-namespace
# Each release (publishes the .vsix matching the current version):
#   OVSX_PAT=... make publish-ovsx
VSIX_VERSION = $(shell node -p "require('./packages/vscode/package.json').version")
VSIX_FILE = dist/llama-aio-vs-$(VSIX_VERSION).vsix

ovsx-namespace:
	@test -n "$$OVSX_PAT" || { echo "OVSX_PAT is not set"; exit 1; }
	npx --yes ovsx create-namespace erpel-dev

publish-ovsx: vsix
	@test -n "$$OVSX_PAT" || { echo "OVSX_PAT is not set"; exit 1; }
	npx --yes ovsx publish $(VSIX_FILE)

clean:
	rm -rf \
		packages/core/out packages/core/out-test \
		packages/vscode/out packages/vscode/prompt-replacements \
		packages/tui/out \
		dist \
		*.vsix
