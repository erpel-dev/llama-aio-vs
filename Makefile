.PHONY: all deps compile test package vsix tui clean install

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

clean:
	rm -rf \
		packages/core/out packages/core/out-test \
		packages/vscode/out packages/vscode/prompt-replacements \
		packages/tui/out \
		dist \
		*.vsix
