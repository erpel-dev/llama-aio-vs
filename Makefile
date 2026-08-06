.PHONY: all compile test package clean install deps

all: package

deps:
	npm install

compile: deps
	npm run compile

test: deps
	npm test

package: compile test
	npx --yes @vscode/vsce package --no-dependencies

install: package
	code --install-extension $$(ls -1t *.vsix | head -n1) --force

clean:
	rm -rf out out-test *.vsix
