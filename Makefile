.PHONY: all compile package clean install deps

all: package

deps:
	npm install

compile: deps
	npm run compile

package: compile
	npx --yes @vscode/vsce package --no-dependencies

install: package
	code --install-extension $$(ls -1t *.vsix | head -n1) --force

clean:
	rm -rf out *.vsix
