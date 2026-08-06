/**
 * Preload that makes `require("vscode")` resolve to the test stub.
 * Used by `npm test`: node --require ./src/test/vscodeShim.js --test …
 */
const Module = require("module");
const path = require("path");

const stubPath = path.join(__dirname, "vscodeStub.js");
const originalResolve = Module._resolveFilename;

Module._resolveFilename = function (request, ...rest) {
  if (request === "vscode") {
    return stubPath;
  }
  return originalResolve.call(this, request, ...rest);
};
