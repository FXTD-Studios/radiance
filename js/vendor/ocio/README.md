# OpenColorIO — vendored WebAssembly build

Third-party code. Do not edit anything here except to re-vendor a new upstream
release, and record the change below when you do.

| | |
| :-- | :-- |
| Package | [`@bb-studio/ocio`](https://www.npmjs.com/package/@bb-studio/ocio) |
| Version | 0.0.7 |
| Upstream | <https://github.com/promto-c/ocio-js> |
| Contains | OpenColorIO 2.5.0, compiled to WebAssembly |
| Licence | BSD-3-Clause (see `LICENSE`; OpenColorIO's own notices in `THIRD_PARTY_NOTICES.md`) |

## Why it is vendored rather than fetched

Radiance ships as a ComfyUI custom node: no bundler runs at install time and no
package manager resolves its front-end dependencies. Colour management also has
to work on a facility network with no outbound access — a viewer that cannot
load a show's config until it has downloaded 4.7 MB is a viewer that cannot be
used on the show. So the build is committed, pinned, and auditable.

## The patch

`index.js` differs from the published source in exactly one place, at the top of
the file. Upstream resolves the Emscripten module through the package's
`#ocio-wasm` imports map and the `.wasm` URL through `./wasm-url.js`; neither
exists outside an npm install, and ComfyUI serves this directory as plain static
files. Both are resolved locally instead:

- `#ocio-wasm` → a dynamic `import()` of `./ocio-wasm.js` in the browser or
  `./ocio-wasm.node.js` under Node, chosen at call time inside `createOCIO`.
- `./wasm-url.js` → `new URL('./ocio-wasm.wasm', import.meta.url)`.

Everything else — `ocio-wasm.js`, `ocio-wasm.node.js`, `ocio-wasm.wasm`,
`LICENSE`, `THIRD_PARTY_NOTICES.md`, `ocio-js.d.ts` — is byte-identical to the
published tarball.

## Re-vendoring

```
npm pack @bb-studio/ocio
tar -xzf bb-studio-ocio-<version>.tgz
cp package/dist/ocio-wasm.js package/dist/ocio-wasm.node.js \
   package/dist/ocio-wasm.wasm package/LICENSE \
   package/THIRD_PARTY_NOTICES.md package/src/ocio-js.d.ts js/vendor/ocio/
cp package/src/index.js js/vendor/ocio/index.js
# then re-apply the patch above, and update the version in this file
node --test js/tests/ocio.test.mjs
```

`js/tests/ocio.test.mjs` runs against this build directly under Node, so a bad
re-vendor fails there rather than in someone's browser.

## Known caveat

Upstream is at 0.0.x with a single maintainer. The colour maths is
OpenColorIO's and is not in question; the risk sits in the binding layer, which
is why the façade in `js/radiance_ocio.js` is deliberately narrow and why the
viewer keeps its own ACES 1.3 path as the fallback when no config is loaded.
