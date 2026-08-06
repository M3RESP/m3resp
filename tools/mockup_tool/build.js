#!/usr/bin/env node
// Concatenates tools/mockup_tool/src/** into the single distributable
// stage3_gui_mockup.html. No dependencies — Node built-ins only.
//
// Order matters: files are concatenated in the exact order listed below,
// which mirrors the original single-file layout, so declaration-before-use
// ordering the JS relies on is preserved automatically.

const fs = require('fs');
const path = require('path');

const ROOT = __dirname;
const SRC = path.join(ROOT, 'src');
const OUT = path.join(ROOT, 'stage3_gui_mockup.html');

const TITLE = '<title>m3resp — Stage 3 GUI mockup</title>';

const CSS_FILES = [
  'css/base.css',
  'css/prepare.css',
  'css/prepare-eit.css',
  'css/prepare-vent.css',
  'css/prepare-slicer.css',
  'css/design.css',
  'css/popover.css',
  'css/results.css',
];

const HTML_FILES = [
  'html/shell.html',
  'html/prepare.html',
  'html/design.html',
  'html/check.html',
  'html/results.html',
];

const JS_FILES = [
  'js/core/tabs.js',
  'js/design/graph-data.js',
  'js/design/selection.js',
  'js/design/canvas-zoom.js',
  'js/design/node-drag.js',
  'js/design/palette-panel.js',
  'js/design/data-preview.js',
  'js/design/workflow-blocks.js',
  'js/design/connections.js',
  'js/prepare/signal-lanes.js',
  'js/prepare/window-slicer.js',
  'js/prepare/saved-sequences.js',
  'js/prepare/sidebar.js',
  'js/prepare/eit-workspace.js',
  'js/prepare/vent-stack.js',
  'js/prepare/emg-panel.js',
  'js/prepare/zoom-engine.js',
  'js/results/plots.js',
  'js/results/params-filter.js',
];

// Maps an __INJECT_DATA__ marker's data file name to its JSON file under src/data/.
const DATA_FILES = {
  'eit-frame-data.json': 'data/eit-frame-data.json',
  'review-figures.json': 'data/review-figures.json',
};

function read(relPath) {
  return fs.readFileSync(path.join(SRC, relPath), 'utf8');
}

function readTrimmed(relPath) {
  // Section files are cut at line boundaries from the original file, so
  // each one owns a trailing newline; join without doubling blank lines.
  return read(relPath).replace(/\n$/, '');
}

function injectData(jsSource) {
  return jsSource.replace(
    /\/\*__INJECT_DATA__:([\w.-]+)__\*\//g,
    (match, dataFileName) => {
      const relPath = DATA_FILES[dataFileName];
      if (!relPath) {
        throw new Error(`build.js: no data file registered for marker "${dataFileName}"`);
      }
      return read(relPath).replace(/\n$/, '');
    }
  );
}

function build() {
  const css = CSS_FILES.map(readTrimmed).join('\n');
  const html = HTML_FILES.map(readTrimmed).join('\n');
  const js = injectData(JS_FILES.map(readTrimmed).join('\n'));

  const out = [
    TITLE,
    '<style>',
    css,
    '</style>',
    html,
    '<script>',
    js,
    '</script>',
    '',
    '',
  ].join('\n');

  fs.writeFileSync(OUT, out);
  console.log(`Built ${path.relative(ROOT, OUT)} (${(out.length / 1024).toFixed(0)} KB)`);
}

build();
