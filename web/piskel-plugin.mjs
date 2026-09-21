// The vendored Piskel, built the way upstream builds it - as one concatenated
// classic script and one concatenated stylesheet - by the shell's own Vite
// config, so the shell needs none of upstream's toolchain (spritesmith,
// terser, karma, puppeteer, nw-builder).
//
// Piskel is 200-odd files of pre-module JavaScript that share the `pskl.*`
// namespace through globals and are loaded in the order of
// vendor/piskel/src/piskel-script-list.js. That is not something Rollup can
// bundle, and upstream does not try either: its vite-plugins/concat-scripts.js
// joins the list with ';'. This file does the same three things upstream's
// plugins do (join the scripts, join the styles, resolve the `@@include`
// partials of index.html) and writes them where Vite serves static files:
//
//     public/piskel/piskel.js            the script list, joined
//     public/piskel/css/piskel-style.css the style list, joined
//     public/piskel/css/fonts/**         copied - the CSS reads fonts/ next to it
//     public/piskel/img/**               copied - the CSS reads ../img/
//
// and hands the body markup to Vite's transformIndexHtml, which drops it into
// index.html where `<!--@@piskel-->` is. The output directory is generated
// and ignored by git; `npm run build` and `npm run dev` both regenerate it
// (buildStart), and `npm run piskel` does it on its own.
//
// Node only, stdlib only: it runs inside the Dockerfile's node stage.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

export const VENDOR = path.join(here, "vendor", "piskel", "src");
export const OUT = path.join(here, "public", "piskel");
export const PLACEHOLDER = "<!--@@piskel-->";

function walk(dir) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name);
    return entry.isDirectory() ? walk(full) : [full];
  });
}

function readList(file, property) {
  // The lists are CommonJS (`exports.scripts = [...]`) in upstream's tree;
  // this package is ESM, so they are evaluated with an `exports` of their
  // own rather than require()d.
  const exports = {};
  new Function("exports", fs.readFileSync(path.join(VENDOR, file), "utf8"))(exports);
  const list = exports[property];
  if (!Array.isArray(list) || !list.length) {
    throw new Error(`piskel: ${file} exports no ${property}`);
  }
  return list;
}

function joined(paths, separator) {
  const missing = paths.filter((rel) => !fs.existsSync(path.join(VENDOR, rel)));
  if (missing.length) {
    throw new Error(`piskel: files in the load-order list are missing: ${missing.join(", ")}`);
  }
  return paths.map((rel) => fs.readFileSync(path.join(VENDOR, rel), "utf8")).join(separator);
}

/** `@@include('templates/x.html', {})` -> the file, recursively (upstream's rule). */
function resolveIncludes(content, baseDir) {
  const pattern = /@@include\(\s*'([^']+)'\s*,\s*\{[^}]*\}\s*\)/g;
  return content.replace(pattern, (match, includePath) => {
    const full = path.resolve(baseDir, includePath);
    if (!fs.existsSync(full)) {
      throw new Error(`piskel: index.html includes a file that is not there: ${includePath}`);
    }
    return resolveIncludes(fs.readFileSync(full, "utf8"), path.dirname(full));
  });
}

/**
 * Piskel's markup: what upstream's index.html keeps between
 * <!--body-main-start--> and <!--body-main-end--> "for the editor integrated
 * in piskelapp.com" - the same seam this shell uses. The inlined boot script
 * (piskel-boot.js, which loads the bundle and calls pskl.app.init) is cut:
 * the shell loads the bundle from index.html and boots Piskel itself.
 */
export function piskelBody() {
  const html = fs.readFileSync(path.join(VENDOR, "index.html"), "utf8");
  const start = html.indexOf("<!--body-main-start-->");
  const end = html.indexOf("<!--body-main-end-->");
  if (start < 0 || end < 0) {
    throw new Error("piskel: index.html lost its body-main-start/end markers");
  }
  let body = html.slice(start + "<!--body-main-start-->".length, end);
  body = body.replace(/<script type="text\/javascript">\s*@@include\('piskel-boot\.js'[^<]*<\/script>/, "");
  body = resolveIncludes(body, VENDOR);
  if (body.includes("@@")) {
    throw new Error("piskel: an @@ directive survived the include pass");
  }
  return body;
}

/** Write public/piskel/. Idempotent; returns the list of files written. */
export function generate() {
  const written = [];
  const put = (rel, data) => {
    const full = path.join(OUT, rel);
    fs.mkdirSync(path.dirname(full), { recursive: true });
    fs.writeFileSync(full, data);
    written.push(rel);
  };

  put("piskel.js", joined(readList("piskel-script-list.js", "scripts"), ";\n"));

  // Upstream replaces var(--highlight-color) with gold in the packaged CSS
  // (its css-replace plugin) - kept for parity, the variable is defined too.
  const css = joined(readList("piskel-style-list.js", "styles"), "\n")
    .replace(/var\(--highlight-color\)/g, "gold");
  put(path.join("css", "piskel-style.css"), css);

  for (const sub of ["img", path.join("css", "fonts")]) {
    for (const full of walk(path.join(VENDOR, sub))) {
      put(path.relative(VENDOR, full), fs.readFileSync(full));
    }
  }
  put("logo.png", fs.readFileSync(path.join(VENDOR, "logo.png")));
  return written;
}

/** The Vite plugin: regenerate at buildStart, inject the body into index.html. */
export default function piskel() {
  return {
    name: "anim-studio-piskel",
    buildStart() {
      const files = generate();
      this.info?.(`piskel: ${files.length} files -> public/piskel/`);
    },
    transformIndexHtml(html) {
      if (!html.includes(PLACEHOLDER)) {
        throw new Error(`index.html has no ${PLACEHOLDER} for the editor's markup`);
      }
      return html.replace(PLACEHOLDER, piskelBody());
    },
  };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const files = generate();
  console.log(`piskel: ${files.length} files -> ${path.relative(process.cwd(), OUT)}/`);
}
