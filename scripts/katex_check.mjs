// Parse every formula in the given markdown files with KaTeX (npm install katex).
// Called by scripts/check_notes.py; prints one line per formula KaTeX rejects and exits 1 if any.
import katex from "katex";
import fs from "fs";

let bad = 0;
const inline = /(^|[^\\\w$])\$(?!\s)([^$]+?)(?<!\s)\$(?![\w$])/g;
for (const file of process.argv.slice(2)) {
  const lines = fs.readFileSync(file, "utf8").split("\n");
  let fence = null, buf = [], start = 0;
  const parse = (tex, display, where) => {
    try { katex.renderToString(tex, { displayMode: display, throwOnError: true, strict: "ignore" }); }
    catch (e) { bad++; console.log(`${where}: ${e.message.split("\n")[0]}`); }
  };
  lines.forEach((line, i) => {
    const m = line.match(/^\s*(```+|~~~+)\s*(\w*)/);
    if (fence) {
      if (m && m[1].startsWith(fence.mark) && !m[2]) {
        if (fence.lang === "math") parse(buf.join("\n"), true, `${file}:${start}`);
        fence = null; buf = [];
      } else buf.push(line);
      return;
    }
    if (m) { fence = { mark: m[1], lang: m[2] }; start = i + 1; return; }
    for (const mm of line.replace(/`[^`]*`/g, "").matchAll(inline)) parse(mm[2], false, `${file}:${i + 1}`);
  });
}
process.exit(bad ? 1 : 0);
