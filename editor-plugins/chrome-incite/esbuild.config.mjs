import { build, context } from "esbuild";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const watch = process.argv.includes("--watch");

const sharedConfig = {
  bundle: true,
  sourcemap: true,
  target: "es2022",
  alias: {
    "@incite/shared": resolve(__dirname, "../shared/src/index.ts"),
  },
  loader: { ".ts": "ts" },
};

const entryPoints = [
  { in: "src/background/service-worker.ts", out: "service-worker" },
  { in: "src/panel/panel.ts", out: "panel" },
  { in: "src/options/options.ts", out: "options" },
  { in: "src/content/googledocs.ts", out: "googledocs" },
  { in: "src/content/overleaf-isolated.ts", out: "overleaf-isolated" },
  { in: "src/content/overleaf-main.ts", out: "overleaf-main" },
];

// Service worker and content scripts must be IIFE (classic scripts in Chrome MV3)
const iifeEntries = [
  "src/background/service-worker.ts",
  "src/content/googledocs.ts",
  "src/content/overleaf-isolated.ts",
  "src/content/overleaf-main.ts",
  "src/content/translator-runner.ts",
];

// Panel, options, and popup are loaded via <script> in their own HTML pages
const pageEntries = [
  "src/panel/panel.ts",
  "src/options/options.ts",
  "src/popup/popup.ts",
];

const iifeConfig = {
  ...sharedConfig,
  entryPoints: iifeEntries,
  outdir: "dist",
  entryNames: "[name]",
  format: "iife",
};

const pageConfig = {
  ...sharedConfig,
  entryPoints: pageEntries,
  outdir: "dist",
  entryNames: "[name]",
  format: "iife",
};

if (watch) {
  const ctx1 = await context(iifeConfig);
  const ctx2 = await context(pageConfig);
  await ctx1.watch();
  await ctx2.watch();
  console.log("Watching for changes...");
} else {
  await Promise.all([build(iifeConfig), build(pageConfig)]);
  console.log("Build complete.");
}
