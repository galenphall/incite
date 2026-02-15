import esbuild from "esbuild";
import process from "process";
import builtins from "builtin-modules";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const prod = process.argv[2] === "production";

esbuild
	.build({
		entryPoints: ["src/main.ts"],
		bundle: true,
		external: [
			"obsidian",
			"electron",
			"@codemirror/autocomplete",
			"@codemirror/collab",
			"@codemirror/commands",
			"@codemirror/language",
			"@codemirror/lint",
			"@codemirror/search",
			"@codemirror/state",
			"@codemirror/view",
			"@lezer/common",
			"@lezer/highlight",
			"@lezer/lr",
			...builtins,
		],
		alias: {
			"@incite/shared": path.resolve(__dirname, "../shared/src/index.ts"),
		},
		format: "cjs",
		target: "es2018",
		logLevel: "info",
		sourcemap: prod ? false : "inline",
		treeShaking: true,
		outfile: "main.js",
		minify: prod,
	})
	.catch(() => process.exit(1));
