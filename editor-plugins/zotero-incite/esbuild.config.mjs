import esbuild from "esbuild";
import process from "process";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const prod = process.argv[2] === "production";

esbuild
	.build({
		entryPoints: ["src/index.ts"],
		bundle: true,
		alias: {
			"@incite/shared": path.resolve(__dirname, "../shared/src/index.ts"),
		},
		format: "iife",
		globalName: "InciteZotero",
		target: "es2020",
		logLevel: "info",
		sourcemap: prod ? false : "inline",
		treeShaking: true,
		outfile: "addon/content/scripts/index.js",
		minify: prod,
	})
	.catch(() => process.exit(1));
