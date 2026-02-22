import esbuild from "esbuild";
import process from "process";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const prod = process.argv[2] === "production";

const sharedAlias = {
	"@incite/shared": path.resolve(__dirname, "../shared/src/index.ts"),
};

// Build 1: Main plugin bundle (hooks, endpoints, etc.)
const mainBuild = esbuild.build({
	entryPoints: ["src/index.ts"],
	bundle: true,
	alias: sharedAlias,
	format: "iife",
	globalName: "InciteZotero",
	target: "es2020",
	logLevel: "info",
	sourcemap: prod ? false : "inline",
	treeShaking: true,
	outfile: "addon/content/scripts/index.js",
	minify: prod,
});

// Build 2: Panel — bundle TS, inline CSS + JS into HTML template
const panelBuild = esbuild
	.build({
		entryPoints: ["src/panel/control-center.ts"],
		bundle: true,
		alias: sharedAlias,
		format: "iife",
		target: "es2020",
		logLevel: "info",
		sourcemap: false,
		treeShaking: true,
		write: false,
		minify: prod,
	})
	.then((result) => {
		const jsCode = result.outputFiles[0].text;
		const cssCode = fs.readFileSync(
			path.resolve(__dirname, "src/panel/control-center.css"),
			"utf-8"
		);
		let htmlTemplate = fs.readFileSync(
			path.resolve(__dirname, "src/panel/control-center.html"),
			"utf-8"
		);

		htmlTemplate = htmlTemplate.replace(
			"/* INLINE_CSS_PLACEHOLDER */",
			cssCode
		);
		htmlTemplate = htmlTemplate.replace(
			"/* INLINE_JS_PLACEHOLDER */",
			jsCode
		);

		const outDir = path.resolve(__dirname, "addon/content/panel");
		fs.mkdirSync(outDir, { recursive: true });
		fs.writeFileSync(path.join(outDir, "panel.html"), htmlTemplate);
		console.log("  addon/content/panel/panel.html");
	});

Promise.all([mainBuild, panelBuild]).catch(() => process.exit(1));
