import esbuild from "esbuild";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const prod = process.argv.includes("production");
const watch = process.argv.includes("--watch");

const distDir = path.join(__dirname, "dist");
const assetsDistDir = path.join(distDir, "assets");

// Ensure dist directories exist
fs.mkdirSync(assetsDistDir, { recursive: true });

// Copy static files to dist
function copyStaticFiles() {
	// Copy HTML and CSS
	fs.copyFileSync(
		path.join(__dirname, "src", "taskpane.html"),
		path.join(distDir, "taskpane.html")
	);
	fs.copyFileSync(
		path.join(__dirname, "src", "taskpane.css"),
		path.join(distDir, "taskpane.css")
	);

	// Copy assets
	const assetsDir = path.join(__dirname, "assets");
	if (fs.existsSync(assetsDir)) {
		for (const file of fs.readdirSync(assetsDir)) {
			fs.copyFileSync(
				path.join(assetsDir, file),
				path.join(assetsDistDir, file)
			);
		}
	}
}

copyStaticFiles();

const buildOptions = {
	entryPoints: ["src/taskpane.ts"],
	bundle: true,
	alias: {
		"@incite/shared": path.resolve(__dirname, "../shared/src/index.ts"),
	},
	format: "iife",
	target: "es2020",
	logLevel: "info",
	sourcemap: prod ? false : "inline",
	treeShaking: true,
	outfile: "dist/taskpane.js",
	minify: prod,
};

if (watch) {
	const ctx = await esbuild.context(buildOptions);
	await ctx.watch();
	console.log("Watching for changes...");
} else {
	esbuild.build(buildOptions).catch(() => process.exit(1));
}
