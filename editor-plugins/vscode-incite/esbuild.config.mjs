import esbuild from "esbuild";
import process from "process";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const prod = process.argv[2] === "production";
const watch = process.argv.includes("--watch");

const buildOptions = {
	entryPoints: ["src/extension.ts"],
	bundle: true,
	external: ["vscode"],
	alias: {
		"@incite/shared": path.resolve(__dirname, "../shared/src/index.ts"),
	},
	format: "cjs",
	platform: "node",
	target: "node18",
	logLevel: "info",
	sourcemap: prod ? false : "inline",
	treeShaking: true,
	outfile: "dist/extension.js",
	minify: prod,
};

if (watch) {
	const ctx = await esbuild.context(buildOptions);
	await ctx.watch();
	console.log("Watching for changes...");
} else {
	esbuild.build(buildOptions).catch(() => process.exit(1));
}
