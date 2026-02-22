/**
 * System utilities for detecting Python, managing incite installation via uv,
 * and controlling the local incite server process.
 */

let managedServerPid: number | null = null;

let installState: { status: "idle" | "running" | "done" | "error"; output: string } = {
	status: "idle",
	output: "",
};

let processState: { status: "idle" | "running" | "done" | "error"; output: string } = {
	status: "idle",
	output: "",
};

/** Home directory for the current platform. */
function homeDir(): string {
	return Zotero.isWin
		? (Components.classes["@mozilla.org/process/environment;1"]
			.getService(Components.interfaces.nsIEnvironment)
			.get("USERPROFILE"))
		: (Components.classes["@mozilla.org/process/environment;1"]
			.getService(Components.interfaces.nsIEnvironment)
			.get("HOME"));
}

/** Path to the ~/.incite directory. */
function inciteDir(): string {
	return homeDir() + (Zotero.isWin ? "\\.incite" : "/.incite");
}

/** Path to the server log file. */
function serverLogPath(): string {
	return inciteDir() + (Zotero.isWin ? "\\server.log" : "/server.log");
}

/** Path to the uv binary. */
function uvBinPath(): string {
	return inciteDir() + (Zotero.isWin ? "\\bin\\uv.exe" : "/bin/uv");
}

/** Path to the dedicated incite virtual environment. */
function venvDir(): string {
	return inciteDir() + (Zotero.isWin ? "\\venv" : "/venv");
}

/** Path to the Python binary inside the venv. */
export function venvPython(): string {
	return venvDir() + (Zotero.isWin ? "\\Scripts\\python.exe" : "/bin/python3");
}

/** Run a command and capture its stdout via a temp file. */
async function runCommand(cmd: string, args: string[]): Promise<{ exitCode: number; stdout: string }> {
	const tmpFile = Zotero.getTempDirectory().path + "/incite_cmd_" + Date.now() + ".txt";
	const fullCmd = [cmd, ...args].map(a => a.includes(" ") ? `"${a}"` : a).join(" ");

	// exec() needs an absolute path; resolves true on exit 0, rejects on non-zero
	const shell = Zotero.isWin ? "C:\\Windows\\System32\\cmd.exe" : "/bin/sh";
	const shellArgs = Zotero.isWin
		? ["/c", `${fullCmd} > "${tmpFile}" 2>&1`]
		: ["-c", `${fullCmd} > "${tmpFile}" 2>&1`];

	let exitCode = -1;
	try {
		await Zotero.Utilities.Internal.exec(shell, shellArgs);
		exitCode = 0; // exec() resolves on success (exit 0)
	} catch (e) {
		// exec() rejects on non-zero exit — still check for output
		Zotero.debug(`inCite: exec non-zero or failed: ${e}`);
		exitCode = 1;
	}

	let stdout = "";
	try {
		stdout = await Zotero.File.getContentsAsync(tmpFile);
	} catch {
		// Temp file may not exist if command failed to start
	}

	return { exitCode, stdout: stdout.trim() };
}

/** Platform-aware Python candidate paths. */
function pythonCandidates(): string[] {
	const candidates: string[] = [];

	// Always check venv Python first
	candidates.push(venvPython());

	// Check uv-managed Python (glob pattern resolved at runtime)
	if (Zotero.isWin) {
		// uv installs Python to ~/.incite/python/cpython-3.12.*/python.exe on Windows
		// We check common paths
	} else {
		// uv installs to ~/.local/share/uv/python/cpython-3.12.*/bin/python3
		// but we told it to use our venv, so venvPython() covers it
	}

	if (Zotero.isWin) {
		candidates.push("python", "python3", "py");
	} else if (Zotero.isMac) {
		candidates.push("/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3", "python3");
	} else {
		candidates.push("/usr/bin/python3", "/usr/local/bin/python3", "python3");
	}

	return candidates;
}

/** Find a working Python 3 installation. */
export async function findPython(): Promise<{ found: boolean; path?: string; version?: string }> {
	for (const candidate of pythonCandidates()) {
		const { exitCode, stdout } = await runCommand(candidate, ["--version"]);
		if (exitCode === 0 && stdout.toLowerCase().includes("python 3")) {
			const version = stdout.replace(/^Python\s*/i, "").trim();
			return { found: true, path: candidate, version };
		}
	}
	return { found: false };
}

/** Check if incite is installed (checks venv first, then system Python). */
export async function checkIncite(pythonPath: string): Promise<{ installed: boolean; version?: string; pythonPath?: string }> {
	// Check venv first
	const vpy = venvPython();
	const venvResult = await runCommand(vpy, ["-m", "incite", "--version"]);
	if (venvResult.exitCode === 0 && venvResult.stdout) {
		const version = venvResult.stdout.replace(/^incite\s*/i, "").trim();
		return { installed: true, version, pythonPath: vpy };
	}

	// Fall back to system Python
	const { exitCode, stdout } = await runCommand(pythonPath, ["-m", "incite", "--version"]);
	if (exitCode === 0 && stdout) {
		const version = stdout.replace(/^incite\s*/i, "").trim();
		return { installed: true, version, pythonPath };
	}
	return { installed: false };
}

/** Get the current install state. */
export function getInstallState(): { status: string; output: string } {
	return { ...installState };
}

/** Download the uv binary to ~/.incite/bin/uv for the current platform. */
export async function downloadUv(): Promise<void> {
	const binDir = inciteDir() + (Zotero.isWin ? "\\bin" : "/bin");

	// Create ~/.incite/bin directory
	if (Zotero.isWin) {
		await runCommand("cmd.exe", ["/c", `mkdir "${binDir}"`]);
	} else {
		await runCommand("/bin/sh", ["-c", `mkdir -p "${binDir}"`]);
	}

	// Determine the download URL for the current platform
	let url: string;
	if (Zotero.isMac) {
		// Detect architecture via uname -m
		const { stdout: arch } = await runCommand("/usr/bin/uname", ["-m"]);
		if (arch.includes("arm64")) {
			url = "https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin.tar.gz";
		} else {
			url = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-apple-darwin.tar.gz";
		}
	} else if (Zotero.isWin) {
		url = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip";
	} else {
		// Linux
		url = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz";
	}

	if (Zotero.isWin) {
		// Download zip and extract on Windows
		const zipPath = binDir + "\\uv.zip";
		await runCommand("cmd.exe", ["/c", `curl -L -o "${zipPath}" "${url}"`]);
		await runCommand("cmd.exe", ["/c", `tar -xf "${zipPath}" -C "${binDir}" --strip-components=1`]);
		await runCommand("cmd.exe", ["/c", `del "${zipPath}"`]);
	} else {
		// Download tarball and extract on Unix — tarball contains uv-*/uv
		const { exitCode, stdout } = await runCommand("/bin/sh", [
			"-c",
			`curl -L "${url}" | tar xz -C "${binDir}" --strip-components=1`,
		]);
		if (exitCode !== 0) {
			throw new Error(`Failed to download uv: ${stdout}`);
		}
		// Ensure the binary is executable
		await runCommand("/bin/sh", ["-c", `chmod +x "${uvBinPath()}"`]);
	}
}

/** Ensure Python 3.12 is available via uv. */
export async function ensurePython(): Promise<void> {
	const uv = uvBinPath();
	const { exitCode, stdout } = await runCommand(uv, ["python", "install", "3.12"]);
	if (exitCode !== 0) {
		throw new Error(`Failed to install Python via uv: ${stdout}`);
	}
}

/** Install incite using uv (async — poll getInstallState for progress). */
export async function installIncite(): Promise<void> {
	if (installState.status === "running") return;
	installState = { status: "running", output: "Downloading uv package manager..." };

	try {
		// Step 1: Download uv if not already present
		const uv = uvBinPath();
		const uvCheck = await runCommand(uv, ["--version"]);
		if (uvCheck.exitCode !== 0) {
			await downloadUv();
			// Verify uv works
			const verify = await runCommand(uv, ["--version"]);
			if (verify.exitCode !== 0) {
				installState = { status: "error", output: verify.stdout || "Failed to install uv" };
				return;
			}
		}

		// Step 2: Ensure Python is available
		installState = { status: "running", output: "Ensuring Python 3.12 is available..." };
		await ensurePython();

		// Step 3: Create venv
		installState = { status: "running", output: "Creating virtual environment..." };
		const venv = venvDir();
		const venvResult = await runCommand(uv, ["venv", venv, "--python", "3.12"]);
		if (venvResult.exitCode !== 0) {
			installState = { status: "error", output: venvResult.stdout || "Failed to create virtual environment" };
			return;
		}

		// Step 4: Install incite[lite] into the venv
		installState = { status: "running", output: "Installing incite[lite]..." };
		const vpy = venvPython();
		const { exitCode, stdout } = await runCommand(uv, ["pip", "install", "incite[lite]", "--python", vpy]);
		if (exitCode === 0) {
			installState = { status: "done", output: stdout };
		} else {
			installState = { status: "error", output: stdout || "uv pip install failed" };
		}
	} catch (e) {
		installState = { status: "error", output: String(e) };
	}
}

/** Start the incite server as a background process (prefers venv Python). */
export async function startServer(pythonPath: string, embedder?: string): Promise<{ pid?: number }> {
	// Use venv Python if available, otherwise fall back to provided path
	const vpy = venvPython();
	const venvCheck = await runCommand(vpy, ["--version"]);
	const py = venvCheck.exitCode === 0 ? vpy : pythonPath;

	const emb = embedder ?? "minilm-ft-onnx";
	const pidFile = Zotero.getTempDirectory().path + "/incite_server.pid";
	const logFile = serverLogPath();

	if (Zotero.isWin) {
		// On Windows, use start /B
		await runCommand("cmd.exe", [
			"/c",
			`start /B ${py} -m incite serve --embedder ${emb} > "${logFile}" 2>&1 & echo %errorlevel% > "${pidFile}"`,
		]);
	} else {
		// On Unix, use nohup + background, capture PID, log output to file
		await runCommand("/bin/sh", [
			"-c",
			`nohup ${py} -m incite serve --embedder ${emb} > "${logFile}" 2>&1 & echo $! > "${pidFile}"`,
		]);
	}

	try {
		const pidStr = await Zotero.File.getContentsAsync(pidFile);
		const pid = parseInt(pidStr.trim(), 10);
		if (!isNaN(pid)) {
			managedServerPid = pid;
			return { pid };
		}
	} catch {
		// PID file may not exist
	}

	return {};
}

/** Stop the managed server process if one is running. */
export async function stopManagedServer(): Promise<void> {
	if (managedServerPid == null) return;

	try {
		if (Zotero.isWin) {
			await runCommand("taskkill", ["/F", "/PID", String(managedServerPid)]);
		} else {
			await runCommand("kill", [String(managedServerPid)]);
		}
	} catch (e) {
		Zotero.debug(`inCite: failed to stop server (PID ${managedServerPid}): ${e}`);
	}

	managedServerPid = null;
}

/** Get the managed server PID (if any). */
export function getManagedServerPid(): number | null {
	return managedServerPid;
}

/** Get the current library processing state. */
export function getProcessState(): { status: string; output: string } {
	return { ...processState };
}

/** Check if a FAISS index exists for the given embedder. */
export async function checkIndexExists(embedder?: string): Promise<boolean> {
	const emb = embedder ?? "minilm-ft-onnx";
	const indexPath = inciteDir() + (Zotero.isWin ? "\\" : "/")
		+ `zotero_index_${emb}` + (Zotero.isWin ? "\\" : "/") + "index.faiss";

	const { exitCode } = Zotero.isWin
		? await runCommand("cmd.exe", ["/c", `if exist "${indexPath}" (echo yes) else (echo no)`])
		: await runCommand("/bin/sh", ["-c", `test -f "${indexPath}"`]);

	return exitCode === 0;
}

/** Read the last N lines from the server log file. */
export async function getServerLog(lines: number = 50): Promise<string> {
	const logFile = serverLogPath();
	try {
		if (Zotero.isWin) {
			const { exitCode, stdout } = await runCommand("cmd.exe", [
				"/c",
				`powershell -Command "Get-Content '${logFile}' -Tail ${lines}"`,
			]);
			return exitCode === 0 ? stdout : "";
		} else {
			const { exitCode, stdout } = await runCommand("/bin/sh", [
				"-c",
				`tail -n ${lines} "${logFile}"`,
			]);
			return exitCode === 0 ? stdout : "";
		}
	} catch {
		return "";
	}
}

/** Process the Zotero library by starting the server (which auto-builds the index). */
export async function processLibrary(embedder?: string): Promise<void> {
	if (processState.status === "running") return;
	processState = { status: "running", output: "Checking Python environment..." };

	try {
		// Step 1: Resolve Python path (prefer venv)
		const vpy = venvPython();
		const venvCheck = await runCommand(vpy, ["--version"]);
		if (venvCheck.exitCode !== 0) {
			processState = { status: "error", output: "Python venv not found. Please install incite first." };
			return;
		}

		// Step 2: Check if index already exists
		const emb = embedder ?? "minilm-ft-onnx";
		const hasIndex = await checkIndexExists(emb);

		if (hasIndex) {
			processState = { status: "running", output: "Index found. Starting server..." };
		} else {
			processState = { status: "running", output: "Building index from Zotero library (this may take a few minutes)..." };
		}

		// Step 3: Start the server (which builds the index on first run)
		// startServer() already logs output to ~/.incite/server.log
		const result = await startServer(vpy, emb);

		if (result.pid) {
			processState = { status: "done", output: `Server running (PID ${result.pid})` };
		} else {
			// Check log for errors
			const log = await getServerLog(20);
			processState = { status: "error", output: log || "Server failed to start (no PID captured)" };
		}
	} catch (e) {
		processState = { status: "error", output: String(e) };
	}
}
