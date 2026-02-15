export function showToast(message: string, duration = 3000): void {
  const existing = document.getElementById("incite-toast");
  if (existing) existing.remove();

  const toast = document.createElement("div");
  toast.id = "incite-toast";
  toast.textContent = message;
  Object.assign(toast.style, {
    position: "fixed",
    bottom: "20px",
    right: "20px",
    padding: "10px 18px",
    background: "#1a1a2e",
    color: "#fff",
    borderRadius: "8px",
    fontSize: "13px",
    zIndex: "999999",
    boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
  });
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), duration);
}
