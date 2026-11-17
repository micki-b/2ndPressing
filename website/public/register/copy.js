// "Copy code" on the confirmation page (shown only when scripts run).
(() => {
  const button = document.getElementById("copy");
  const code = document.getElementById("code");
  if (!button || !code) return;
  button.hidden = false;
  const label = button.querySelector(".lbl");
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(code.textContent.trim());
      label.textContent = "Copied";
      setTimeout(() => { label.textContent = "Copy code"; }, 2500);
    } catch {
      // No clipboard access: select the code so Cmd/Ctrl+C works.
      const range = document.createRange();
      range.selectNodeContents(code);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      label.textContent = "Press Cmd+C or Ctrl+C";
    }
  });
})();
