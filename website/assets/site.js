const copyButton = document.querySelector("#copy-install");

if (navigator.clipboard && copyButton) {
  copyButton.hidden = false;
  copyButton.addEventListener("click", async () => {
    const status = document.querySelector("#copy-status");
    try {
      await navigator.clipboard.writeText(
        document.querySelector("#install-command").textContent,
      );
      status.textContent = "Install commands copied to clipboard.";
    } catch {
      status.textContent = "Select the commands above to copy them manually.";
    }
  });
}
