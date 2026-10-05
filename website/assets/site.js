if (navigator.clipboard) {
  document.querySelectorAll("[data-copy]").forEach((button) => {
    button.hidden = false;
    button.addEventListener("click", async () => {
      const status = document.getElementById(button.dataset.status);
      const source = document.getElementById(button.dataset.copy);
      try {
        await navigator.clipboard.writeText(source.textContent);
        status.textContent = "Code copied to clipboard.";
      } catch {
        status.textContent = "Select the code above to copy it manually.";
      }
    });
  });
}
