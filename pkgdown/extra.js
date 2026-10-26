// pkgdown 2.2.1 can search before its first-focus index fetch has completed.
// Refresh a query entered during that fetch once the index is ready.
$(function () {
  const input = document.getElementById("search-input");
  if (!input) return;

  let dismissed = false;
  const resume = () => {
    if (dismissed || input.value.length < 2) return;
    const query = input.value;
    const start = input.selectionStart;
    const end = input.selectionEnd;
    const direction = input.selectionDirection;
    $(input).autocomplete("val", "").autocomplete("val", query);
    input.setSelectionRange(start, end, direction);
  };
  const loaded = new MutationObserver(() => {
    if (input.classList.contains("loading")) return;
    loaded.disconnect();
    if (document.activeElement === input) {
      resume();
    } else {
      input.addEventListener("focus", resume, { once: true });
    }
  });

  input.addEventListener("focus", () => {
    dismissed = false;
    if (input.classList.contains("loading")) {
      loaded.observe(input, { attributes: true, attributeFilter: ["class"] });
    }
  });
  input.addEventListener("input", () => { dismissed = false; });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      dismissed = true;
      // Let autocomplete close the list without the search field clearing itself.
      event.preventDefault();
    }
  });
});
