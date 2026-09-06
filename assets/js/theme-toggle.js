document.addEventListener("DOMContentLoaded", function () {
  const button = document.querySelector(".theme__toggle");
  const stylesheet = document.getElementById("dark-mode-stylesheet");

  if (!button || !stylesheet) return;

  const icon = button.querySelector("i");

  function applyTheme(isDark) {
    stylesheet.disabled = !isDark;

    document.documentElement.dataset.theme = isDark ? "dark" : "light";
    localStorage.setItem("abr-theme", isDark ? "dark" : "light");

    if (isDark) {
      icon.className = "fas fa-sun";
      button.setAttribute("aria-label", "Switch to light mode");
      button.setAttribute("title", "Switch to light mode");
    } else {
      icon.className = "fas fa-moon";
      button.setAttribute("aria-label", "Switch to dark mode");
      button.setAttribute("title", "Switch to dark mode");
    }
  }

  const isDark = localStorage.getItem("abr-theme") === "dark";
  applyTheme(isDark);

  button.addEventListener("click", function () {
    applyTheme(stylesheet.disabled);
  });
});