// "On this page" marks the first section of the page when the page opens.
// When you scroll, the theme moves the mark to the section at the top of the window.
window.addEventListener("load", () => {
  const nav = document.getElementById("pst-page-toc-nav");
  if (!nav || nav.querySelector("a.nav-link.active")) {
    return;
  }
  const first = nav.querySelector("a.nav-link");
  if (!first) {
    return;
  }
  first.classList.add("active");
  first.setAttribute("aria-current", "true");
  const entry = first.closest(".toc-entry");
  if (entry) {
    entry.classList.add("active");
  }
});
