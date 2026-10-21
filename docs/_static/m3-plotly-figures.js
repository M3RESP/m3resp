// The plotly figures of the tutorials have a transparent background, so they
// show the page background. This file gives the text and the grid lines of
// each figure the colours of the light or the dark theme, and fits the figure
// to the width of the text column. docs/_ext/plotly_gallery.py calls
// m3PlotlyFigure for each figure.

// Return a CSS colour variable of the page as "rgb(...)" or "rgba(...)",
// the colour format that plotly reads.
function m3ThemeColor(name) {
  const probe = document.createElement("span");
  probe.style.color = `var(${name})`;
  document.body.appendChild(probe);
  const color = getComputedStyle(probe).color;
  probe.remove();
  return color;
}

function m3ApplyThemeColors(figure) {
  const text = m3ThemeColor("--m3-c-text");
  const line = m3ThemeColor("--m3-c-line");
  const lineStrong = m3ThemeColor("--m3-c-line-strong");
  const update = { "font.color": text };
  for (const key of Object.keys(figure._fullLayout)) {
    if (/^[xy]axis[0-9]*$/.test(key)) {
      update[`${key}.gridcolor`] = line;
      update[`${key}.linecolor`] = lineStrong;
      update[`${key}.zerolinecolor`] = lineStrong;
    }
  }
  Plotly.relayout(figure, update);
}

function m3PlotlyFigure(figure) {
  m3ApplyThemeColors(figure);
  // The theme switch changes the data-theme attribute of the page.
  new MutationObserver(() => m3ApplyThemeColors(figure)).observe(
    document.documentElement,
    { attributes: true, attributeFilter: ["data-theme"] },
  );
  // The page scripts change the width of the text column after plotly draws
  // the figure, so plotly draws the figure again at each change of width.
  new ResizeObserver(() => Plotly.Plots.resize(figure)).observe(
    figure.parentElement,
  );
}
