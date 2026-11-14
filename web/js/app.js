// Loads the pre-processed data once, then renders a page per hash route: #/<page>?<params>.
import { destroyCharts, esc } from "./ui.js";
import * as overview from "./pages/overview.js";
import * as records from "./pages/records.js";
import * as videos from "./pages/videos.js";

export const pages = {
  overview: { title: "Overview", mod: overview },
  records: { title: "Recognition records", mod: records },
  videos: { title: "Video playback", mod: videos },
};

const view = document.getElementById("view");
const title = document.getElementById("page-title");
const sidebar = document.getElementById("sidebar");
const menuBtn = document.querySelector(".menu-btn");
let data;

async function loadData() {
  const out = {};
  for (const name of ["records", "summary", "metrics"]) {
    const file = `data/${name}.json`;
    let res;
    try {
      res = await fetch(file);
    } catch {
      throw new Error(file);
    }
    if (!res.ok) throw new Error(`${file} (HTTP ${res.status})`);
    out[name] = await res.json();
  }
  return out;
}

function setMenu(open) {
  sidebar.classList.toggle("open", open);
  menuBtn.setAttribute("aria-expanded", String(open));
}

function route() {
  const [path, query = ""] = location.hash.replace(/^#\/?/, "").split("?");
  let name = path;
  if (!pages[name]) {
    name = "overview";
    history.replaceState(null, "", "#/overview");
  }
  destroyCharts();
  for (const a of sidebar.querySelectorAll("nav a")) {
    if (a.dataset.page === name) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
  title.textContent = pages[name].title;
  document.title = `${pages[name].title}: LPR Console`;
  view.innerHTML = "";
  pages[name].mod.render(view, data, new URLSearchParams(query));
  setMenu(false);
  window.scrollTo(0, 0);
}

menuBtn.addEventListener("click", () => setMenu(!sidebar.classList.contains("open")));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") setMenu(false);
});

loadData()
  .then((d) => {
    data = d;
    route();
    window.addEventListener("hashchange", route);
  })
  .catch((e) => {
    view.innerHTML = `<p class="state error">Failed to load data: ${esc(e.message)}. Serve the web/ folder over HTTP (python -m http.server -d web).</p>`;
  });
