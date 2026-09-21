// The shell's entry: boot the vendored Piskel, then mount the studio panel
// beside it. Piskel's bundle is a classic <script> in index.html and has run
// by the time this module does (modules are deferred), so `pskl` is here.

import "./studio.css";
import { StudioPanel } from "./panel";
import { bootPiskel } from "./piskel-bridge";

const COLLAPSED_KEY = "anim-studio.panel-collapsed";

function mountPanel(): void {
  const body = document.getElementById("studio-body");
  const health = document.getElementById("studio-health");
  const toggle = document.getElementById("studio-toggle");
  if (!body || !health || !toggle) throw new Error("index.html lost the studio panel's markup");

  const setCollapsed = (collapsed: boolean) => {
    document.body.classList.toggle("studio-collapsed", collapsed);
    toggle.textContent = collapsed ? "▶" : "◀";
    try {
      localStorage.setItem(COLLAPSED_KEY, collapsed ? "1" : "0");
    } catch {
      // private mode; the choice just does not survive a reload
    }
  };
  let collapsed = false;
  try {
    collapsed = localStorage.getItem(COLLAPSED_KEY) === "1";
  } catch {
    collapsed = false;
  }
  setCollapsed(collapsed);
  toggle.addEventListener("click", () => setCollapsed(!document.body.classList.contains("studio-collapsed")));
  document.addEventListener("keydown", (event) => {
    if (event.altKey && event.key.toLowerCase() === "p") {
      event.preventDefault();
      setCollapsed(!document.body.classList.contains("studio-collapsed"));
    }
  });

  const panel = new StudioPanel(body, health);
  void panel.start();
}

function main(): void {
  try {
    bootPiskel();
  } catch (error) {
    const mask = document.getElementById("loading-mask");
    if (mask) mask.textContent = error instanceof Error ? error.message : String(error);
    throw error;
  }
  mountPanel();
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", main);
else main();
