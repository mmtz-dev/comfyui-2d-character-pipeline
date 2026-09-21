// Everything that touches Piskel's globals lives here, so the panel and the
// pure modules never do. Piskel's model is the editor's truth: a job's frames
// become one pskl.model.Piskel (one layer, one frame per PNG, the job's fps)
// and Save reads them back out flattened. Nothing is kept on our side.

import { binariseAlpha, type AlphaReport } from "./pixels";

/** Boot Piskel once its bundle and markup are in the page (index.html). */
export function bootPiskel(): void {
  if (typeof pskl === "undefined" || !pskl.app) {
    throw new Error("Piskel did not load: /piskel/piskel.js is missing - run `npm run build`");
  }
  pskl.app.init();
  const mask = document.getElementById("loading-mask");
  if (mask) {
    mask.style.opacity = "0";
    window.setTimeout(() => mask.remove(), 400);
  }
}

export function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`could not load ${url}`));
    image.src = url;
  });
}

export interface LoadedAnimation {
  width: number;
  height: number;
  frames: number;
}

/**
 * Show the PNGs as one animation: one layer, frame k = the k-th PNG, at
 * `fps`. Every PNG must be the same size (a set from the pipeline is).
 * Alpha is rounded to 0/255 by Piskel on import (FrameUtils.createFromImage
 * without preserveOpacity), which matches the frame contract.
 */
export async function openAnimation(name: string, urls: string[], fps: number): Promise<LoadedAnimation> {
  if (!urls.length) throw new Error("no frames to open");
  const images = await Promise.all(urls.map(loadImage));
  const first = images[0]!;
  const odd = images.find((image) => image.width !== first.width || image.height !== first.height);
  if (odd) {
    throw new Error(`frames differ in size: ${first.width}x${first.height} and ${odd.width}x${odd.height}`);
  }
  const layer = new pskl.model.Layer(name);
  for (const image of images) layer.addFrame(pskl.utils.FrameUtils.createFromImage(image));
  const descriptor = new pskl.model.piskel.Descriptor(name, "", false);
  const piskel = new pskl.model.Piskel(first.width, first.height, fps, descriptor);
  piskel.addLayer(layer);
  pskl.app.piskelController.setPiskel(piskel);
  pskl.app.piskelController.setFPS(fps);
  return { width: first.width, height: first.height, frames: images.length };
}

export interface ExportedFrame {
  index: number;
  width: number;
  height: number;
  alpha: AlphaReport;
  png: Blob;
}

function toBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) =>
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("toBlob failed"))), "image/png"),
  );
}

/**
 * Every frame of the animation, all layers flattened (layer opacity
 * applied), alpha rounded to 0/255 with transparent pixels zeroed - the
 * frame contract, applied here so the API receives what it would have made.
 */
export async function exportFrames(): Promise<ExportedFrame[]> {
  const controller = pskl.app.piskelController;
  const layers = controller.getLayers();
  const count = controller.getFrameCount();
  const out: ExportedFrame[] = [];
  for (let index = 0; index < count; index += 1) {
    const canvas = pskl.utils.LayerUtils.flattenFrameAt(layers, index, true);
    const context = canvas.getContext("2d");
    if (!context) throw new Error("no 2d context");
    const data = context.getImageData(0, 0, canvas.width, canvas.height);
    const alpha = binariseAlpha(data.data);
    context.putImageData(data, 0, 0);
    out.push({ index, width: canvas.width, height: canvas.height, alpha, png: await toBlob(canvas) });
  }
  return out;
}

export function frameCount(): number {
  return pskl.app.piskelController.getFrameCount();
}

export function frameSize(): [number, number] {
  const controller = pskl.app.piskelController;
  return [controller.getWidth(), controller.getHeight()];
}

/** The timeline's tag per frame (the fork's hook, vendor/piskel/CHANGES.md). */
export function setFrameTags(tags: (string | null)[]): void {
  pskl.app.frameTagProvider = tags.length ? (index) => tags[index] ?? null : null;
  pskl.app.framesListController.flagForRedraw_();
}

export const PROJECT_PALETTE_ID = "__anim-project";

/**
 * The project palette in Piskel's palette list, selected. A dynamic palette
 * (like Piskel's own "Current colors"), so it is never written to the
 * browser's saved palettes and disappears with the page.
 */
export function setProjectPalette(name: string, colours: string[]): void {
  const service = pskl.app.paletteService;
  const existing = service.dynamicPalettes.find((palette) => palette.id === PROJECT_PALETTE_ID);
  if (existing) {
    existing.name = name;
    existing.setColors(colours.slice());
  } else {
    service.addDynamicPalette(new pskl.model.Palette(PROJECT_PALETTE_ID, name, colours.slice()));
  }
  $.publish(Events.PALETTE_LIST_UPDATED!);
  pskl.app.palettesListController.selectPalette(PROJECT_PALETTE_ID);
}

/** Subscribe to Piskel's bus (a PISKEL_RESET after a load, a save state). */
export function onPiskelEvent(name: string, handler: () => void): void {
  const event = Events[name];
  if (!event) throw new Error(`Piskel has no event ${name}`);
  $.subscribe(event, handler);
}
