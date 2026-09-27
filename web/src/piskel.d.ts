// The globals the vendored Piskel bundle (public/piskel/piskel.js) leaves on
// window. Piskel is untyped JavaScript; only the surface piskel-bridge.ts
// touches is named here, everything else is `any` on purpose - a wider
// declaration would be a promise the bundle does not make.

interface PiskelFrame {
  getWidth(): number;
  getHeight(): number;
  getHash(): string;
}

interface PiskelLayer {
  getName(): string;
  getFrames(): PiskelFrame[];
  getFrameAt(index: number): PiskelFrame;
  addFrame(frame: PiskelFrame): void;
  getOpacity(): number;
  size(): number;
}

interface PiskelModel {
  getWidth(): number;
  getHeight(): number;
  getFPS(): number;
  getLayers(): PiskelLayer[];
  getFrameCount(): number;
  addLayer(layer: PiskelLayer): void;
}

interface PiskelController {
  getPiskel(): PiskelModel;
  setPiskel(piskel: PiskelModel, options?: { preserveState?: boolean; noSnapshot?: boolean }): void;
  getLayers(): PiskelLayer[];
  getCurrentLayer(): PiskelLayer;
  getFrameCount(): number;
  getWidth(): number;
  getHeight(): number;
  getFPS(): number;
  setFPS(fps: number): void;
  setCurrentFrameIndex(index: number): void;
}

interface PiskelPalette {
  id: string;
  name: string;
  getColors(): string[];
  setColors(colours: string[]): void;
}

interface PiskelNamespace {
  app: {
    init(): void;
    piskelController: PiskelController;
    paletteService: {
      dynamicPalettes: PiskelPalette[];
      addDynamicPalette(palette: PiskelPalette): void;
      getPaletteById(id: string): PiskelPalette | null;
    };
    palettesListController: { selectPalette(id: string): void };
    framesListController: { flagForRedraw_(regenerateDom?: boolean): void };
    savedStatusService: { isDirty(): boolean };
    /** The sprite studio fork's hook (vendor/piskel/CHANGES.md). */
    frameTagProvider?: ((index: number) => string | null) | null;
    [key: string]: unknown;
  };
  model: {
    Piskel: new (width: number, height: number, fps: number, descriptor: unknown) => PiskelModel;
    Layer: new (name: string) => PiskelLayer;
    Frame: new (width: number, height: number) => PiskelFrame;
    Palette: new (id: string, name: string, colours: string[]) => PiskelPalette;
    piskel: { Descriptor: new (name: string, description: string, isPublic?: boolean) => unknown };
  };
  utils: {
    FrameUtils: {
      createFromImage(image: HTMLImageElement | HTMLCanvasElement, preserveOpacity?: boolean): PiskelFrame;
      toImage(frame: PiskelFrame, zoom?: number, opacity?: number): HTMLCanvasElement;
    };
    LayerUtils: {
      flattenFrameAt(layers: PiskelLayer[], index: number, preserveOpacity: boolean): HTMLCanvasElement;
    };
  };
  UserSettings: {
    get(key: string): unknown;
    set(key: string, value: unknown): void;
    SELECTED_PALETTE: string;
  };
  [key: string]: unknown;
}

declare const pskl: PiskelNamespace;

/** Piskel's jQuery + pubsub (js/lib/pubsub.js): the event bus. */
declare const $: {
  publish(event: string, args?: unknown): void;
  subscribe(event: string, handler: (...args: unknown[]) => void): void;
  (selector: string): { tooltip(options: unknown): unknown };
};

declare const Events: Record<string, string>;
declare const Constants: Record<string, unknown> & { CURRENT_COLORS_PALETTE_ID: string };

interface Window {
  pskl: PiskelNamespace;
}
