# Changes to the vendored Piskel

Upstream is pinned in `UPSTREAM`. Everything under `src/` is upstream's file
byte for byte except what is listed here, so a re-pull is: copy upstream's
`src/` over this one, then re-apply this list.

## Generated files committed (not edits)

`src/img/icons.png`, `src/img/icons@2x.png`, `src/css/icons.css` are outputs
of upstream's `scripts/sprite.js` (spritesmith over `src/img/icons/`), which
upstream runs as `npm run build:sprites` and does not commit. They are committed
here so the shell's build needs neither spritesmith nor its 114-package tree.
Regenerate after an icon change with upstream's script:

    NODE_PATH=<dir with spritesmith@3.5.1 + spritesheet-templates@10.5.2> \
      node scripts/sprite.js        # from an upstream checkout

## Edits

### `src/js/controller/FramesListController.js` — a text tag per timeline tile

Adds `getTileTag_(index)`, which asks `pskl.app.frameTagProvider(index)` when
the shell has set one, and a `.tile-overlay.tile-tag` element on each preview
tile that shows the answer (created in `createPreviewTile_`, refreshed in
`updatePreviews_`). Upstream shows only the 1-based frame number; the studio
shows the keyframe phase (`wind-up`, `action`, ...) beside it. No behaviour
changes when no provider is set. Styling lives in the shell
(`web/src/studio.css`), not in upstream's CSS.

## Not used

`src/index.html` is read for its body markup only (between
`<!--body-main-start-->` and `<!--body-main-end-->`, which upstream keeps for
exactly this purpose); its `<head>`, loading mask and the inlined
`piskel-boot.js` are not - the shell boots Piskel itself
(`web/src/piskel-bridge.ts`). `src/piskel-boot.js` is kept for the re-pull diff.
