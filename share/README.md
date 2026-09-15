# The Mac edit share feature

Shares `game-02-tactical-rpg/assets/` over the network so you can open the unit PNGs in
**Pixelorama on the Mac**, save them, and have this Fedora host see the new bytes straight away —
same files, same paths, no copying back and forth. Commits still happen on Fedora.

Runs as one small Samba container on **Docker Desktop** (context `desktop-linux`). Nothing here
touches the native Docker daemon.

---

## The address to connect to

```
smb://192.168.1.158:8445/game02-assets
```

**Why the `:8445`.** Port 445 is the normal SMB port, but this host is already running its own
Samba (the systemd `smb.service` that serves the `Storage`, `Private` and `homes` shares). It owns
445 and 139, so this container publishes **8445** instead. Finder is fine with a port in the URL —
you just have to type the URL rather than pick the machine out of the sidebar. Network browsing
(the "Shared" list in Finder) will not show this share, because discovery can only ever advertise
port 445.

Two shares are exported:

| Share | Host path | What's in it |
|---|---|---|
| `game02-assets` | `game-02-tactical-rpg/assets/` | the PNGs the game actually loads — this is the one you edit |
| `game02-assets-src` | `game-02-tactical-rpg/assets_src/` | source art, the palette (`palette.gpl` / `palette.png`), extraction scripts |

Both are read-write.

---

## Mounting it from the Mac

1. In Finder: **Go → Connect to Server…** (or `⌘K`).
2. Type `smb://192.168.1.158:8445/game02-assets` and hit **Connect**.
3. Choose **Registered User** (not Guest — guests are refused on purpose).
   - **Name:** `workstation`
   - **Password:** see below.
4. Tick *Remember this password in my keychain* so you only do this once.
5. Optional: drag the mounted volume into the Finder sidebar, or **System Settings → General →
   Login Items** → add it, so it reconnects after a reboot.

**Where the password is.** `tools/anim-pipeline/share/.env`, on the `SMB_PASS` line. It was
generated randomly when this was set up and is not any of your other passwords. That file is
gitignored (see `.gitignore` here) and is mode `600`, so it never leaves the machine.

```bash
grep SMB_PASS /home/workstation/Repos/Game-Dev/tools/anim-pipeline/share/.env
```

To change it: edit `.env`, then `docker compose -f share/compose.yml up -d --force-recreate`.
You will also need to forget the old one in Keychain Access on the Mac.

`.env.example` is the committed template and must never hold the real password.

---

## Editing in Pixelorama

Pixelorama is free and on the Mac App Store / itch.io. Open the mounted volume, open a PNG,
change it, **save over the same file** (`⌘S`, not "Save As" into some other folder) — that is what
makes the host see it.

Three rules so the edits stay usable by the rest of the pipeline:

- **Keep the canvas 64×64.** The asset audit compares every PNG against `contract.canvas` in its
  sidecar and fails the build if they disagree. If you need room to work, zoom in; don't resize.
- **Save as PNG.** Pixelorama's own `.pxo` project format is fine as a working file, but the thing
  the game loads has to be the `.png`. If you keep a `.pxo`, put it in `assets_src/`, not `assets/`.
- **No dithering, no anti-aliasing, no soft brushes.** The palette is locked — every colour has to
  be one that's already in `assets_src/palette.png`. Dithering and soft edges invent in-between
  colours, the palette check then fails, and the sprite stops matching everything around it. Use
  the pencil and the bucket with tolerance 0, and load the palette into Pixelorama
  (**Palettes → Import** → `palette.gpl` from the `game02-assets-src` share) so you're only ever
  clicking colours that exist.

Alpha is treated as on-or-off, never partial. Erase with a hard eraser.

---

## Things to know

**Finder droppings can't get into the repo.** `.DS_Store`, `._*` and the rest of the macOS
metadata files are vetoed by the share — Finder's attempt to write one is simply refused, and it
carries on without complaining. Verified: writing `.DS_Store` and `._sneaky.png` through the share
both fail and nothing appears on the host.

**Mac file metadata isn't stored.** Finder tags, colour labels and resource forks won't stick.
Docker Desktop's bind mount can't hold extended attributes, which is where that information
normally goes. It has no effect on pixels. The long comment in `smb.conf` explains why
`streams_xattr` is deliberately left out of the config — with it loaded, *every rename on the
share fails*, which is much worse than losing colour labels.

**The stale-content quirk.** After a save that goes write-temp-then-rename, a *container* reading
the same directory through Docker Desktop's file sharing can keep seeing the old contents for a
while — it caches by inode and the rename gives the file a new one. Two things to know:

- **Godot and anything else on this host are not affected.** They read the real filesystem
  directly and see your edit immediately.
- The pipeline's `publish` and audit steps re-copy each file in place before reading it, which
  forces a fresh view. So if a container ever seems to be looking at an old version of a sprite,
  that's this quirk, not a lost save. `docker restart anim-pipeline-samba` clears it too.

**Everything is owned by you.** Files written through the share land on the host as
`workstation:workstation` (1000:1000), mode `0644`, so git sees a normal modification and you
never need to `chown` anything.

---

## Starting and stopping

Run these from `tools/anim-pipeline/`:

```bash
# start (also what you run after editing smb.conf or .env)
docker compose -f share/compose.yml up -d

# stop, and remove the container
docker compose -f share/compose.yml down

# just pause it
docker compose -f share/compose.yml stop
docker compose -f share/compose.yml start

# is it healthy? (healthy = a real login against it succeeded)
docker ps --filter name=anim-pipeline-samba

# what Samba is saying
docker compose -f share/compose.yml logs -f
```

It has `restart: unless-stopped`, so it comes back by itself when Docker Desktop starts.

Only ever bare `docker compose` — never `--context docker-default`, and never `docker context use`.

---

## Firewall

**Nothing to do.** Checked on 2026-09-13: firewalld is active, the default zone is
`FedoraWorkstation` on `br0` + `enp4s0`, and that zone already opens `1025-65535/tcp`, which
covers 8445. `firewall-cmd --query-port=8445/tcp` answers `yes`.

If that ever changes, or if this share is moved to port 445, the fix is:

```bash
sudo firewall-cmd --add-service=samba --permanent && sudo firewall-cmd --reload
```

(`samba` is already in the allowed service list, which is how the host's own Samba is reachable.)

---

## If you'd rather not run a container at all

This host's existing `smb.service` already exports your home directory as `homes` for the user
`workstation`, on the normal port 445. So `smb://192.168.1.158/workstation` gets you to
`Repos/Game-Dev/game-02-tactical-rpg/assets/` today, with no container, real extended attributes,
no port in the URL, and no stale-content quirk.

The trade-off: `[homes]` exposes your whole home directory rather than just the assets folder, and
it has none of the Mac tuning or the `.DS_Store` veto that this share has — Finder *would* drop
metadata files into the repo. Adding a proper `[game02-assets]` section to `/etc/samba/smb.conf`
(copying the options from the `Storage` share, which is already Mac-tuned) would be the best of
both, but that file is host configuration outside this repo, so it was left alone here. Worth
doing if you end up living in this share.

---

## Files here

| File | What it is |
|---|---|
| `compose.yml` | the container: image digest, the two bind mounts, port 8445, uid/gid 1000 |
| `smb.conf` | the Samba config, mounted read-only into the container |
| `.env` | `SMB_USER` / `SMB_PASS` / `SMB_PORT` — **gitignored**, mode 600 |
| `.env.example` | committed template, no real password |
| `.gitignore` | keeps `.env` out of git |

Image: `dockurr/samba@sha256:404f91bdf317d390241a41165579cd25199f34171b44677d877c75dc40184042`
(Samba 4.23.10, MIT, rebuilt 2026-08-22). Pinned by digest so an upstream rebuild can't silently
change what runs.
