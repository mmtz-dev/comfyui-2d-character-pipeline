# How do real games and AI game devs animate 2D characters, and what is our pipeline missing?

## Short answer
Shipped pixel games mostly hand-key a small number of frames (or render a rigged 3D model with no anti-aliasing), then control timing per frame; AI devs mostly do image-gen, image-to-video, frame extraction, background removal, then a fix-up pass for grid, jitter and palette. The main gaps in our pipeline versus both are no keyed extremes, no enforced loop closure, and per-frame (not per-animation) grid and palette handling. Evidence quality varies: primary sources are cited where found; vendor blogs and secondary pages are labelled, and Reddit could not be read at all (see Blocked).

## 1. How games animate 2D characters

Frame-by-frame pixel sprites
- Owlboy's art director says the whole game is made "pixel by pixel", with no tracing or 3D captures and a palette as limited as he could allow; rotation was the one concession. (Kill Screen, as relayed by search excerpt, secondary) https://www.killscreen.com/owlboy/
- Hollow Knight / Silksong: the Australian museum ACMI says Team Cherry made "hundreds of hand-drawn sprites" for Hornet. Claims that frames were drawn in Photoshop come only from SEO sites and are UNVERIFIED. https://www.acmi.net.au/works/125388--hollow-knight-silksong-game-worlds-acmi/
- Celeste and Hyper Light Drifter: no primary source on their pipelines was found (see Blocked). Not claiming anything beyond: Hyper Light Drifter was built in GameMaker and layers soft lighting/glow over limited-palette pixels, per press and a Steam user (weak). https://avclub.com/hyper-light-drifter-blended-past-and-present-to-become-1798255620

Skeletal / cutout
- Spine's own page: it stores "only the bone data, which is very small" instead of an image per frame; animations "can be reused for characters that look different"; animations can be blended (shoot while walking); meshes deform with bone weights. https://us.esotericsoftware.com/spine-in-depth
- Darkest Dungeon uses Spine (a .atlas, .png and .skel per move), per a modder on the Esoteric forum; heroes' skeletons can be imported into Spine. https://de.esotericsoftware.com/forum/d/6865-game-engines-that-work-with-skel and https://steamcommunity.com/app/262060/discussions/4/1727575977545379683
- Rayman Legends / UbiArt: any scanned or digitised art is split into pieces and attached to a skeleton; "all the animator has to do is design the animation poses, and the tool takes care of the image deformation" (UbiArt blog quoted by Giant Bomb). Main characters can have hundreds of animations (EveryEye, secondary, Italian). https://giantbomb.com/wiki/Concepts/Ubiart_Framework
- Hollow Knight is NOT confirmed as skeletal or frame-by-frame by any primary source found; I did not verify the tween/skeleton question.

3D rendered to sprites
- Dead Cells (Thomas Vasseur, Game Developer): model and skeleton built in 3DS Max from a pixel model sheet, exported as FBX, rendered at low resolution with no anti-aliasing by an in-house tool; each frame exported as PNG plus normal map; a toon shader adds volume. Reasons given: "No need to redraw each frame", reuse of animations across models, "automatic generation of the interpolation frames", quick retakes. Keyframes use "the least amount of frames possible". In-game character height "only 50 pixels". Flickering pixels are left unfixed: "we could clean that by hand... but the point of this workflow is to go fast." The 30 fps figure in the article belongs to the earlier ScarKrow prototype. https://www.gamedeveloper.com/production/art-design-deep-dive-using-a-3d-pipeline-for-2d-animation-in-i-dead-cells-i-
- Donkey Kong Country: Rare modelled characters in NURBS in Alias PowerAnimator on Silicon Graphics workstations, rendered animations frame by frame (hours per render, often overnight), then used a compression technique called ACM to fit them into SNES sprite memory; the technique had earlier appeared in the Amiga game Stardust (1993). https://en.wikipedia.org/wiki/Donkey_Kong_Country
- Diablo II as a 3D-rendered-sprite example: not confirmed by any fetched source (the Wikipedia page did not cover it). Left unverified.
- Rotoscoping and Mixamo-style hybrids: no primary source gathered for games; not claimed.

Pixel-art conventions (numbers are from a VENDOR blog, Sprite AI, which sells animation tools; treat as rules of thumb, not studio data)
- Frames per action: idle 2-4, walk 4-8 (4 for small sprites, 6-8 for 32x32+), run 6-8, jump 3-5, attack 3-6. https://www.sprite-ai.art/blog/sprite-animation-frames
- Timing per frame: idle 400-800 ms, walk 80-120 ms, run 50-80 ms; attack phases anticipation 120-200 ms, action 30-60 ms, follow-through 80-150 ms. Its thesis: "frame timing beats frame count". Holds are repeated or longer frames on wind-up so the strike reads faster. Same source.
- The same blog claims Celeste's run is 4 frames and Shovel Knight's walk is 6; I could not confirm either in a developer source, so UNVERIFIED. Same source.
- Another vendor guide (Sorceress, 2026-05-10): walk 4 or 8 frames, idle 2-4, attack 4-6, walk about 6-8 fps, idle 2-3 fps, 6-frame attack about 12 fps; frame sizes 32/48/64/96 chosen by camera distance. https://sorceress.games/blog/ai-sprite-generator-pixel-walk-cycles-idles-and-vfx
- Smear frames, sub-pixel animation, 8-direction construction (e.g. draw 5 directions and mirror), and how studios keep characters consistent across hundreds of frames: NO primary source found. The mirror-five-directions method is general knowledge only, UNVERIFIED. Saint11 (Pedro Medeiros) tutorials exist on motion blur, squash and stretch, easing, but their contents were not read. https://creativebloq.com/news/artist-makes-animated-gif-tutorials-to-teach-you-pixel-art

## 2. How AI game devs build animation pipelines

Reddit could not be read directly (blocked). What follows comes from tool docs, blogs, itch.io pages and one search-excerpted r/aigamedev thread. Dates are given where the page states them.

Image-gen then video model then frame extraction (the most commonly reported path)
- r/aigamedev, 2025 (Veo 3 sprite tests; excerpt only): sprite animated from one image with Veo 3, frames imported into Photoshop to remove the background; poster "would still hire a pixel artist to clean everything up" but finds it useful for prototyping; a commenter suggested first/last-frame interpolation for more control. https://redlib.ssps.io/r/aigamedev/comments/1la885i/more_veo_3_sprite_animation_tests (mirror not fetchable; content from search snippet)
- MP4toSprite (itch.io dev, versions dated Apr 21 to May 31 2026): found direct AI sprite sheets "often unusable... inconsistent motion, unstable style"; so "generate short videos with AI first, then extract the frames I need". Pain: per-frame background removal and semi-transparent effects; exports Godot 4 SpriteFrames .tres. https://yewlark.itch.io/mp4tosprite
- Scenario guide (2026-04-27, vendor): clean side-view base image, convert to video with image as first frame, extract key poses (contact, lift-off, mid-stride), assemble grid-aligned sheet, test loop. Pain: "Generate each animation state independently and your character's proportions will shift"; recommends training a custom character model on 5-15 references. Frame counts: 8-16 per core state. https://www.scenario.com/blog/ai-sprite-generator
- Sarthak Mishra (2026-01-07): ChatGPT base image, Photoshop cleanup, Midjourney animation, Python for frames. Reported: AI tends to "hallucinate a grid that doesn't align"; "the biggest issue with AI video is flicker. Pixels that should be static tend to dance around"; fixes were a static-region mask assigning each pixel its mode colour across frames, and one shared 24-colour palette, no dithering, across all frames. Output a 5x5 sheet of 62 px frames at 10 fps. https://sarthakmishra.com/blog/building-animated-sprite-hero
- Wan 2.2 specifically: a side-view "pixel walk" LoRA for Wan 2.2 I2V-A14B (weight 0.7-0.9, 24 frames, 16 fps, use both high- and low-noise files; author "used down sampling and scaling to get better pixel style"). https://comfy.icu/models/2172038/wan2-2-lora-walk-animation-side-view-sprite-animation-pixel-style/2445934/v1-0 . Another Wan 2.2 I2V pixel-sprite LoRA: trained on 226 sprite clips at 600x370, frame buckets 8/16/24/32, recommended 45 frames (beyond its training buckets, my inference); the card gives no post-processing advice. https://huggingface.co/styly-agents/Wan2-2-pixel-animate
- pixie.haus (undated vendor page): standard temporal models cause "morphing" where pixels shift and colours blend; its image-to-video uses the sprite as the exact first frame, fixed camera, transparent background input; looping is only prompt-driven ("seamless loop"), advice that first and last frames should match. https://pixie.haus/resources/creating-movement-engineering-pixel-art-animation-ai

Sheet-in-one-image generators
- Retro Diffusion `rd-animation` (Replicate): styles with "dedicated look and format" for position and action consistency; a four-direction, four-frame humanoid walk (48 px width only); input palette for strict colour adherence; image references are not used as init images; sizes restricted per style. https://replicate.com/retro-diffusion/rd-animation/readme (the four_angle_walking detail came from a search excerpt of https://replicate.com/retro-diffusion/rd-animation)
- Sorceress "Quick Sprites": the whole animation returned in one image; Auto-Sprite v2: upload, extract frames, remove background, align, export with JSON grid sidecar. Reported pain: wobble if silhouette alignment missed, palette drift, edge halos, a weak "up/back" row at 48 px, motion blur in video sources ("use 3D-rendered source" because import cannot fix it), prompt overload. Fixes: one fixed game-wide palette, 1-px erode, re-roll single direction rows. https://sorceress.games/blog/ai-sprite-generator-pixel-walk-cycles-idles-and-vfx

Pose control / skeleton
- PixelLab "animate with skeleton": canvases 256/128/64/32/16; set a reference frame (which estimates a skeleton you can hand-edit); modes "Freeze 1, Generate 2", "Freeze 2, Generate 1" and custom inpainting of chosen frames; "Animation to animation"; consistency tips: fixed head "always", reuse seed, raise init-image strength gradually, reuse saved skeletons across characters. This is the closest AI analogue to keying extremes by hand. https://www.pixellab.ai/docs/tools/animate-with-skeleton

3D-rig to sprite and Mixamo img2img
- No AI-dev report of either was fetched. The only related note: Sorceress suggests a 3D-rendered source to avoid motion blur (above). Not claiming more.

Grid snapping and downscaling (the problem we hit)
- Tools exist that detect true pixel size then snap: unfake (runs-based GCD and Sobel-edge scale detection, quantize, dominant-colour-per-block downscale), Pixel Snapper (Sprite Fusion), PixelFixer, DeAI PixelKit, spritecook's Pixel Grid Detector (advice: generate at the highest resolution available so the detector has more data). A DEV article notes that even after finding the grid, pixels often "bleed over, blur at the edges, or just float slightly off". These are from a search summary; the individual repos/pages were not read. https://pypi.org/project/unfake/ , https://dev.to/jenissimo/how-to-tame-your-ai-pixel-art-3pk5
- Astropulse's pixelDetector (the repair tool built into Aseprite under Retro Diffusion) targets JPEG damage in pixel art and caps/predicts a palette; it does not claim grid detection. https://github.com/Astropulse/pixelDetector
- Multi-frame batch tools with consistent settings across frames exist (AnimSprite Pixelizer; True Pixel shared-palette mode extracts one palette across all frames), via search summary only. https://uhiyama-lab.itch.io/animsprite-pixelizer

Hand cleanup is universal in reports: Veo poster would hire a pixel artist; MP4toSprite offers per-frame touch-up; Mishra cleans the base image in Photoshop; Sorceress lists erode and re-roll steps.

## 3. How AI pipelines differ from traditional ones

Stage by stage (synthesis; each row is my reading of the cited sources, not a claim from one source)

| Stage | Traditional (Dead Cells, hand pixel, Spine) | AI video pipeline | Failure mode introduced |
|---|---|---|---|
| Concept | Pixel model sheet, then 3D or sprite built from it (Vasseur) | One generated image, often the only identity anchor | Identity drift between states (Scenario) |
| Keys | Animator poses extremes; "the least amount of frames possible" (Vasseur); skeleton poses (Spine) | Model invents the motion; at best first/last-frame control or skeleton control (PixelLab) | No keyframe control; non-looping unless prompted (pixie.haus) |
| In-betweens | Interpolated from keys, by rig or by hand | Video model produces dense frames; we then pick a few | Picked frames are not designed extremes and holds |
| Cleanup | By hand, or accepted as is (Vasseur skips flicker) | Background removal, jitter masking, grid snapping, palette quantise (Mishra, Sorceress) | Grid/palette noise: "hallucinate a grid", flicker |
| Colour | Fixed palette from the start (Owlboy limited palette) | Palette applied after generation, per frame unless shared | Palette drift (Sorceress, Mishra) |
| Export | Per-frame durations, tags, sheet | Uniform fps from video, grid sheet + JSON/.tres | Timing is uniform; holds are lost |

What AI pipelines tend to skip that matters for game-ready sprites (derived from the sources above)
- Native-resolution drawing: hand pixel art and Dead Cells render at target size with no anti-aliasing; AI frames are drawn large then reduced, which creates the grid problem (Vasseur; Mishra; spritecook advice).
- Designed timing: Sprite AI's timing table (anticipation hold, short strike, recovery) has no counterpart in uniform-fps video extraction (vendor source).
- Loop closure and root-motion stability: only prompt-based in pixie.haus; Sorceress de-duplicates leading frames and aligns centre of gravity.
- One palette and one grid per animation or character, not per frame (Mishra, Sorceress, True Pixel).

## Implications for this pipeline
Prioritised; each cites the motivating source.
1. Measure the pixel grid and palette once per animation (or per character), then apply to all frames. Reports of jitter and per-frame palette drift: Mishra (shared 24-colour palette across all frames); True Pixel shared-palette mode; the unfake/Pixel Snapper family. https://sarthakmishra.com/blog/building-animated-sprite-hero
2. Add a static-region/jitter stabiliser after downscale (mode colour across frames for pixels that barely change). Mishra's fix for "pixels dance around". Same source.
3. Enforce loop closure: after frame picking, require first and last frame near-identical (score and reject, or blend), and drop duplicate leading frames. pixie.haus (loop only prompt-driven), Sorceress (dedupe). https://pixie.haus/resources/creating-movement-engineering-pixel-art-animation-ai , https://sorceress.games/blog/ai-sprite-generator-pixel-walk-cycles-idles-and-vfx
4. Pick frames as designed extremes with explicit per-frame durations (not uniform fps): contact, passing, and holds on wind-up; export durations to Godot SpriteFrames. Sprite AI timing (vendor), Scenario key-pose list, Vasseur keyframe-first approach. https://www.sprite-ai.art/blog/sprite-animation-frames , https://www.scenario.com/blog/ai-sprite-generator
5. Add keyed control where the video model allows: first-and-last-frame or pose-conditioned generation, and skeleton/pose reuse across characters. PixelLab's Freeze/Generate modes and saved skeletons; r/aigamedev commenter on first/last-frame. https://www.pixellab.ai/docs/tools/animate-with-skeleton
6. Reduce video artifacts upstream: transparent or flat background, static camera, no motion blur, highest source resolution. pixie.haus, Sorceress (motion blur not fixable at import), spritecook advice. Same sources as above.
7. Test a native-resolution path for small sprites (around 50 px tall like Dead Cells, or 32-64 px): generate large, then run only the once-per-animation snap, or render a rig at target size with no anti-aliasing for hard-to-control actions. Vasseur; Sorceress suggestion of 3D-rendered source. https://www.gamedeveloper.com/production/art-design-deep-dive-using-a-3d-pipeline-for-2d-animation-in-i-dead-cells-i-
8. Keep identity anchored per character (a character LoRA or reference-locked generation, and a fixed palette) rather than generating each state independently. Scenario ("proportions will shift"), Retro Diffusion palette input. https://replicate.com/retro-diffusion/rd-animation/readme
9. Be careful with frame counts: the Wan pixel LoRA card recommends 45 frames while training on up to 32 (my inference), so test the short-clip lengths our LoRA was trained on. https://huggingface.co/styly-agents/Wan2-2-pixel-animate

## Sources
- https://www.gamedeveloper.com/production/art-design-deep-dive-using-a-3d-pipeline-for-2d-animation-in-i-dead-cells-i-
- https://us.esotericsoftware.com/spine-in-depth
- https://de.esotericsoftware.com/forum/d/6865-game-engines-that-work-with-skel
- https://steamcommunity.com/app/262060/discussions/4/1727575977545379683
- https://giantbomb.com/wiki/Concepts/Ubiart_Framework
- https://en.wikipedia.org/wiki/Donkey_Kong_Country
- https://www.killscreen.com/owlboy/
- https://www.acmi.net.au/works/125388--hollow-knight-silksong-game-worlds-acmi/
- https://avclub.com/hyper-light-drifter-blended-past-and-present-to-become-1798255620
- https://creativebloq.com/news/artist-makes-animated-gif-tutorials-to-teach-you-pixel-art
- https://www.sprite-ai.art/blog/sprite-animation-frames
- https://sorceress.games/blog/ai-sprite-generator-pixel-walk-cycles-idles-and-vfx
- https://www.scenario.com/blog/ai-sprite-generator
- https://yewlark.itch.io/mp4tosprite
- https://sarthakmishra.com/blog/building-animated-sprite-hero
- https://pixie.haus/resources/creating-movement-engineering-pixel-art-animation-ai
- https://www.pixellab.ai/docs/tools/animate-with-skeleton
- https://replicate.com/retro-diffusion/rd-animation/readme
- https://replicate.com/retro-diffusion/rd-animation
- https://comfy.icu/models/2172038/wan2-2-lora-walk-animation-side-view-sprite-animation-pixel-style/2445934/v1-0
- https://huggingface.co/styly-agents/Wan2-2-pixel-animate
- https://pypi.org/project/unfake/
- https://dev.to/jenissimo/how-to-tame-your-ai-pixel-art-3pk5
- https://github.com/Astropulse/pixelDetector
- https://uhiyama-lab.itch.io/animsprite-pixelizer
- https://redlib.ssps.io/r/aigamedev/comments/1la885i/more_veo_3_sprite_animation_tests (search snippet only)

## Blocked
- old.reddit.com, www.reddit.com (.json and search) · bot block · run from a logged-in browser or paste thread text / export via Reddit API key
- redlib.ssps.io (Reddit mirror) · DNS failure · paste the r/aigamedev thread text or name a working mirror
- Celeste, Hyper Light Drifter, Owlboy, Hollow Knight developer-side animation pipelines · no primary source found · provide GDC/interview links or the Celeste art book
- Diablo II sprite pipeline, rotoscoping and Mixamo-img2img AI reports · no source found · supply links
- Smear frames, sub-pixel animation, 8-direction construction, studio character-consistency practice · no primary source found · provide Saint11/Slynyrd/MiniBoss tutorial pages (Patreon is login-gated)
- Sprite AI blog, Scenario, Sorceress, pixie.haus · vendor sources, not independent · independent studio data would replace them
- Layer.ai, Ludo.ai, God Mode AI, SpriteCook docs · not fetched · time/scope; can be added on request
