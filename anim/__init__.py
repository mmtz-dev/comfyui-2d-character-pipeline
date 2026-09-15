"""CPU stages of the pose-to-pixel animation pipeline feature.

Ten stages take a filmed clip to playable 64x64 frames in game-02:

    extract -> pose -> pick -> review -> strip -> render -> post -> publish
    base (the standing-sprite variant of the same chain)
    run  (the whole chain in one command)

`render` is the only one that talks to a GPU: it POSTs the workflows
in gpu/workflows/ to the GPU image service feature over HTTP. Everything else is
deterministic CPU work, which is the point - the parts that must be reproducible
are not inside a sampler.
"""

__version__ = "0.1.0"
