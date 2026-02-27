---
title: "Barnes-Hut C Runtime Parallelization Request"
summary: "Implementation request to parallelize clustering, collisions, and inverse-square forces in C using Barnes-Hut."
category: "implementation"
created_at: "2026-02-22T12:24:49"
original_filename: "2026.02.22.12.24.49.md"
original_relpath: "docs/notes/implementation/2026.02.22.12.24.49.md"
tags:
  - implementation
  - barnes-hut
  - c-runtime
---

using the code in @hacks/ as a reference,
I want you to paralellize our clustering, collision, and inverse square force application through Barnes Hut
Please do this in the C side of the code, so shared read only memory locking can take advantage completely of the double buffer architecture.

@hacks/client/colliding-particles - This is a multi threaded implementation of a barne's hut optimized N body 2d gravity sim.
I used a BSP here instead of a quad tree, because it was easier for me to figure out how to represent a BSP in the double buffers and safely paralellize
the data structure.
But I think that a quad tree would be more optimial. I'll leave that up to you if you can figure out how to do a quad tree in a stable double buffer.

@hacks/client/obstacles - This is a vector field based ACO algorithim that was the inspiration for the Nooi fields, and the daimoi particles.

Both have slightly different ways of approaching entities.

@hacks/shared is a large collection of libraries I made to facilitate these simulations.
