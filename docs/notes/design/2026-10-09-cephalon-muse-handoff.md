# Cephalon, Muse, and Fork Tales: design and runtime handoff

October 9, 2026. Design exploration; runtime not restored or tested in this session.

## Intent

The user wants a persistent agent team integrated into existing tools and working practices. Starting a conversation should mean talking to an agent with relevant context, without selecting a thread or reconstructing ongoing work. Threads remain underlying records that the system finds and connects, analogous to the user's Emacs shortcut that creates a timestamped note without requiring a title.

The user sets direction across teams. Agents coordinate directly, review each other's work, share responsibility, and bring decisions requiring human judgment back to the user. The aim is to make existing systems legible to agents and integrate agents into familiar methodologies, rather than discard established software-development practices.

## Cephalon: the head

A fast conversational and perceptual interface with memory and a working world model. Its key skill is conversational judgment: following unfinished thoughts, recognizing changed intent, understanding what is needed, and preserving that meaning when coordinating work. It needs enough technical understanding to route tasks and question results; specialists supply deeper expertise.

Audio perception beyond transcription motivates the user's exploration of Gemma-family models and audio embeddings. Specific model names and capabilities were not independently verified. Audio cues can inform clarification but do not establish certainty about a user's emotions.

Graph memory connects observations, decisions, projects, and active agent work. The working picture must distinguish observations, assumptions, older information, disagreements, and unknowns. Responsiveness matters while deeper work occurs asynchronously.

Long-term interfaces include an animated face, a character interacting in a 3D virtual world, or a robot with sensors and actions. These are possible embodiments of the same core.

## Muse: the whole system

The Muse includes the cephalon, memory, ongoing commitments, and execution processes. Internal workers are metaphorically its hands: concurrent activity belonging to one coordinated system, rather than unrelated agents whose context the user must manage.

Separate Muses can cooperate as peers. Central conversational access does not require every peer-to-peer exchange to pass through the cephalon. Ending a call should not end authorized ongoing work; returning should reconnect the user with accurate results. This requires persistent runtime support, not merely a chat promise.

The head/body/person-like language describes an architectural and interaction metaphor, not consciousness or human identity. Clear AI identity, attribution, responsibility, and human agency matter.

## Cross-agent communication

Nox is envisioned as both an agent server and a neutral communication/event layer. A motivating problem is Edamu projects moving to separate repositories/submodules while agents continue changing the parent repository.

A useful message carries what is changing, why, where work belongs, affected tasks, and overlap or dependencies. Acknowledgment and conflict reporting close the loop. The user wants to notice something in one conversation and have that context reach another agent without manually navigating its thread.

Agents sharing the user's GitHub identity make attribution confusing. Distinct agent identities are desired; explicit sender/recipient metadata could support interim attribution. No identity configuration was performed.

## Why revisit Fork Tales

Fork Tales is a prototype, proof of concept, and artwork with narrative compression, graph memory, presences, Muses, visualization, music, and story elements. The user believes valuable ideas remain to be recovered and wants to preserve the artistic meaning while making the architecture understandable.

The user liked the visualization but was unsure what its behavior represented and recalls model drift. Drift is a reported observation, not a diagnosed failure. The user wants meaningful output, such as music, images, or another product made with the system; infrastructure that only produces more agent tools is not the entire objective.

The latest direction is to get Fork Tales running again, rediscover its visualization and existing mechanisms, and then choose a useful demonstration. A broad refactor or redesign has not been agreed.

## Evidence read from the repository

- README: Python runtime and APIs, React/TypeScript frontend, agent specifications, and narrative audio submodule.
- AGENTS.md: repository guidance prefers Docker runtime startup, focused changes, preservation of existing state, and relevant verification when behavior changes.
- world_building/notes/Notes_Loops.md: perception, representation, narrative meaning compression, projection, action, and feedback form a loop closely related to the cephalon concept.
- docs/notes/implementation/2026-02-27-daimoi-crawler-muse-architecture-note.md: describes a canonical graph, fact snapshots, named queries, grounded Muse responses, and provenance receipts.
- docs/notes/implementation/2026-02-26-implementation-brief-muse-facts-graph.md: describes asynchronous Muse chat grounded in simulation facts and graph queries with compact model context.
- part64/docker-compose.yml: existing startup configuration includes model-service dependencies and hardware-specific GPU/NPU settings.

These are documented designs and historical implementation descriptions. Current runtime behavior, available hardware, and startup feasibility were not tested.

## Repositories and execution context

Primary: octave-commons/fork_tales.
Supporting development repository: riatzukiza/devel.
Previous running machine: the user's Stealth notebook.

The user reports this session started without selecting an environment. They want this document to carry the discussion into an environment-enabled conversation and authorized cloning the repositories and creating cloud workspaces.

This session had GitHub access but no ordinary Bash terminal, direct SSH, or Codex workspace/thread-management controls. Attempts to create a Knox temporary sandbox returned authentication prompts and did not establish a sandbox. Neither repository was cloned, no cloud workspace was created, Fork Tales was not launched, and Stealth was not accessed. A subagent here would share the same tools and access limitations.

## Next work in an environment

Inspect the current repositories and their local guidance. If remote access is configured, inspect Stealth for an existing checkout and saved state. Check actual model-service and hardware requirements. Restore the existing runtime and visualization, verify it, and explain what its elements represent using runtime evidence.

Trace one Muse conversation through retrieval, execution, and feedback. Identify which parts of the head-and-hands model are documented, implemented, and actually verified.

Choose a small demonstration with the user after rediscovery: either coordination around repository/submodule boundaries, or a creative artifact whose decisions, progress, and provenance remain available across conversations. The latest emphasis favors Fork Tales and meaningful creative output. Preserve the narrative and artwork rather than treating them as irrelevant decoration.

## Success criteria

The user can start a conversation without locating a thread, relevant context is retrieved with clear uncertainty, authorized work continues in a persistent runtime, affected agents coordinate visibly, and the user returns to an accurate account of progress and decisions requiring their judgment.
