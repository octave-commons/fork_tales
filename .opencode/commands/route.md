---
description: Route a prompt via PromptDB and print recommended skills/tools
agent: plan
---
Call tool router.decide with:
- prompt: $ARGUMENTS
- mode: \"recommend\"
Then print the returned witness report and suggested skill loads.
