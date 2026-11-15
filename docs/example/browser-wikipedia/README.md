# Example output: the browser task

What prentice produced from a live recording of
[`eval/tasks/browser-wikipedia`](../../../eval/tasks/browser-wikipedia/task.md):
open en.wikipedia.org, search "Large language model", scroll down. Recorded
by `prentice-scripted-run` in a Chrome guest window. Unedited.

- [`SKILL.md`](SKILL.md): the skill Claude Code would follow.
- [`../../demo/wikipedia-input-output.gif`](../../demo/wikipedia-input-output.gif):
  the recording next to these steps, each appearing when its action starts.
  Built with `scripts/build_demo_gif.py`.

How it scored against the task's answer key (`prentice-score`):

| | |
|---|---|
| Real steps found | 3 of 3 (steps 2, 6 and 8) |
| Steps that match a real one | 3 of 8 |
| Right action type | 3 of 3 |
| Values captured | 1 of 2: the search text; the address is only in step 2's description, not stored as a value |

The other five steps are noise: window-title changes recorded as "switched
to" steps (1, 3, 4, 7) and the Ctrl+Alt+F shortcut that focuses Wikipedia's
search field (5), which is also mislabelled as a terminal command. Nothing is
scripted, because no terminal command was run; every step is a UI instruction
Claude Code would carry out in the browser.
