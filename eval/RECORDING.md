# Recording the live test

Three short live captures, one per task in `eval/tasks/`. About 15 minutes in
all. Recordings stay in `eval/recordings/` on this disk and are never
committed (see `.gitignore`).

## 1. Once: permissions

Pick the terminal app you'll record from (Terminal or iTerm) and give it all
three in System Settings → Privacy & Security:

- **Screen Recording**
- **Accessibility**
- **Input Monitoring**

Quit and reopen that terminal app, then check:

```sh
cd path/to/prentice
uv run python scripts/check_permissions.py
```

## The easy way: let the computer do the tasks

After the permissions above, one command records all three tasks while the
computer does them with synthetic mouse and keyboard input, and writes an
exact answer key into each session folder (`answer_key.json`, status
`"scripted"`), so nothing needs labelling:

```sh
uv run prentice-scripted-run --dry-run   # checks everything, touches nothing
uv run prentice-scripted-run             # about 3 minutes; press Esc to abort
```

Don't touch the mouse or keyboard while it runs. Close or hide private
windows first: the whole screen is recorded. The Finder task works only in
`~/prentice-sandbox`, which the script creates and removes again. A single
task: `--task terminal-git` (or `browser-wikipedia`, `finder-folder`); Safari
instead of Chrome: `--browser Safari`.

A script acts more cleanly than a person, so its numbers are optimistic; the
manual recording below measures the human case.

## The manual way

## 2. Before each recording

- **Turn on Do Not Disturb**, so no notification ends up in the video.
- **Close or hide anything private:** mail, chats, documents, other browser tabs.
- **Never type a password or secret.** Every keystroke is logged as text.
- Keep the recorder's terminal window small and out of the way.
- Act at a calm pace: about a second between actions helps the segmenter.

## 3. Record

For each task, in this order:

| Order | Task | Read first | Before you start |
|---|---|---|---|
| 1 | Terminal | `eval/tasks/terminal-git/task.md` | Open a **second** Terminal window for the task |
| 2 | Browser | `eval/tasks/browser-wikipedia/task.md` | Open a new, empty browser tab |
| 3 | Finder | `eval/tasks/finder-folder/task.md` | Run `echo test > ~/Desktop/prentice-dummy.txt`, open a Finder window on the Desktop |

Then:

```sh
uv run prentice-capture start
```

Do the task, switch back to the recorder window and press **Ctrl+C**. The
switch back is expected; it shows up as one extra step.

Recording the three tasks in this order is enough for the sessions to be told
apart (each session folder is named by its start time). A second take of a
task is welcome but optional.

**After the Finder task:** delete the `prentice-test` folder from the Desktop.

## 4. Afterwards

Say which sessions are which (or just that you recorded them in this order).
The rest is processing:

```sh
uv run prentice-segment   eval/recordings/<session>
uv run prentice-interpret eval/recordings/<session>   # local model, a few minutes
uv run prentice-refine    eval/recordings/<session>
uv run prentice-ground    eval/recordings/<session>
uv run prentice-score     eval/recordings/<session> eval/ground_truth/<task>/steps.json
```

Before the numbers count, a human watches each recording once and corrects
`eval/ground_truth/<task>/steps.json` to what actually happened, then sets
`"status": "verified"` and `"verified_against": "<session>"`.
