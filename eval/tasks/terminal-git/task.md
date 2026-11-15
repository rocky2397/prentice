# Task: terminal-git

**App:** Terminal, in a **second** window (the first one is running the recorder).

**Start state:** a fresh prompt in your home folder in that second window.

**Steps, in order:**
1. `cd` into your prentice checkout (e.g. `cd ~/code/prentice`) and Enter.
2. `ls` and Enter.
3. `git log --oneline -3` and Enter.

**Success:** the last three commit subjects are on screen.

**What it exercises:** typed shell commands, which should come out of the pipeline as runnable script steps rather than UI replay.
