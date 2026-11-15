# Task: finder-folder

**App:** Finder.

**Start state:** a file `prentice-dummy.txt` on the Desktop (create it with
`echo test > ~/Desktop/prentice-dummy.txt`), no folder named `prentice-test`
there, and a Finder window showing the Desktop.

**Steps, in order:**
1. Create a new folder on the Desktop (File → New Folder, or Shift-Command-N).
2. Name it `prentice-test` and press Enter.
3. Drag `prentice-dummy.txt` into `prentice-test`.
4. Open `prentice-test`.
5. Rename the file to `prentice-renamed.txt` (select it, press Enter, type the name, press Enter).

**Success:** `~/Desktop/prentice-test/prentice-renamed.txt` exists.

**Clean-up afterwards:** delete the `prentice-test` folder.

**What it exercises:** clicks on named items, which only a live capture can tie to accessibility identifiers, plus a drag and typed names.
