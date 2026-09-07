#!/bin/bash
# Double-click this file to launch the spatial-evolution explorer widget.
# (macOS opens it in Terminal. It needs Python 3 + marimo, already installed on
#  this machine. To edit the widget instead of just running it, change "run" to
#  "edit" on the last line.)
cd "$(dirname "$0")"
echo "Starting the spatial-evolution explorer..."
echo "A browser tab will open shortly. To stop it, close this window or press Ctrl-C."
echo
python3 -m marimo run app.py
