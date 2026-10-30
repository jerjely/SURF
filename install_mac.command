#!/bin/bash
# SURF (Spontaneous Undulation in Retina Finder) - macOS installer. Double-click (or: bash install_mac.command).
# Installs into ~/Library/Application Support/WaveAnalysis and creates ~/Applications/SURF.app
set -e
SRC="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HOME/Library/Application Support/WaveAnalysis"
APP="$ROOT/app"; VENV="$ROOT/venv"
BUNDLE="$HOME/Applications/SURF.app"
rm -rf "$HOME/Applications/Wave Analysis.app"      # old name
echo "=== SURF - Spontaneous Undulation in Retina Finder - installer ==="

PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3 "$HOME/anaconda3/bin/python" "$HOME/miniconda3/bin/python" /opt/homebrew/bin/python3 /usr/local/bin/python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "No Python 3.9+ found. Install it from https://www.python.org/downloads/ and run this again."
  read -n 1 -s -r -p "Press any key to close"; exit 1
fi
echo "Using $("$PY" --version) ($PY)"

mkdir -p "$APP"
cp "$SRC"/wave_gui.py "$SRC"/wave_core.py "$SRC"/wave_plots.py "$SRC"/wave_stats.py "$SRC"/compare_tab.py "$SRC"/surf_calibrate.py "$SRC"/presets.py "$SRC"/info_tab.py "$SRC"/player.py "$SRC"/mpl_nav.py "$SRC"/figexport.py "$SRC"/cells_core.py "$SRC"/cells_tab.py "$SRC"/surf.ico "$SRC"/surf.png "$SRC"/requirements.txt "$SRC"/README.md "$APP/"
cp -R "$SRC"/logo "$APP/"
cp -R "$SRC"/defaults "$APP/"
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
echo "Installing packages (first time only: a few minutes)..."
"$VENV/bin/python" -m pip install --disable-pip-version-check --upgrade pip >/dev/null
"$VENV/bin/python" -m pip install --disable-pip-version-check -r "$APP/requirements.txt"
"$VENV/bin/python" -c "import PySide6, pyqtgraph, cv2, sklearn, scipy, roifile, openpyxl, statsmodels; print('All packages OK')"

# minimal .app bundle that runs the GUI
mkdir -p "$BUNDLE/Contents/MacOS" "$BUNDLE/Contents/Resources"
cat > "$BUNDLE/Contents/MacOS/SURF" <<EOF
#!/bin/bash
exec "$VENV/bin/python" "$APP/wave_gui.py" "\$@" 2>>"\$HOME/surf_error.log"
EOF
chmod +x "$BUNDLE/Contents/MacOS/SURF"
cat > "$BUNDLE/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>SURF</string>
  <key>CFBundleExecutable</key><string>SURF</string>
  <key>CFBundleIdentifier</key><string>hu.pte.surf</string>
  <key>CFBundleIconFile</key><string>surf</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
EOF
# icon (optional; uses macOS built-in tools)
if command -v iconutil >/dev/null 2>&1; then
  T="$(mktemp -d)/surf.iconset"; mkdir -p "$T"
  for s in 16 32 64 128 256; do sips -z $s $s "$APP/surf.png" --out "$T/icon_${s}x${s}.png" >/dev/null 2>&1 || true; done
  iconutil -c icns "$T" -o "$BUNDLE/Contents/Resources/surf.icns" 2>/dev/null || true
fi
touch "$BUNDLE"
echo ""
echo "Done. 'SURF' is in ~/Applications (also findable with Spotlight)."
echo "To update: unzip the new version and run this again. To remove: delete the app and"
echo "  ~/Library/Application Support/WaveAnalysis"
read -n 1 -s -r -p "Press any key to close"
