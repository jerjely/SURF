# PyInstaller recipe for a standalone SURF folder (no Python needed on the target PC).
# Build with install/build_standalone_windows.bat (Windows) or:
#   pyinstaller --noconfirm install/surf.spec
from PyInstaller.utils.hooks import collect_submodules

HERE = SPECPATH + "/.."

a = Analysis(
    [HERE + "/wave_gui.py"],
    pathex=[HERE],
    datas=[(HERE + "/defaults/surf_defaults.json", "defaults"), (HERE + "/surf.ico", "."), (HERE + "/surf.png", "."), (HERE + "/README.md", "."),
           (HERE + "/logo", "logo")],
    hiddenimports=collect_submodules(
        "pyqtgraph", filter=lambda n: not any(k in n for k in (".examples", ".opengl", ".jupyter")))
    + ["roifile", "openpyxl", "threadpoolctl", "sklearn.cluster", "scipy.spatial",
       "statsmodels.formula.api", "statsmodels.stats.anova", "patsy", "compare_tab", "wave_stats", "surf_calibrate", "presets", "info_tab", "player", "mpl_nav", "figexport", "cells_core", "cells_tab",
       # figure exporters are imported lazily by savefig()
       "matplotlib.backends.backend_agg", "matplotlib.backends.backend_ps",
       "matplotlib.backends.backend_svg", "matplotlib.backends.backend_pdf"],
    excludes=["tkinter", "PyQt5", "PyQt6", "PySide2", "IPython", "jupyter", "notebook",
              "pyqtgraph.examples", "pyqtgraph.opengl", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="SURF",
          console=False, icon=HERE + "/surf.ico", upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="SURF", upx=False)
