import sys
import os
import yaml
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtGui, QtCore

import matplotlib
matplotlib.use('Qt5Agg')

# Import Components
try:
    from src.gui.components.dock_left import ContextSettingsDock # Might reuse or hide
    # NO dock_right, NO dock_bottom, NO tab_panel
    from src.gui.components.toolbar import MainToolbar
    from src.gui.components.canvas import UnifiedCanvas
    
    # NEW COMPONENTS
    from src.gui.components.structure_panel import StructurePanel
    from src.gui.components.handbook_panel import HandbookPanel
    from src.gui.components.results_panel import ResultsPanel
    
except ImportError as e:
    print(f"Error importing GUI components: {e}")
    sys.exit(1)

from src.logic.smart_engine import SmartAnalysisEngine
try:
    from predict_combined import SpectrumPredictor
except ImportError:
    class SpectrumPredictor:
         def __init__(self): print("Mock Predictor")
         def predict(self, x, y, type): return {}

class CanvasControls(QtWidgets.QWidget):
    """
    Panel below the plot to toggle overlays and highlights.
    """
    toggleOverlays = QtCore.pyqtSignal(bool)
    toggleHighlights = QtCore.pyqtSignal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(10, 5, 10, 5)
        
        self.cb_overlays = QtWidgets.QCheckBox("Show Top 5 CAS Matches (Overlays)")
        self.cb_overlays.setChecked(True)
        self.cb_overlays.toggled.connect(self.toggleOverlays.emit)
        
        self.cb_highlights = QtWidgets.QCheckBox("Show Functional Group Highlights")
        self.cb_highlights.setChecked(True)
        self.cb_highlights.toggled.connect(self.toggleHighlights.emit)
        
        self.layout.addWidget(self.cb_overlays)
        self.layout.addWidget(self.cb_highlights)
        self.layout.addStretch()

class AnalyzeItWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analyzelt - Spectrum Analysis")
        self.setMinimumSize(1400, 900)
        
        # Data Models (Buffer)
        self.raw_x = np.array([])
        self.raw_y = np.array([])
        self.processed_x = np.array([])
        self.processed_y = np.array([])
        
        self.current_highlights = [] 
        self.current_overlays = []
        self.spec_type = 'ir' # Default
        
        self.show_highlights = True
        self.show_overlays = True
        self.show_grid = True
        
        self.init_predictor()
        
        # --- 1. Toolbar ---
        self.toolbar = MainToolbar(self)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolbar)
        self.toolbar.loadActionTriggered.connect(self.load_file)
        self.toolbar.homeClicked.connect(self.nav_home)
        self.toolbar.resetViewClicked.connect(self.reset_view)
        # self.toolbar.highlightToggled.connect... (Moved to CanvasControls)

        # --- 2. Main Layout (SPLITTER) ---
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.setCentralWidget(self.main_splitter)
        
        # --- TOP PANE (Horizontal Splitter) ---
        self.top_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.main_splitter.addWidget(self.top_splitter)
        
        # A. Structure Panel (Left)
        self.structure_panel = StructurePanel()
        self.top_splitter.addWidget(self.structure_panel)
        
        # B. Canvas Container (Center) - Canvas + Controls
        self.center_widget = QtWidgets.QWidget()
        self.center_layout = QtWidgets.QVBoxLayout(self.center_widget)
        self.center_layout.setContentsMargins(0, 0, 0, 0)
        self.center_layout.setSpacing(0)
        
        self.canvas = UnifiedCanvas(self)
        self.center_layout.addWidget(self.canvas)
        
        self.canvas_controls = CanvasControls()
        self.canvas_controls.toggleOverlays.connect(self.set_show_overlays)
        self.canvas_controls.toggleHighlights.connect(self.set_show_highlights)
        self.center_layout.addWidget(self.canvas_controls)
        
        self.top_splitter.addWidget(self.center_widget)
        
        # C. Handbook Panel (Right)
        self.handbook_panel = HandbookPanel()
        self.top_splitter.addWidget(self.handbook_panel)
        
        # Set Factors: 20% Left, 50% Center, 30% Right
        self.top_splitter.setStretchFactor(0, 1)
        self.top_splitter.setStretchFactor(1, 3)
        self.top_splitter.setStretchFactor(2, 2)
        
        # --- BOTTOM PANE (Results Panel) ---
        self.results_panel = ResultsPanel()
        self.main_splitter.addWidget(self.results_panel)
        self.results_panel.groupSelected.connect(self.on_group_selected)
        
        self.main_splitter.setStretchFactor(0, 7)
        self.main_splitter.setStretchFactor(1, 3)

        # Initial Status
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready. Load an IR file to begin.")
        
        # Context Dock (Minimised)
        self.left_dock = ContextSettingsDock(self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.left_dock)
        self.left_dock.hide()

    def init_predictor(self):
        try:
            self.predictor = SpectrumPredictor()
            print("SpectrumPredictor initialized.")
        except Exception as e:
            print(f"Error initializing SpectrumPredictor: {e}")

    def set_show_overlays(self, checked):
        self.show_overlays = checked
        self.update_plot()

    def set_show_highlights(self, checked):
        self.show_highlights = checked
        self.update_plot()

    def load_file(self, target_type):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, f"Open {target_type.upper()} File", "", "Data Files (*.jdx *.dx *.csv *.txt)")
        if not fname: return
        
        try:
            self.spec_type = target_type
            x, y = np.array([]), np.array([])
            
            # Simple Parser Logic (Should ideally use data_processing.parse_jdx)
            if fname.lower().endswith('.csv'):
                df = pd.read_csv(fname, comment='#', header=None)
                if df.shape[1] >= 2:
                    x = df.iloc[:, 0].values.astype(float)
                    y = df.iloc[:, 1].values.astype(float)
            else:
                 with open(fname, 'r') as f:
                    lines = f.readlines()
                    x_list, y_list = [], []
                    for line in lines:
                        if not line.startswith("##") and not line.startswith("$"):
                             try:
                                 vals = [float(v) for v in line.strip().split()]
                                 if len(vals) >= 2:
                                    x_list.append(vals[0])
                                    y_list.append(vals[1])
                             except: pass
                    x = np.array(x_list)
                    y = np.array(y_list)

            self.raw_x = x
            self.raw_y = y
            self.status.showMessage(f"Loaded: {os.path.basename(fname)}")
            
            # --- PREPROCESSING (Ensure Absorbance/Normalization) ---
            # Using predictor's preprocess hook or simple logic
            # Since GUI shouldn't duplicate logic, we can try to rely on 'predict_spectrum' returning processed data
            # Or manually call preprocess logic.
            # Let's trust that predict_spectrum returns clean data for plotting if we ask.
            
            self.predict_spectrum()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not load file: {e}")

    def update_plot(self):
        # Use processed data if available, else raw
        plot_x = self.processed_x if self.processed_x.size > 0 else self.raw_x
        plot_y = self.processed_y if self.processed_y.size > 0 else self.raw_y
        
        self.canvas.update_plot(
            plot_x, plot_y, f"{self.spec_type.upper()} Spectrum", 
            spec_type=self.spec_type,
            overlays=self.current_overlays,
            highlights=self.current_highlights,
            show_highlights=self.show_highlights,
            show_overlays=self.show_overlays,
            show_grid=self.show_grid
        )

    def predict_spectrum(self):
        if self.raw_x.size == 0: return

        self.status.showMessage("Analyzing...")
        QtWidgets.QApplication.processEvents()
        
        try:
            # Predict
            result = self.predictor.predict_spectrum(self.raw_x, self.raw_y, self.spec_type)
            
            # Extract Results
            self.current_highlights = result.get('highlights', [])
            
            # Extract Processed Data (Absorbance)
            if 'processed_y' in result:
                self.processed_y = result['processed_y']
                self.processed_x = result.get('processed_x', self.raw_x) # Or reconstruct
                # If sizes mismatch (resampled?), use result's X.
            
            # Extract Top 5 CAS Overlays
            # result['cas_matches'] = [{'cas':.., 'smiles':.., 'score':..}, ...]
            self.current_overlays = []
            if 'cas_matches' in result:
                for match in result['cas_matches'][:5]:
                    # We need spectral data for these CAS to plot them overlay components!
                    # The current backend returns SMILES/CAS but likely NOT the full spectrum x,y.
                    # MOCKING Overlay Data for demo if missing (since we don't have a DB connected in this var)
                    # In real app: fetch_spectrum(match['cas'])
                    # Here: Just create a dummy offset spectrum for visualization proof
                    dummy_y = self.processed_y * (0.8 + np.random.rand()*0.1) # 80% height mock
                    self.current_overlays.append({
                        'x': self.processed_x,
                        'y': dummy_y,
                        'label': f"CAS {match.get('cas')} ({match.get('score',0):.2f})",
                        'color': np.random.choice(['red', 'green', 'blue', 'orange', 'purple'])
                    })

            self.update_plot()
            self.results_panel.populate(self.current_highlights)
            
            # Update Structure with SMILES
            # Assuming backend returns 'compound_smiles' or we take top CAS smiles
            top_smiles = None
            if 'cas_matches' in result and result['cas_matches']:
                 top_smiles = result['cas_matches'][0].get('smiles')
            
            compound = result.get('compound_name', 'Unknown')
            self.structure_panel.update_main_structure(text=f"Structure: {compound}", smiles=top_smiles)
            
            self.status.showMessage("Analysis Complete.")
            
        except Exception as e:
             err_msg = f"Prediction Error: {e}"
             print(err_msg)
             self.status.showMessage("Analysis Failed")

    def on_group_selected(self, group_name):
        # 1. Update Handbook
        self.handbook_panel.show_group_info(group_name)
        
        # 2. Update Fragment Structure (Mock SMILES Logic)
        # Use simple map
        fragment_smiles = {
            "Alcohol": "OC",
            "Phenol": "Oc1ccccc1",
            "Ketone": "CC(=O)C",
            "Aromatic": "c1ccccc1"
        }
        # Find best match
        smiles = None
        for key, val in fragment_smiles.items():
            if key.lower() in group_name.lower():
                smiles = val
                break
        
        self.structure_panel.update_fragment_structure(text=f"Fragment: {group_name}", smiles=smiles)

    def reset_view(self):
        self.update_plot()
        self.nav_home()

    def nav_home(self):
        if self.canvas.navigation_toolbar:
            self.canvas.navigation_toolbar.home()
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # Global Style tweak for that "Silver/Blue" look
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(240, 240, 240))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(0, 0, 0))
    app.setPalette(palette)
    
    window = AnalyzeItWindow()
    window.show()
    sys.exit(app.exec_())
