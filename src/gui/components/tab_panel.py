from PyQt5 import QtWidgets, QtCore, QtGui
from .canvas import UnifiedCanvas

class SpectrumTabPanel(QtWidgets.QWidget):
    """
    A scrollable tab panel containing:
    1. Unified Canvas (Plot)
    2. Analysis Summary (Text Area)
    3. Action Buttons (Analyze, Clear, Export)
    """
    # Signal to trigger analysis in main window
    analyzeTriggered = QtCore.pyqtSignal()
    
    def __init__(self, parent=None, spec_type='ir'):
        super().__init__(parent)
        self.spec_type = spec_type
        
        # Main Layout
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll Area
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.main_layout.addWidget(self.scroll_area)
        
        # Container Widget for Scroll Content
        self.content_widget = QtWidgets.QWidget()
        self.scroll_layout = QtWidgets.QVBoxLayout(self.content_widget)
        self.scroll_layout.setSpacing(15)
        self.scroll_layout.setContentsMargins(10, 10, 10, 10)
        self.scroll_area.setWidget(self.content_widget)
        
        # 1. Canvas
        self.canvas_container = QtWidgets.QWidget()
        self.canvas_layout = QtWidgets.QVBoxLayout(self.canvas_container)
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)
        # Minimum height to ensure plot is readable and scroll activates if needed
        self.canvas_container.setMinimumHeight(500) 
        
        self.canvas = UnifiedCanvas(self)
        self.canvas_layout.addWidget(self.canvas)
        self.scroll_layout.addWidget(self.canvas_container)
        
        # 2. Results Section
        self.results_group = QtWidgets.QGroupBox("Analysis Summary")
        self.results_layout = QtWidgets.QVBoxLayout(self.results_group)
        
        self.summary_text = QtWidgets.QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setPlaceholderText("Analysis results will appear here...")
        self.summary_text.setMinimumHeight(150)
        self.results_layout.addWidget(self.summary_text)
        
        self.scroll_layout.addWidget(self.results_group)
        
        # 3. Actions Toolbar (Inside Tab)
        self.actions_layout = QtWidgets.QHBoxLayout()
        self.actions_layout.addStretch()
        
        self.btn_analyze = QtWidgets.QPushButton("⚡ Analyze Spectrum")
        self.btn_analyze.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71; 
                color: white; 
                font-weight: bold; 
                padding: 8px 15px; 
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #27ae60; }
        """)
        self.btn_analyze.clicked.connect(self.analyzeTriggered.emit)
        
        self.btn_clear = QtWidgets.QPushButton("🧹 Clear Results")
        self.btn_clear.clicked.connect(self.clear_results)
        
        self.btn_export = QtWidgets.QPushButton("💾 Export Report")
        self.btn_export.clicked.connect(self.export_report)
        
        self.actions_layout.addWidget(self.btn_clear)
        self.actions_layout.addWidget(self.btn_export)
        self.actions_layout.addWidget(self.btn_analyze)
        
        self.scroll_layout.addLayout(self.actions_layout)
        
        # Push content to top
        self.scroll_layout.addStretch()

    def update_summary(self, text):
        self.summary_text.setHtml(text)
        
    def append_summary(self, text):
        self.summary_text.append(text)
        
    def clear_results(self):
        self.summary_text.clear()
        
    def export_report(self):
        text = self.summary_text.toPlainText()
        if not text:
            QtWidgets.QMessageBox.information(self, "Info", "No analysis to export.")
            return
            
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Report", "", "Text Files (*.txt)")
        if path:
            with open(path, 'w') as f:
                f.write(text)
            QtWidgets.QMessageBox.information(self, "Success", f"Report saved to {path}")
