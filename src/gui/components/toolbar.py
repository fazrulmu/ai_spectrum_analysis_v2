from PyQt5 import QtWidgets, QtCore, QtGui

class MainToolbar(QtWidgets.QToolBar):
    # Signals
    loadActionTriggered = QtCore.pyqtSignal(str) # 'ir', 'uv', etc.
    homeClicked = QtCore.pyqtSignal()
    panToggled = QtCore.pyqtSignal(bool)
    zoomToggled = QtCore.pyqtSignal(bool)
    saveClicked = QtCore.pyqtSignal()
    highlightToggled = QtCore.pyqtSignal(bool)
    gridToggled = QtCore.pyqtSignal(bool)
    resetViewClicked = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setIconSize(QtCore.QSize(24, 24))
        self.setStyleSheet("QToolBar { spacing: 5px; }")
        self.init_ui()

    def init_ui(self):
        # Load Button
        load_btn = QtWidgets.QToolButton()
        load_btn.setText("Load")
        load_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon))
        load_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        load_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        
        load_menu = QtWidgets.QMenu()
        act_ir = load_menu.addAction("Load IR Spectrum")
        act_ir.triggered.connect(lambda: self.loadActionTriggered.emit('ir'))
        act_uv = load_menu.addAction("Load UV Spectrum")
        act_uv.triggered.connect(lambda: self.loadActionTriggered.emit('uv'))
        act_nmr = load_menu.addAction("Load NMR Spectrum")
        act_nmr.triggered.connect(lambda: self.loadActionTriggered.emit('nmr'))
        act_ms = load_menu.addAction("Load Mass Spectrum")
        act_ms.triggered.connect(lambda: self.loadActionTriggered.emit('ms'))
        
        load_btn.setMenu(load_menu)
        self.addWidget(load_btn)
        
        self.addSeparator()
        
        # Navigation
        nav_home = QtWidgets.QAction(self.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton), "Home", self)
        nav_home.triggered.connect(self.homeClicked.emit)
        self.addAction(nav_home)
        
        nav_pan = QtWidgets.QAction(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowUp), "Pan", self)
        nav_pan.setCheckable(True)
        nav_pan.toggled.connect(self.panToggled.emit)
        self.addAction(nav_pan)
        self.nav_pan_action = nav_pan
        
        nav_zoom = QtWidgets.QAction(self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogContentsView), "Zoom", self)
        nav_zoom.setCheckable(True)
        nav_zoom.toggled.connect(self.zoomToggled.emit)
        self.addAction(nav_zoom)
        self.nav_zoom_action = nav_zoom
        
        self.addSeparator()
        
        # Visualization Toggles
        highlight_act = QtWidgets.QAction("👁️", self)
        highlight_act.setToolTip("Toggle Highlights")
        highlight_act.setCheckable(True)
        highlight_act.setChecked(True)
        highlight_act.toggled.connect(self.highlightToggled.emit)
        self.act_highlight = highlight_act
        self.addAction(highlight_act)
        
        grid_act = QtWidgets.QAction("📊", self)
        grid_act.setToolTip("Toggle Grid")
        grid_act.setCheckable(True)
        grid_act.setChecked(True)
        grid_act.toggled.connect(self.gridToggled.emit)
        self.act_grid = grid_act
        self.addAction(grid_act)
        
        reset_act = QtWidgets.QAction("🏠", self)
        reset_act.setToolTip("Reset View")
        reset_act.triggered.connect(self.resetViewClicked.emit)
        self.addAction(reset_act)
        
        self.addSeparator()
        
        # Save
        save_act = QtWidgets.QAction(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton), "Save", self)
        save_act.triggered.connect(self.saveClicked.emit)
        self.addAction(save_act)
