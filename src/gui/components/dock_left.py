from PyQt5 import QtWidgets, QtCore, QtGui

class ContextSettingsDock(QtWidgets.QDockWidget):
    # Signals to communicate with the main window
    settingChanged = QtCore.pyqtSignal()
    analyzeClicked = QtCore.pyqtSignal()
    statsClicked = QtCore.pyqtSignal()
    peaksClicked = QtCore.pyqtSignal()
    unitChanged = QtCore.pyqtSignal(str)
    reverseXChanged = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__("Properties", parent)
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.init_ui()

    def init_ui(self):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(15)
        
        # Group: Results Summary (AnalyzeIt Style)
        group_res = QtWidgets.QGroupBox("AI Analysis Summary")
        res_layout = QtWidgets.QVBoxLayout()
        
        self.results_table = QtWidgets.QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Group", "Range", "Conf"])
        self.results_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.results_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.results_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.results_table.setMinimumHeight(150)
        res_layout.addWidget(self.results_table)
        group_res.setLayout(res_layout)
        layout.addWidget(group_res)

        # Group: Context Settings (Stacked Widget)
        group_settings = QtWidgets.QGroupBox("Context Settings")
        settings_layout = QtWidgets.QVBoxLayout()
        
        self.settings_stack = QtWidgets.QStackedWidget()
        
        # Page 0: IR Settings
        page_ir = QtWidgets.QWidget()
        form_ir = QtWidgets.QFormLayout()
        self.matrix_combo = QtWidgets.QComboBox()
        self.matrix_combo.addItems(["ATR_NEAT", "KBR", "NUJOL", "CCL4", "CS2", "COMPOSITE"])
        form_ir.addRow("Matrix:", self.matrix_combo)
        
        self.reverse_x_cb = QtWidgets.QCheckBox("Reverse X-Axis")
        self.reverse_x_cb.setChecked(True)
        self.reverse_x_cb.stateChanged.connect(self.reverseXChanged.emit)
        form_ir.addRow(self.reverse_x_cb)
        page_ir.setLayout(form_ir)
        self.settings_stack.addWidget(page_ir) # Index 0

        # Page 1: UV Settings
        page_uv = QtWidgets.QWidget()
        form_uv = QtWidgets.QFormLayout()
        self.solvent_combo = QtWidgets.QComboBox()
        self.solvent_combo.addItems(["Ethanol", "Methanol", "Water", "Acetonitrile", "Hexane"])
        form_uv.addRow("Solvent:", self.solvent_combo)
        page_uv.setLayout(form_uv)
        self.settings_stack.addWidget(page_uv) # Index 1

        # Page 2: NMR Settings
        page_nmr = QtWidgets.QWidget()
        form_nmr = QtWidgets.QFormLayout()
        self.nmr_solvent_combo = QtWidgets.QComboBox()
        self.nmr_solvent_combo.addItems(["CDCl3", "DMSO-d6", "D2O", "Acetone-d6"])
        form_nmr.addRow("Solvent:", self.nmr_solvent_combo)
        self.nmr_freq = QtWidgets.QSpinBox()
        self.nmr_freq.setRange(60, 900)
        self.nmr_freq.setValue(400)
        self.nmr_freq.setSuffix(" MHz")
        form_nmr.addRow("Frequency:", self.nmr_freq)
        page_nmr.setLayout(form_nmr)
        self.settings_stack.addWidget(page_nmr) # Index 2
        
        # Page 3: MS Settings
        page_ms = QtWidgets.QWidget()
        form_ms = QtWidgets.QFormLayout()
        self.ms_mode_combo = QtWidgets.QComboBox()
        self.ms_mode_combo.addItems(["EI", "ESI", "CI", "MALDI"])
        form_ms.addRow("Ionization:", self.ms_mode_combo)
        page_ms.setLayout(form_ms)
        self.settings_stack.addWidget(page_ms) # Index 3

        # Page 4: Chrom Settings
        page_chrom = QtWidgets.QWidget()
        form_chrom = QtWidgets.QFormLayout()
        self.chrom_col_combo = QtWidgets.QComboBox()
        self.chrom_col_combo.addItems(["C18", "C8", "Silica", "Phenyl"])
        form_chrom.addRow("Column:", self.chrom_col_combo)
        page_chrom.setLayout(form_chrom)
        self.settings_stack.addWidget(page_chrom) # Index 4
        
        settings_layout.addWidget(self.settings_stack)
        group_settings.setLayout(settings_layout)
        layout.addWidget(group_settings)
        
        # Global Data Settings (Unit)
        group_data = QtWidgets.QGroupBox("Data Settings")
        form_data = QtWidgets.QFormLayout()
        self.unit_combo = QtWidgets.QComboBox()
        self.unit_combo.addItems(["cm-1", "µm", "nm", "ppm", "m/z"])
        self.unit_combo.currentTextChanged.connect(self.unitChanged.emit)
        form_data.addRow("Unit:", self.unit_combo)
        group_data.setLayout(form_data)
        layout.addWidget(group_data)

        # Group: Analysis Actions
        group_anl = QtWidgets.QGroupBox("Actions")
        a_layout = QtWidgets.QVBoxLayout()
        
        btn_predict = QtWidgets.QPushButton("🚀 Predict AI")
        btn_predict.setStyleSheet("font-weight: bold; background-color: #d62728; color: white; padding: 5px;")
        btn_predict.clicked.connect(self.analyzeClicked.emit)
        a_layout.addWidget(btn_predict)
        
        h_layout = QtWidgets.QHBoxLayout()
        btn_stats = QtWidgets.QPushButton("📈 Stats")
        btn_stats.clicked.connect(self.statsClicked.emit)
        h_layout.addWidget(btn_stats)
        
        btn_peaks = QtWidgets.QPushButton("🏔️ Peaks")
        btn_peaks.clicked.connect(self.peaksClicked.emit)
        h_layout.addWidget(btn_peaks)
        
        a_layout.addLayout(h_layout)
        group_anl.setLayout(a_layout)
        layout.addWidget(group_anl)
        
        layout.addStretch()
        content.setLayout(layout)
        scroll.setWidget(content)
        self.setWidget(scroll)

    def set_context_index(self, index):
        self.settings_stack.setCurrentIndex(index)

    def update_results_table(self, highlights):
        """Populate the Results Table"""
        self.results_table.setRowCount(0)
        for row, item in enumerate(highlights):
            self.results_table.insertRow(row)
            self.results_table.setItem(row, 0, QtWidgets.QTableWidgetItem(item['label']))
            r_start, r_end = item['range']
            self.results_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{r_start}-{r_end}"))
            conf_item = QtWidgets.QTableWidgetItem(f"{item.get('confidence', 0.0):.2f}")
            if item.get('confidence', 0.0) > 0.8: conf_item.setBackground(QtGui.QColor("#d4edda"))
            self.results_table.setItem(row, 2, conf_item)
