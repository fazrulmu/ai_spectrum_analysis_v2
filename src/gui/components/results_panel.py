from PyQt5 import QtWidgets, QtCore, QtGui

class ResultTree(QtWidgets.QTreeWidget):
    """
    Bottom Left: Hierarchical Tree of Analysis Results.
    Cols: [Status, Classification, Group, Bond, Range]
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["S..", "Classification", "Group", "Bond", "Range"])
        self.setColumnWidth(0, 30) # Icon col
        self.setColumnWidth(1, 150)
        self.setAlternatingRowColors(True)
        self.setRootIsDecorated(True)

    def populate(self, highlights):
        """
        highlights: List of dicts {'label', 'range', 'color', 'confidence'}
        """
        self.clear()
        
        # Categorize (Mock logic for hierarchy)
        categories = {}
        for hl in highlights:
            lbl = hl.get('label', 'Unknown')
            # Fake logic to bucket
            cat = "Others"
            if "Alcohol" in lbl or "OH" in lbl: cat = "Alcohols"
            elif "Alkane" in lbl or "CH" in lbl: cat = "Alkanes"
            elif "Aromatic" in lbl: cat = "Aromatics"
            elif "C=O" in lbl: cat = "Carbonyls"
            
            if cat not in categories: categories[cat] = []
            categories[cat].append(hl)
            
        # Add to Tree
        for cat, items in categories.items():
            cat_item = QtWidgets.QTreeWidgetItem(self)
            cat_item.setText(1, cat)
            cat_item.setExpanded(True)
            
            for hl in items:
                item = QtWidgets.QTreeWidgetItem(cat_item)
                item.setText(0, "+") # Fake status icon
                item.setText(2, hl.get('label'))
                item.setText(3, "N/A") # Bond info mock
                rng = hl.get('range', [0,0])
                item.setText(4, f"{int(rng[0])}-{int(rng[1])}")
                
                # Colorize background of row slightly to match group?
                # item.setBackground(1, QtGui.QColor(hl.get('color'))) 

class SpectralBarWidget(QtWidgets.QWidget):
    """
    Bottom Right: Visual 'Barcode' of spectral ranges.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        self.highlights = []
        self.x_range = (4000, 400) # Standard IR
        
    def set_data(self, highlights):
        self.highlights = highlights
        self.update() # Trigger repaint
        
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Draw tracks
        # For each highlight, draw a rect
        # Map X range 4000->400 to 0->w
        
        span = self.x_range[0] - self.x_range[1]
        
        row_height = 20
        y_offset = 10
        
        for i, hl in enumerate(self.highlights):
            # Calculate x pos
            r_start, r_end = hl.get('range', [0,0])
            
            # x = (val - min) / (max - min) * w
            # But IR is reversed: 4000 is left (0), 400 is right (w)
            
            x1 = (self.x_range[0] - r_start) / span * w
            x2 = (self.x_range[0] - r_end) / span * w
            
            rect_x = min(x1, x2)
            rect_w = abs(x2 - x1)
            
            color = QtGui.QColor(hl.get('color', '#000000'))
            painter.setBrush(color)
            painter.setPen(QtCore.Qt.NoPen)
            
            # Draw rect
            # Stagger rows if too many?
            y = (i % 5) * row_height + y_offset
            painter.drawRect(int(rect_x), int(y), int(max(2, rect_w)), int(row_height - 5))

class ResultsPanel(QtWidgets.QSplitter):
    """
    Splitter containing ResultTree (Left) and SpectralBarWidget (Right).
    """
    groupSelected = QtCore.pyqtSignal(str) # Emits group name on click
    
    def __init__(self, parent=None):
        super().__init__(QtCore.Qt.Horizontal, parent)
        
        # Left: Tree
        self.tree = ResultTree()
        self.tree.itemClicked.connect(self.on_item_clicked)
        self.addWidget(self.tree)
        
        # Right: Bars
        self.bars = SpectralBarWidget()
        self.addWidget(self.bars)
        
        # Sizing
        self.setStretchFactor(0, 1)
        self.setStretchFactor(1, 3) # Bars take more space
        
    def populate(self, highlights):
        self.tree.populate(highlights)
        self.bars.set_data(highlights)
        
    def on_item_clicked(self, item, col):
        grp = item.text(2)
        if grp and grp != "N/A":
            self.groupSelected.emit(grp)
