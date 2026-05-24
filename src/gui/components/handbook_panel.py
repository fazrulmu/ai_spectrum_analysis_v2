from PyQt5 import QtWidgets, QtCore, QtGui

class HandbookPanel(QtWidgets.QWidget):
    """
    Right Panel: Displays rich text info about selected groups.
    Matches 'Sadtler Handbook' from reference.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Header
        self.header = self.create_header_frame("Sadtler Handbook")
        self.layout.addWidget(self.header)
        
        # Text Browser
        self.text_browser = QtWidgets.QTextBrowser()
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.setStyleSheet("border: 1px solid #ccc; border-top: none;")
        self.layout.addWidget(self.text_browser)
        
        # Initial Content
        self.set_default_content()

    def create_header_frame(self, title):
        frame = QtWidgets.QFrame()
        frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #dcecfb, stop:1 #bedaf7);
                border: 1px solid #999;
                border-bottom: none;
                border-top-left-radius: 2px;
                border-top-right-radius: 2px;
            }
        """)
        framelayout = QtWidgets.QHBoxLayout(frame)
        framelayout.setContentsMargins(5, 2, 5, 2)
        
        lbl = QtWidgets.QLabel(title)
        lbl.setStyleSheet("font-weight: bold; color: #222; background: transparent; border: none;")
        framelayout.addWidget(lbl)
        framelayout.addStretch()
        
        # Minimal window controls for look
        btn_min = QtWidgets.QPushButton("-")
        btn_min.setFixedSize(16, 16)
        btn_max = QtWidgets.QPushButton("□")
        btn_max.setFixedSize(16, 16)
        btn_close = QtWidgets.QPushButton("x")
        btn_close.setFixedSize(16, 16)
        
        for btn in [btn_min, btn_max, btn_close]:
             btn.setStyleSheet("border: none; background: transparent; font-size: 10px;")
             framelayout.addWidget(btn)
        
        return frame

    def set_content(self, title, html_body):
        html = f"""
        <h2 style='font-family: sans-serif; color: #333;'>{title}</h2>
        <hr>
        <div style='font-family: sans-serif; font-size: 13px; color: #444;'>
        {html_body}
        </div>
        """
        self.text_browser.setHtml(html)

    def set_default_content(self):
        self.set_content("Welcome", """
        <p>Select a functional group from the <b>Summary Tree</b> (bottom left) to view detailed handbook information here.</p>
        <p>This panel displays:</p>
        <ul>
            <li>Group description</li>
            <li>Characteristic absorption bands</li>
            <li>Structural diagrams</li>
        </ul>
        """)
    
    def show_group_info(self, group_name):
        # Mock Data Logic - In real app, fetch from database
        info = { # Simple kb
            "Alcohols": """
            <h3>Alcohols (O-H Stretch)</h3>
            <p>1. <b>O-H stretching vibration:</b></p>
            <ul>
                <li>Free O-H: 3650-3590 cm⁻¹ (sharp)</li>
                <li>H-bonded (dimer): near 3500 cm⁻¹</li>
                <li>H-bonded (polymer): 3400-3200 cm⁻¹ (broad, strong)</li>
            </ul>
            <p>2. <b>C-O stretching vibration:</b> 1260-1000 cm⁻¹</p>
            """,
            "Ketones": """
            <h3>Ketones (C=O Stretch)</h3>
            <p>Saturated acyclic ketones absorb at 1715 cm⁻¹.</p>
            <p>Conjugation lowers frequency to 1690 cm⁻¹.</p>
            """,
            "Aromatics": """
            <h3>Aromatics (C-H & C=C)</h3>
            <p><b>C-H Stretch:</b> >3000 cm⁻¹ (typically 3030 cm⁻¹).</p>
            <p><b>C=C Ring Stretch:</b> Pairs at 1600 cm⁻¹ and 1475 cm⁻¹.</p>
            """
        }
        
        content = info.get(group_name, f"<p>No handbook entry for <b>{group_name}</b>.</p>")
        self.set_content(group_name, content)
