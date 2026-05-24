from PyQt5 import QtWidgets, QtCore

class AssistantDock(QtWidgets.QDockWidget):
    messageSent = QtCore.pyqtSignal(str)
    linkClicked = QtCore.pyqtSignal(QtCore.QUrl)

    def __init__(self, parent=None):
        super().__init__("Data & Assistant", parent)
        self.setAllowedAreas(QtCore.Qt.RightDockWidgetArea | QtCore.Qt.LeftDockWidgetArea)
        self.init_ui()

    def init_ui(self):
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        
        # Chat History
        self.chat_history = QtWidgets.QTextBrowser()
        self.chat_history.setReadOnly(True)
        self.chat_history.setOpenExternalLinks(False)
        self.chat_history.anchorClicked.connect(self.linkClicked.emit)
        layout.addWidget(self.chat_history)
        
        # Input Area
        input_layout = QtWidgets.QHBoxLayout()
        self.chat_input = QtWidgets.QLineEdit()
        self.chat_input.setPlaceholderText("Ask AI...")
        self.chat_input.returnPressed.connect(self.on_send_clicked)
        input_layout.addWidget(self.chat_input)
        
        send_btn = QtWidgets.QPushButton("Send")
        send_btn.clicked.connect(self.on_send_clicked)
        input_layout.addWidget(send_btn)
        
        layout.addLayout(input_layout)
        
        self.setWidget(content)

    def on_send_clicked(self):
        msg = self.chat_input.text().strip()
        if msg:
            self.messageSent.emit(msg)
            self.chat_input.clear()

    def append_message(self, sender, content):
        if sender == "AI":
            color = "#2c3e50"
            bg = "#e3f2fd"
            align = "left"
        elif sender == "System":
            color = "#7f8c8d"
            bg = "#f0f0f0"
            align = "center"
        else: # User
            color = "#2980b9"
            bg = "#ffffff"
            align = "right"
            
        html = f"""
        <div style='text-align:{align}; margin:5px;'>
            <div style='display:inline-block; background-color:{bg}; padding:8px; border-radius:10px; color:{color}; text-align:left;'>
                <b>{sender}:</b> {content}
            </div>
        </div>
        """
        self.chat_history.append(html)
