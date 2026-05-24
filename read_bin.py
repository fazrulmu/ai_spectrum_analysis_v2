import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("View Toggle Developer PyQt5")
        self.setGeometry(100, 100, 500, 400)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)
        
        self.debug_mode = False

        self.setup_ui()

    def setup_ui(self):
        # Tombol Toggle
        self.toggle_button = QPushButton("Toggle Debug View", self)
        self.toggle_button.clicked.connect(self.toggle_debug_view)
        self.main_layout.addWidget(self.toggle_button)

        # Container untuk "Cards" atau "Fragments"
        cards_container = QWidget()
        cards_layout = QHBoxLayout()
        cards_container.setLayout(cards_layout)

        # Placeholder untuk "Card View" (QLabel sebagai contoh)
        self.card1 = QLabel("Card/Fragment 1")
        self.card1.setAlignment(Qt.AlignCenter)
        self.card1.setFixedSize(150, 100)
        
        self.card2 = QLabel("Card/Fragment 2")
        self.card2.setAlignment(Qt.AlignCenter)
        self.card2.setFixedSize(150, 100)

        cards_layout.addWidget(self.card1)
        cards_layout.addWidget(self.card2)
        
        self.main_layout.addWidget(cards_container)

    def toggle_debug_view(self):
        self.debug_mode = not self.debug_mode
        
        if self.debug_mode:
            # Terapkan stylesheet untuk menampilkan batas (border)
            debug_stylesheet = "border: 2px solid red; background-color: #f0f0f0;"
            self.card1.setStyleSheet(debug_stylesheet)
            self.card2.setStyleSheet(debug_stylesheet)
            self.toggle_button.setText("Debug View AKTIF")
        else:
            # Hapus stylesheet
            self.card1.setStyleSheet("")
            self.card2.setStyleSheet("")
            self.toggle_button.setText("Toggle Debug View")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
