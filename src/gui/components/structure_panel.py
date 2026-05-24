from PyQt5 import QtWidgets, QtCore, QtGui

class StructurePanel(QtWidgets.QWidget):
    """
    Left Panel: Displays the main compound structure and selected fragment structure.
    Matches 'Structure' and 'Selected Fragment Structure' from reference.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(2)
        
        # 1. Main Structure Section
        self.main_struct_group = self.create_header_frame("Structure")
        self.main_struct_label = QtWidgets.QLabel("No Structure Loaded")
        self.main_struct_label.setAlignment(QtCore.Qt.AlignCenter)
        self.main_struct_label.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        self.main_struct_label.setMinimumHeight(150)
        
        self.layout.addWidget(self.main_struct_group)
        self.layout.addWidget(self.main_struct_label, 1) # Expandable
        
        # 2. Fragment Structure Section
        self.frag_struct_group = self.create_header_frame("Selected Fragment Structure")
        self.frag_struct_label = QtWidgets.QLabel("Select a group...")
        self.frag_struct_label.setAlignment(QtCore.Qt.AlignCenter)
        self.frag_struct_label.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        self.frag_struct_label.setMinimumHeight(150)
        
        self.layout.addWidget(self.frag_struct_group)
        self.layout.addWidget(self.frag_struct_label, 1) # Expandable
        
        # Bottom Info
        self.info_label = QtWidgets.QLabel("* - Any Attachments")
        self.info_label.setStyleSheet("font-style: italic; color: #555; padding: 5px;")
        self.layout.addWidget(self.info_label)

    def create_header_frame(self, title):
        """Creates a styled header look-alike (Gradient background)"""
        frame = QtWidgets.QFrame()
        frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e0e0e0, stop:1 #b0b0b0);
                border: 1px solid #999;
                border-radius: 2px;
            }
        """)
        framelayout = QtWidgets.QHBoxLayout(frame)
        framelayout.setContentsMargins(5, 2, 5, 2)
        
        lbl = QtWidgets.QLabel(title)
        lbl.setStyleSheet("font-weight: bold; color: #333; background: transparent; border: none;")
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        framelayout.addWidget(lbl)
        
        return frame

    def update_main_structure(self, image_path=None, text=None, smiles=None):
        if smiles:
             pix = self.render_smiles_to_pixmap(smiles)
             if pix:
                 self.main_struct_label.setPixmap(pix)
                 return

        if image_path:
            pixmap = QtGui.QPixmap(image_path)
            self.main_struct_label.setPixmap(pixmap.scaled(self.main_struct_label.size(), QtCore.Qt.KeepAspectRatio))
        elif text:
            self.main_struct_label.setText(text)
            
    def update_fragment_structure(self, image_path=None, text=None, smiles=None):
        if smiles:
             pix = self.render_smiles_to_pixmap(smiles)
             if pix:
                 self.frag_struct_label.setPixmap(pix)
                 return

        if image_path:
            pixmap = QtGui.QPixmap(image_path)
            self.frag_struct_label.setPixmap(pixmap.scaled(self.frag_struct_label.size(), QtCore.Qt.KeepAspectRatio))
        elif text:
            self.frag_struct_label.setText(text)

    def render_smiles_to_pixmap(self, smiles):
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            
            mol = Chem.MolFromSmiles(smiles)
            if not mol: return None
            
            # Draw to image
            img = Draw.MolToImage(mol, size=(300, 200)) # PIL Image
            
            # Convert PIL to QPixmap
            from PyQt5.QtGui import QImage, QPixmap
            
            if img.mode == "RGB":
                r, g, b = img.split()
                img = Image.merge("RGB", (b, g, r))
                
            im_data = img.convert("RGBA").tobytes("raw", "RGBA")
            qim = QImage(im_data, img.size[0], img.size[1], QImage.Format_RGBA8888)
            return QPixmap.fromImage(qim)
            
        except ImportError:
            return None
        except Exception as e:
            print(f"SMILES Render Error: {e}")
            return None
