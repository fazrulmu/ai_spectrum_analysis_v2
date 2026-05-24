from PyQt5 import QtWidgets, QtCore, QtGui
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import numpy as np

# Try importing NavigationToolbar
try:
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
except ImportError:
    NavigationToolbar2QT = None

class UnifiedCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#f0f0f0') 
        
        self.ax1 = self.fig.add_subplot(111) # Single Plot
        
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.updateGeometry()
        self.fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95)
        
        self.navigation_toolbar = None
        if NavigationToolbar2QT:
            self.navigation_toolbar = NavigationToolbar2QT(self, parent)
            self.navigation_toolbar.hide()

    def update_plot(self, x_data, y_data, title_prefix, spec_type='ir', 
                    overlays=None, highlights=None, show_highlights=True, show_overlays=True, 
                    show_grid=True, reverse_x=True, matrix_rules=None, selected_matrix=None):
        
        self.ax1.clear()
        
        if x_data.size == 0:
            self.ax1.set_title(f"{title_prefix} (No Data)", fontsize=11, fontweight='bold', pad=10)
            self.draw()
            return

        # --- Plot Sample (Black) ---
        x, y = x_data, y_data
        
        if spec_type == 'ms':
            self.ax1.stem(x, y, linefmt='purple', markerfmt='mo', basefmt=" ")
        else:
            self.ax1.plot(x, y, label='Sample', color='black', linewidth=1.5, zorder=3)
            
        self.ax1.set_title(f"{title_prefix} - Sample View", fontsize=10, fontweight='bold')
        self.ax1.grid(show_grid, linestyle='--', alpha=0.6)
        
        # Axis Config
        if spec_type == 'ir' or spec_type == 'nmr':
             if reverse_x: self.ax1.invert_xaxis()
        if spec_type == 'ir': self.ax1.set_xlabel("Wavenumber (cm⁻¹)")
        elif spec_type == 'uv': self.ax1.set_xlabel("Wavelength (nm)")
        
        # --- Overlays (Top 5 CAS) ---
        # overlays = [{'x':.., 'y':.., 'label':.., 'color':..}, ...]
        if show_overlays and overlays:
            for item in overlays[:5]: # Cap at 5
                ox = item.get('x')
                oy = item.get('y')
                lbl = item.get('label', 'Ref')
                col = item.get('color', 'gray')
                
                if ox is not None and oy is not None:
                     self.ax1.plot(ox, oy, label=lbl, color=col, linestyle='--', linewidth=1.0, alpha=0.7, zorder=2)
            
            # Add Legend if overlays exist
            self.ax1.legend(fontsize=8, loc='upper right')

        # --- Highlights (Cyan Style) ---
        if show_highlights and highlights:
             for i, item in enumerate(highlights):
                r_start, r_end = item['range']
                r_min, r_max = min(r_start, r_end), max(r_start, r_end)
                label = item['label']
                
                mask = (x >= r_min) & (x <= r_max)
                if np.any(mask):
                    x_region = x[mask]
                    y_region = y[mask]
                    self.ax1.fill_between(x_region, y_region, color='cyan', alpha=0.5, zorder=1)
                    
                    peak_idx = np.argmax(y_region)
                    peak_x = x_region[peak_idx]
                    peak_y = y_region[peak_idx]
                    
                    self.ax1.annotate(
                        f"{i+1}", 
                        xy=(peak_x, peak_y),
                        xytext=(0, 15),
                        textcoords='offset points',
                        ha='center',
                        fontsize=9,
                        fontweight='bold',
                        bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="black", lw=1)
                    )

        # --- Matrix Red Zones ---
        if spec_type == 'ir' and show_highlights and matrix_rules and selected_matrix:
             if selected_matrix in matrix_rules:
                    rule = matrix_rules[selected_matrix]
                    for r_start, r_end in rule.get("avoid_ranges", []):
                        self.ax1.axvspan(r_start, r_end, color='#e74c3c', alpha=0.08, hatch='///')

        self.draw()
