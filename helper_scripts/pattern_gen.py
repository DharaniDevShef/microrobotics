import sys
import math
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QPolygonF
from PyQt6.QtCore import Qt, QPointF

class AssemblyGrid(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(650, 650)
        
        self.radius = 45  
        self.modules = {}
        self.build_perfect_interlock_grid()

    def build_perfect_interlock_grid(self):
        """Calculates precise non-overlapping flat-to-flat contact layout"""
        center_to_inner = self.radius * 1.67
        inner_to_outer = self.radius * 1.67 
        
        for i in range(6):
            angle_deg = i * 60
            inner_key = f'I{i+1}'
            pos_angle_rad = math.radians(angle_deg + 30)
            
            ix = center_to_inner * math.cos(pos_angle_rad)
            iy = center_to_inner * math.sin(pos_angle_rad)
            inner_rot = angle_deg + 90
            
            self.modules[inner_key] = {
                'pos': (ix, iy),
                'angle': inner_rot, 
                'active': True,
                'type': 'inner',
                'line_state': 0  # 0: grey dotted, 1: blue solid, 2: violet solid
            }
            
            outer_key = f'O{i+1}'
            ox = ix + inner_to_outer * math.cos(pos_angle_rad)
            oy = iy + inner_to_outer * math.sin(pos_angle_rad)
            outer_rot = inner_rot + 180
            
            self.modules[outer_key] = {
                'pos': (ox, oy),
                'angle': outer_rot,
                'active': True,
                'type': 'outer',
                'line_state': 0  
            }

    def get_module_polygon(self, cx, cy, r, orientation_deg):
        """Generates an accurate 3-tangent flattened polygon profile"""
        points = []
        num_sides = 3
        arc_segments = 24 
        
        for i in range(num_sides):
            corner_angle = math.radians(orientation_deg + (i * 360 / num_sides))
            
            for j in range(arc_segments + 1):
                factor = (j / arc_segments) - 0.5
                sweep_angle = corner_angle + (factor * math.pi / 3)
                current_r = r * (0.94 + 0.06 * math.cos(3 * (sweep_angle - math.radians(orientation_deg))))
                
                px = cx + current_r * math.cos(sweep_angle)
                py = cy - current_r * math.sin(sweep_angle)
                points.append(QPointF(px, py))
                
        return QPolygonF(points)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        center_x = self.width() / 2
        center_y = self.height() / 2

        for key, info in self.modules.items():
            x = center_x + info['pos'][0]
            y = center_y - info['pos'][1]
            
            if info['active']:
                fill_color = QColor(225, 225, 225) if info['type'] == 'inner' else QColor(210, 210, 210)
                line_color = QColor(50, 50, 50)
                line_width = 2
            else:
                fill_color = QColor(248, 248, 248)
                line_color = QColor(230, 230, 230)
                line_width = 1

            # Draw Base Module
            painter.setPen(QPen(line_color, line_width))
            painter.setBrush(QBrush(fill_color))
            module_poly = self.get_module_polygon(x, y, self.radius, info['angle'])
            painter.drawPolygon(module_poly)

            # Draw Interactive Interior Splitting Line (Only if module is active)
            if info['active']:
                # Calculate the angle parallel to the primary horizontal base edge
                # The line cuts cleanly across the module center splitting it in half
                line_angle_rad = math.radians(info['angle']+ 30)  # Perpendicular to the base edge
                
                # Length of the line segment spans slightly past module profile bounds
                line_len = self.radius * 0.95
                
                dx = line_len * math.cos(line_angle_rad)
                dy = line_len * math.sin(line_angle_rad)
                
                p1 = QPointF(x - dx, y + dy)
                p2 = QPointF(x + dx, y - dy)
                
                # Configure pen dynamic states
                if info['line_state'] == 1:       # First Click -> Thick Solid Blue
                    line_pen = QPen(QColor(0, 102, 204), 3, Qt.PenStyle.SolidLine)
                elif info['line_state'] == 2:     # Second Click -> Thick Solid Violet
                    line_pen = QPen(QColor(138, 43, 226), 3, Qt.PenStyle.SolidLine)
                else:                             # Initial / Fourth Click -> Thin Dotted Grey
                    line_pen = QPen(QColor(120, 120, 120), 1, Qt.PenStyle.DotLine)
                
                painter.setPen(line_pen)
                painter.drawLine(p1, p2)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            center_x = self.width() / 2
            center_y = self.height() / 2
            
            for key, info in self.modules.items():
                x = center_x + info['pos'][0]
                y = center_y - info['pos'][1]
                
                if math.hypot(event.position().x() - x, event.position().y() - y) <= self.radius:
                    # Cycles: 0 (Dotted Grey) -> 1 (Solid Blue) -> 2 (Solid Violet)
                    info['line_state'] = (info['line_state'] + 1) % 3
                    self.update()
                    break

    def clear_all(self):
        for key in self.modules:
            self.modules[key]['line_state'] = 0
            self.modules[key]['active'] = False
        self.update()

    def reset_all(self):
        for key in self.modules:
            self.modules[key]['line_state'] = 0
            self.modules[key]['active'] = True
        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Internal Seam Alignment Tool")
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        self.label = QLabel("Click inside modules to toggle center splitting line color states")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("font-size: 13px; font-weight: bold; color: #333; margin: 8px;")
        layout.addWidget(self.label)
        
        self.grid_canvas = AssemblyGrid()
        layout.addWidget(self.grid_canvas)
        
        btn_layout = QHBoxLayout()
        self.clear_btn = QPushButton("Clear Canvas")
        self.reset_btn = QPushButton("Reset Design")
        
        self.clear_btn.clicked.connect(self.grid_canvas.clear_all)
        self.reset_btn.clicked.connect(self.grid_canvas.reset_all)
        
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.reset_btn)
        layout.addLayout(btn_layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())