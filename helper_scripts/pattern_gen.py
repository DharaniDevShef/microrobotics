import sys
import math
import json
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QPolygonF, QFont
from PyQt6.QtCore import Qt, QPointF

class AssemblyGrid(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(750, 750)
        
        self.radius = 45  
        self.modules = {}
        self.on_graph_update = None
        self.build_perfect_interlock_grid()

    def get_hinge_info(self, state):
        if state == 1:
            return {'color': 'blue', 'role': 'bottom'}
        if state == 2:
            return {'color': 'violet', 'role': 'top'}
        return {'color': 'none', 'role': 'none'}

    def get_graph_data(self):
        modules = []

        for key, info in self.modules.items():
            if not info['active']:
                continue

            module = {
                'id': key,
                'type': info['type'],
                'position': [round(info['pos'][0], 3), round(info['pos'][1], 3)],
                'angle': round(info['angle'], 1),
                'hinges': []
            }

            for idx, state in enumerate(info['line_states']):
                if state == 0:
                    continue

                hinge = self.get_hinge_info(state)
                module['hinges'].append({
                    'slot': idx,
                    'state': state,
                    'color': hinge['color'],
                    'role': hinge['role'],
                    'type': 'hinge',
                    'axis': [1, 0, 0],
                    'joint_range': [-45, 0],
                    'limited': True,
                    'armature': 0.001,
                    'damping': 0
                })

            modules.append(module)

        return {'assembly': {'modules': modules}}

    def get_graph_text(self):
        return json.dumps(self.get_graph_data(), indent=2)

    def notify_graph_update(self):
        if self.on_graph_update:
            self.on_graph_update(self.get_graph_text())

    def load_graph_data(self, graph_data):
        assembly = graph_data.get('assembly', {})
        modules_data = assembly.get('modules', [])
        self.modules.clear()

        for module_data in modules_data:
            module_id = module_data.get('id')
            if not module_id:
                continue

            pos = module_data.get('position', [0, 0])
            angle = module_data.get('angle', 0)
            module_type = module_data.get('type', 'outer')
            line_states = [0, 0, 0]

            for hinge in module_data.get('hinges', []):
                slot = hinge.get('slot', 0)
                state = hinge.get('state', 0)
                if 0 <= slot < 3 and state in (1, 2):
                    line_states[slot] = state

            self.modules[module_id] = {
                'pos': (pos[0], pos[1]),
                'angle': angle,
                'active': True,
                'type': module_type,
                'line_states': line_states
            }

        self.update()
        self.notify_graph_update()

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
                'line_states': [0, 0, 0]  # Tracks state independently for the 3 distinct lines
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
                'line_states': [0, 0, 0]  
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

    def get_edge_segments(self, cx, cy, r, orientation_deg):
        """Returns the 3 major flat tangent edge lines aligned with the mating faces"""
        edges = []
        num_sides = 3
        dist_to_edge = r * 0.81
        half_edge_w = r * 0.55  
        
        for i in range(num_sides):
            face_angle = math.radians(orientation_deg + 60 + (i * 360 / num_sides))
            mx = cx + dist_to_edge * math.cos(face_angle)
            my = cy - dist_to_edge * math.sin(face_angle)
            
            side_angle = face_angle + math.pi / 2
            dx = half_edge_w * math.cos(side_angle)
            dy = half_edge_w * math.sin(side_angle)
            
            p1 = QPointF(mx - dx, my + dy)
            p2 = QPointF(mx + dx, my - dy)
            edges.append((p1, p2))
            
        return edges

    def is_edge_connected(self, current_key, edge_midpoint):
        """Checks if a DIFFERENT neighbor module spans across this edge midpoint"""
        mating_threshold = self.radius * 0.90
        
        for key, info in self.modules.items():
            if key == current_key:
                continue
                
            if info['active']:
                mx = self.width() / 2 + info['pos'][0]
                my = self.height() / 2 - info['pos'][1]
                
                dist = math.hypot(edge_midpoint.x() - mx, edge_midpoint.y() - my)
                if dist < mating_threshold:
                    return True
        return False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        center_x = self.width() / 2
        center_y = self.height() / 2

        # Step 1: Draw Module Bases
        for key, info in self.modules.items():
            if not info['active']:
                continue

            x = center_x + info['pos'][0]
            y = center_y - info['pos'][1]
            
            fill_color = QColor(230, 230, 230) if info['type'] == 'inner' else QColor(215, 215, 215)
            painter.setPen(QPen(QColor(60, 60, 60), 2))
            painter.setBrush(QBrush(fill_color))
            module_poly = self.get_module_polygon(x, y, self.radius, info['angle'])
            painter.drawPolygon(module_poly)

            # Step 2: Draw Three Interior Splitting Lines (One for each side angle direction)
            for idx in range(3):
                line_angle_rad = math.radians(info['angle'] + 30 + (idx * 120))
                line_len = self.radius * 0.95
                dx = line_len * math.cos(line_angle_rad)
                dy = line_len * math.sin(line_angle_rad)
                
                p1 = QPointF(x - dx, y + dy)
                p2 = QPointF(x + dx, y - dy)
                
                state = info['line_states'][idx]
                if state == 1:
                    line_pen = QPen(QColor(0, 102, 204), 3, Qt.PenStyle.SolidLine)
                elif state == 2:
                    line_pen = QPen(QColor(138, 43, 226), 3, Qt.PenStyle.SolidLine)
                else:
                    line_pen = QPen(QColor(150, 150, 150), 1, Qt.PenStyle.DotLine)
                
                painter.setPen(line_pen)
                painter.drawLine(p1, p2)

            # Step 3: Render Delete 'X' Anchor at Center
            painter.setPen(QPen(QColor(200, 30, 30), 2))
            painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            painter.drawText(int(x - 5), int(y + 5), "X")

        # Step 4: Evaluate and Draw Unconnected Free Green Edges
        green_pen = QPen(QColor(46, 184, 46), 4, Qt.PenStyle.SolidLine)
        for key, info in self.modules.items():
            if not info['active']:
                continue
                
            x = center_x + info['pos'][0]
            y = center_y - info['pos'][1]
            edges = self.get_edge_segments(x, y, self.radius, info['angle'])
            
            for p1, p2 in edges:
                midpoint = QPointF((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2)
                
                if not self.is_edge_connected(key, midpoint):
                    painter.setPen(green_pen)
                    painter.drawLine(p1, p2)

    def mousePressEvent(self, event):
        center_x = self.width() / 2
        center_y = self.height() / 2
        click_pos = event.position()
        
        if event.button() == Qt.MouseButton.LeftButton:
            # Check 1: Did user click a central 'X' deletion anchor?
            for key, info in self.modules.items():
                if info['active']:
                    mx = center_x + info['pos'][0]
                    my = center_y - info['pos'][1]
                    if math.hypot(click_pos.x() - mx, click_pos.y() - my) <= 12:
                        info['active'] = False
                        self.update()
                        self.notify_graph_update()
                        return

            # Check 2: Did user click a green open edge to construct a new module?
            for key, info in self.modules.items():
                if not info['active']:
                    continue
                    
                mx = center_x + info['pos'][0]
                my = center_y - info['pos'][1]
                edges = self.get_edge_segments(mx, my, self.radius, info['angle'])
                
                for p1, p2 in edges:
                    midpoint = QPointF((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2)
                    
                    if not self.is_edge_connected(key, midpoint):
                        dist_to_mid = math.hypot(click_pos.x() - midpoint.x(), click_pos.y() - midpoint.y())
                        if dist_to_mid <= 12:
                            dx = midpoint.x() - mx
                            dy = midpoint.y() - my
                            angle_rad = math.atan2(-dy, dx)
                            
                            step_dist = self.radius * 1.67
                            new_vx = info['pos'][0] + step_dist * math.cos(angle_rad)
                            new_vy = info['pos'][1] + step_dist * math.sin(angle_rad)
                            
                            new_rot = info['angle'] + 180
                            new_key = f"Custom_{len(self.modules) + 1}"
                            
                            self.modules[new_key] = {
                                'pos': (new_vx, new_vy),
                                'angle': new_rot,
                                'active': True,
                                'type': 'outer',
                                'line_states': [0, 0, 0]
                            }
                            self.update()
                            self.notify_graph_update()
                            return

            # Check 3: Check which specific internal line was clicked by measuring distance to its center
            for key, info in self.modules.items():
                if not info['active']:
                    continue
                mx = center_x + info['pos'][0]
                my = center_y - info['pos'][1]
                
                # If clicking the general module bounds, check if a specific line midpoint is closest
                if math.hypot(click_pos.x() - mx, click_pos.y() - my) <= self.radius:
                    closest_line_idx = -1
                    min_dist = 99999.0
                    
                    # We mathematically find the midpoint coordinates of each of the 3 internal lines
                    for idx in range(3):
                        line_angle_rad = math.radians(info['angle'] + 30 + (idx * 120))
                        # Since the lines cut straight across the true center, their midpoints are at (mx, my)
                        # We evaluate proximity to a small 14px tracking zone along its orientation axis
                        lx = mx + 12 * math.cos(line_angle_rad + math.pi/2)
                        ly = my - 12 * math.sin(line_angle_rad + math.pi/2)
                        
                        dist_to_line = math.hypot(click_pos.x() - mx, click_pos.y() - my)
                        
                        # Project click onto the line normal to see which slice was selected
                        click_angle = math.atan2(-(click_pos.y() - my), click_pos.x() - mx)
                        angle_diff = abs(math.cos(click_angle - line_angle_rad))
                        
                        if angle_diff > 0.85 and dist_to_line < min_dist:
                            min_dist = dist_to_line
                            closest_line_idx = idx
                            
                    if closest_line_idx != -1:
                        # Cycles state ONLY for that chosen clicked line lane index
                        info['line_states'][closest_line_idx] = (info['line_states'][closest_line_idx] + 1) % 3
                        self.update()
                        self.notify_graph_update()
                        return

    def clear_all(self):
        for key in list(self.modules.keys()):
            if "Custom_" in key:
                del self.modules[key]
            else:
                self.modules[key]['active'] = False
                self.modules[key]['line_states'] = [0, 0, 0]
        self.update()
        self.notify_graph_update()

    def reset_all(self):
        self.modules.clear()
        self.build_perfect_interlock_grid()
        self.update()
        self.notify_graph_update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pattern Generator")
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        self.label = QLabel("Pattern Generator")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("font-size: 12px; font-weight: bold; color: #333; margin: 4px;")
        layout.addWidget(self.label)
        
        self.grid_canvas = AssemblyGrid()
        layout.addWidget(self.grid_canvas)

        btn_layout = QHBoxLayout()
        self.clear_btn = QPushButton("Clear Canvas")
        self.reset_btn = QPushButton("Reset Pattern")
        
        self.clear_btn.clicked.connect(self.grid_canvas.clear_all)
        self.reset_btn.clicked.connect(self.grid_canvas.reset_all)
        
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.reset_btn)

        self.save_btn = QPushButton("Save Graph")
        self.save_btn.clicked.connect(self.save_graph_to_file)
        btn_layout.addWidget(self.save_btn)

        self.load_btn = QPushButton("Load Graph")
        self.load_btn.clicked.connect(self.load_graph_from_file)
        btn_layout.addWidget(self.load_btn)

        layout.addLayout(btn_layout)

    def save_graph_to_file(self):
        graph_json = self.grid_canvas.get_graph_text()
        path, _ = QFileDialog.getSaveFileName(self, "Save Graph", "graph.json", "JSON Files (*.json);;All Files (*)")
        if path:
            with open(path, 'w', encoding='utf-8') as file:
                file.write(graph_json)

    def load_graph_from_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Graph", "", "JSON Files (*.json);;All Files (*)")
        if not path:
            return

        try:
            with open(path, 'r', encoding='utf-8') as file:
                graph_data = json.load(file)
        except Exception as exc:
            QMessageBox.warning(self, "Load Graph Failed", f"Could not read JSON file:\n{exc}")
            return

        try:
            self.grid_canvas.load_graph_data(graph_data)
        except Exception as exc:
            QMessageBox.warning(self, "Load Graph Failed", f"Invalid graph data format:\n{exc}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())