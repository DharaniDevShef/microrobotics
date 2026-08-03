import sys
import math
import json
import collections
from pathlib import Path
import networkx as nx
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox,
                             QTabWidget)
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QPolygonF, QFont
from PyQt6.QtCore import Qt, QPointF
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from . import graph_visualizer
    from .mjcf_generator import build_assembly
except ImportError:  # pragma: no cover - direct script execution fallback
    import graph_visualizer
    from mjcf_generator import build_assembly

class AssemblyGrid(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(750, 750)
        
        self.radius = 45  
        self.modules = {}
        self.on_graph_update = None
        self.build_perfect_interlock_grid()

    def build_perfect_interlock_grid(self):
        """Calculates precise non-overlapping flat-to-flat contact layout using sequential naming."""
        center_to_inner = self.radius * 1.67
        inner_to_outer = self.radius * 1.67 
        
        count = 1
        for i in range(6):
            angle_deg = i * 60
            inner_key = f'module_{count}'
            count += 1
            pos_angle_rad = math.radians(angle_deg + 30)
            
            ix = center_to_inner * math.cos(pos_angle_rad)
            iy = center_to_inner * math.sin(pos_angle_rad)
            inner_rot = angle_deg + 90
            
            self.modules[inner_key] = {
                'pos': (ix, iy),
                'angle': inner_rot, 
                'active': True,
                'type': 'inner',
                'line_states': [0, 0, 0]  
            }
            
            outer_key = f'module_{count}'
            count += 1
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

    # =====================================================
    # Dynamic Connector Resolution Logic
    # =====================================================
    def get_connector_mapping(self, line_states):
        """
        Determines which of the 3 boundary edges gets labeled C1, C2, C3.
        - The edge parallel to the active hinge is C1.
        - The top-left edge relative to C1 is C2.
        - The top-right edge relative to C1 is C3.
        """
        active_hinge_idx = -1
        for idx, state in enumerate(line_states):
            if state in (1, 2):
                active_hinge_idx = idx
                break

        # If no active hinge, default: Edge 1 -> C1, Edge 0 -> C2, Edge 2 -> C3
        if active_hinge_idx == -1:
            return {1: 1, 0: 2, 2: 3}

        # Internal line indices:
        # 0: Angled splitting line (Top-Left)
        # 1: Horizontal splitting line (Bottom)
        # 2: Angled splitting line (Top-Right)
        if active_hinge_idx == 1:   # Hinge is bottom horizontal line
            return {0: 1, 2: 2, 1: 3}
        elif active_hinge_idx == 0: # Hinge is top-left angled line
            return {2: 1, 1: 2, 0: 3}
        else:                       # Hinge is top-right angled line (idx == 2)
            return {1: 1, 0: 2, 2: 3}

    # =====================================================
    # NetworkX Integration (Save / Load Core with BFS Renaming)
    # =====================================================

    # Numeric encoding of module_type, kept alongside the human-readable
    # string so the saved JSON can be fed straight into ML tooling
    # (torch_geometric.utils.from_networkx, DGL, pymoo decision vectors, ...)
    # without a separate string->id lookup step.
    MODULE_TYPE_IDS = {"non-foldable": 0, "Mountain fold": 1, "valley fold": 2}

    def get_networkx_graph(self) -> nx.DiGraph:
        """Constructs a directed NetworkX DiGraph with tree-sorted naming.

        Edges are oriented outward from module_1 (BFS parent -> child), so
        the saved graph is a rooted hierarchy rather than an arbitrary
        undirected mating map. Every non-root node also carries explicit
        `parent`/`depth` attributes describing that same hierarchy, which
        downstream consumers (GNNs, graph transformers, pymoo encodings)
        can use directly instead of re-deriving it via BFS.
        """
        raw_G = nx.Graph()
        
        # 1. Add active modules as temporary nodes
        for key, info in self.modules.items():
            if not info['active']:
                continue
            
            # Save: Blue (state 1) -> Mountain fold, Violet (state 2) -> valley fold
            module_type = "non-foldable"
            for state in info['line_states']:
                if state == 1:
                    module_type = "Mountain fold"
                    break
                elif state == 2:
                    module_type = "valley fold"
                    break
                    
            raw_G.add_node(
                key,
                module_type=module_type,
                connectors={1: None, 2: None, 3: None},
                hinge_angle=0,
                _pos=info['pos'],
                _angle=info['angle'],
                _type=info['type'],
                _line_states=info['line_states']
            )

        # 2. Map mating contacts with edge threshold checks
        active_nodes = list(raw_G.nodes())
        mating_threshold = self.radius * 0.90
        
        for i, u in enumerate(active_nodes):
            u_pos = self.modules[u]['pos']
            cx = self.width() / 2 + u_pos[0]
            cy = self.height() / 2 - u_pos[1]
            edges_u = self.get_edge_segments(cx, cy, self.radius, self.modules[u]['angle'])
            port_map_u = self.get_connector_mapping(self.modules[u]['line_states'])
            
            for j, v in enumerate(active_nodes):
                if i >= j:
                    continue
                
                v_pos = self.modules[v]['pos']
                mx_v = self.width() / 2 + v_pos[0]
                my_v = self.height() / 2 - v_pos[1]
                port_map_v = self.get_connector_mapping(self.modules[v]['line_states'])
                
                for edge_idx, (p1, p2) in enumerate(edges_u):
                    midpoint = QPointF((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2)
                    dist = math.hypot(midpoint.x() - mx_v, midpoint.y() - my_v)
                    
                    if dist < mating_threshold:
                        connector_u = port_map_u.get(edge_idx, 1)
                        edges_v = self.get_edge_segments(mx_v, my_v, self.radius, self.modules[v]['angle'])
                        connector_v = 1
                        
                        for ev_idx, (vp1, vp2) in enumerate(edges_v):
                            v_mid = QPointF((vp1.x() + vp2.x()) / 2, (vp1.y() + vp2.y()) / 2)
                            if math.hypot(midpoint.x() - v_mid.x(), midpoint.y() - v_mid.y()) < 5.0:
                                connector_v = port_map_v.get(ev_idx, 1)
                                break
                                
                        raw_G.add_edge(u, v, connector1=connector_u, connector2=connector_v)
                        raw_G.nodes[u]["connectors"][connector_u] = v
                        raw_G.nodes[v]["connectors"][connector_v] = u

        # 3. Perform BFS traversal from module_1 to rename modules sequentially
        # based on tree distance, and simultaneously record each node's
        # hierarchy depth/parent -- this same BFS order also fixes the
        # direction of every edge in the final directed graph (step 4).
        ordered_nodes = []
        depth = {}
        parent = {}
        visited = set()

        # Start BFS from the base root node
        start_node = "module_1"
        if start_node in raw_G:
            queue = collections.deque([start_node])
            visited.add(start_node)
            depth[start_node] = 0
            parent[start_node] = None
            while queue:
                curr = queue.popleft()
                ordered_nodes.append(curr)
                # Sort neighbors to keep traversal deterministic
                neighbors = sorted(list(raw_G.neighbors(curr)))
                for n in neighbors:
                    if n not in visited:
                        visited.add(n)
                        depth[n] = depth[curr] + 1
                        parent[n] = curr
                        queue.append(n)

        # Any remaining disconnected nodes become additional hierarchy roots
        for node in raw_G.nodes():
            if node not in visited:
                depth[node] = 0
                parent[node] = None
                ordered_nodes.append(node)
                visited.add(node)

        # Build systematic rename map starting from module_1
        mapping = {old_name: f"module_{idx + 1}" for idx, old_name in enumerate(ordered_nodes)}
        # BFS visit order, keyed by the *original* node name -- used below to
        # decide which end of a mating edge is upstream (closer to module_1).
        order_index = {old_name: idx for idx, old_name in enumerate(ordered_nodes)}

        # 4. Build the final directed graph. Every node gets renamed and
        # annotated with its hierarchy position; every mating edge points
        # from whichever endpoint the module_1 BFS reached first to the one
        # it reached later, so tree edges become parent->child and any
        # ring-closing edges still point consistently "outward".
        G = nx.DiGraph()
        for old_name in ordered_nodes:
            new_name = mapping[old_name]
            attrs = dict(raw_G.nodes[old_name])
            conns = attrs.get("connectors", {})
            attrs["connectors"] = {
                c_idx: (mapping[nbr] if nbr in mapping else nbr) for c_idx, nbr in conns.items()
            }
            attrs["depth"] = depth[old_name]
            attrs["parent"] = mapping[parent[old_name]] if parent[old_name] is not None else None
            attrs["type_id"] = self.MODULE_TYPE_IDS.get(attrs.get("module_type"), 0)
            G.add_node(new_name, **attrs)

        for u, v, edata in raw_G.edges(data=True):
            if order_index[u] < order_index[v]:
                src, dst = u, v
                conn_src, conn_dst = edata["connector1"], edata["connector2"]
            else:
                src, dst = v, u
                conn_src, conn_dst = edata["connector2"], edata["connector1"]
            G.add_edge(mapping[src], mapping[dst], connector1=conn_src, connector2=conn_dst)

        return G

    def get_graph_text(self) -> str:
        G = self.get_networkx_graph()
        data = nx.node_link_data(G, edges="edges")
        return json.dumps(data, indent=2)

    def load_graph_data(self, graph_data: dict):
        """Reconstructs the interactive PyQt6 canvas elements from NetworkX graph data."""
        G = nx.node_link_graph(graph_data, edges="edges")
        self.modules.clear()
        
        for node, attrs in G.nodes(data=True):
            pos = attrs.get('_pos', (0.0, 0.0))
            angle = attrs.get('_angle', 0.0)
            module_type = attrs.get('_type', 'outer')
            line_states = attrs.get('_line_states', [0, 0, 0])
            
            # Fallback mapper in case loading structural JSON without design states
            if '_line_states' not in attrs:
                m_type = attrs.get('module_type', 'non-foldable')
                if m_type == 'Mountain fold':
                    line_states = [1, 0, 0]
                elif m_type == 'valley fold':
                    line_states = [2, 0, 0]
                else:
                    line_states = [0, 0, 0]
            
            self.modules[node] = {
                'pos': tuple(pos),
                'angle': angle,
                'active': True,
                'type': module_type,
                'line_states': list(line_states)
            }
            
        self.update()
        self.notify_graph_update()

    def notify_graph_update(self):
        if self.on_graph_update:
            self.on_graph_update(self.get_graph_text())

    # =====================================================
    # Visual Painting & Click Events
    # =====================================================
    def get_module_polygon(self, cx, cy, r, orientation_deg):
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
            # Retain standard mapping indexing: [0: Top-Left, 1: Bottom, 2: Top-Right]
            edges.append((p1, p2))
            
        return edges

    def is_edge_connected(self, current_key, edge_midpoint):
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

        # Draw Module Bases
        for key, info in self.modules.items():
            if not info['active']:
                continue

            x = center_x + info['pos'][0]
            y = center_y - info['pos'][1]
            
            # Base color layout
            if key == "module_1":
                # Distinct soft gold color for the protected Base node
                fill_color = QColor(255, 210, 130)
            else:
                fill_color = QColor(230, 230, 230) if info['type'] == 'inner' else QColor(215, 215, 215)
                
            painter.setPen(QPen(QColor(60, 60, 60), 2))
            painter.setBrush(QBrush(fill_color))
            module_poly = self.get_module_polygon(x, y, self.radius, info['angle'])
            painter.drawPolygon(module_poly)

            # Draw Interior Line States
            for idx in range(3):
                line_angle_rad = math.radians(info['angle'] + 30 + (idx * 120))
                line_len = self.radius * 0.95
                dx = line_len * math.cos(line_angle_rad)
                dy = line_len * math.sin(line_angle_rad)
                
                p1 = QPointF(x - dx, y + dy)
                p2 = QPointF(x + dx, y - dy)
                
                state = info['line_states'][idx]
                if state == 1:
                    line_pen = QPen(QColor(0, 102, 204), 3, Qt.PenStyle.SolidLine)  # Mountain Fold (Blue)
                elif state == 2:
                    line_pen = QPen(QColor(138, 43, 226), 3, Qt.PenStyle.SolidLine) # Valley Fold (Violet)
                else:
                    line_pen = QPen(QColor(150, 150, 150), 1, Qt.PenStyle.DotLine)
                
                painter.setPen(line_pen)
                painter.drawLine(p1, p2)

            # Draw the module number clearly at the center of each module body
            painter.setPen(QPen(QColor(20, 20, 20)))
            painter.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            module_number = key.replace("module_", "")
            painter.drawText(int(x - 10), int(y + 6), module_number)

            # Draw connector labels distinctly around the module body
            edges = self.get_edge_segments(x, y, self.radius, info['angle'])
            port_map = self.get_connector_mapping(info['line_states'])
            # Invert mapping to find physical segment matching each connector label
            inv_port_map = {label: phys_idx for phys_idx, label in port_map.items()}
            
            painter.setPen(QPen(QColor(40, 40, 40)))
            painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            for label in [1, 2, 3]:
                phys_idx = inv_port_map.get(label)
                if phys_idx is not None:
                    p1, p2 = edges[phys_idx]
                    mid_x = (p1.x() + p2.x()) / 2
                    mid_y = (p1.y() + p2.y()) / 2
                    
                    # Compute vector from edge midpoint to module center, move labels inside body
                    dx = x - mid_x
                    dy = y - mid_y
                    dist = math.hypot(dx, dy)
                    if dist > 0:
                        lbl_x = mid_x + (dx / dist) * 15
                        lbl_y = mid_y + (dy / dist) * 15
                    else:
                        lbl_x, lbl_y = mid_x, mid_y
                        
                    painter.drawText(int(lbl_x - 5), int(lbl_y + 5), str(label))

            # Delete Anchor (Hidden on the core base node module_1)
            if key != "module_1":
                painter.setPen(QPen(QColor(200, 30, 30), 2))
                painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                painter.drawText(int(x - 5), int(y + 5), "X")

        # Draw Free Green Edges
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
            # Check 1: Deletion (Ensure module_1 cannot be deleted)
            for key, info in self.modules.items():
                if info['active'] and key != "module_1":
                    mx = center_x + info['pos'][0]
                    my = center_y - info['pos'][1]
                    if math.hypot(click_pos.x() - mx, click_pos.y() - my) <= 12:
                        info['active'] = False
                        self.update()
                        self.notify_graph_update()
                        return

            # Check 2: Add New Module via Green Edge
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
                            new_key = f"module_{len(self.modules) + 1}"
                            
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

            # Check 3: Cycle Internal Line State (Single Hinge Enforcement)
            for key, info in self.modules.items():
                if not info['active']:
                    continue
                mx = center_x + info['pos'][0]
                my = center_y - info['pos'][1]
                
                if math.hypot(click_pos.x() - mx, click_pos.y() - my) <= self.radius:
                    closest_line_idx = -1
                    min_dist = 99999.0
                    
                    for idx in range(3):
                        line_angle_rad = math.radians(info['angle'] + 30 + (idx * 120))
                        dist_to_line = math.hypot(click_pos.x() - mx, click_pos.y() - my)
                        click_angle = math.atan2(-(click_pos.y() - my), click_pos.x() - mx)
                        angle_diff = abs(math.cos(click_angle - line_angle_rad))
                        
                        if angle_diff > 0.85 and dist_to_line < min_dist:
                            min_dist = dist_to_line
                            closest_line_idx = idx
                            
                    if closest_line_idx != -1:
                        next_state = (info['line_states'][closest_line_idx] + 1) % 3
                        
                        # Reset all lines to enforce "single hinge maximum"
                        info['line_states'] = [0, 0, 0]
                        info['line_states'][closest_line_idx] = next_state
                        
                        self.update()
                        self.notify_graph_update()
                        return

    def clear_all(self):
        # Clear everything except our root module_1 base
        for key in list(self.modules.keys()):
            if key == "module_1":
                self.modules[key]['active'] = True
                self.modules[key]['line_states'] = [0, 0, 0]
            elif "module_" in key and int(key.split("_")[1]) > 12:
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


class GraphPreviewWidget(QWidget):
    def __init__(self, source_canvas, parent=None):
        super().__init__(parent)
        self.source_canvas = source_canvas
        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.refresh_graph()

    def refresh_graph(self):
        self.figure.clear()
        graph = self.source_canvas.get_networkx_graph()
        graph_visualizer.draw_graph_to_figure(graph, self.figure)
        self.canvas.draw_idle()


def save_json(graph_text, path="../graphs/assembly_graph.json"):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(graph_text)
    return path


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pattern Generator")
        self.script_dir = Path(__file__).resolve().parent
        self.default_graph_path = (self.script_dir / ".." / "graphs" / "assembly_graph.json").resolve()
        self.default_xml_path = (self.script_dir / ".." / "models" / "assembly.xml").resolve()
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        self.label = QLabel("Pattern Generator")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("font-size: 12px; font-weight: bold; color: #333; margin: 4px;")
        layout.addWidget(self.label)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        editor_tab = QWidget()
        editor_layout = QVBoxLayout(editor_tab)
        self.grid_canvas = AssemblyGrid()
        self.grid_canvas.on_graph_update = self.refresh_graph_view
        editor_layout.addWidget(self.grid_canvas)

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

        self.save_xml_btn = QPushButton("Save XML")
        self.save_xml_btn.clicked.connect(self.save_graph_and_build_xml)
        btn_layout.addWidget(self.save_xml_btn)

        editor_layout.addLayout(btn_layout)
        self.tabs.addTab(editor_tab, "Pattern Editor")

        self.graph_view = GraphPreviewWidget(self.grid_canvas)
        self.tabs.addTab(self.graph_view, "Graph View")

    def refresh_graph_view(self, _=None):
        if hasattr(self, "graph_view"):
            self.graph_view.refresh_graph()

    def save_graph_to_file(self):
        graph_json = self.grid_canvas.get_graph_text()
        path, _ = QFileDialog.getSaveFileName(self, "Save Graph", "../graphs/graph.json", "JSON Files (*.json);;All Files (*)")
        if path:
            save_json(graph_json, path)

    def save_graph_and_build_xml(self):
        graph_json = self.grid_canvas.get_graph_text()
        graph_data = json.loads(graph_json)
        self.grid_canvas.load_graph_data(graph_data)
        save_json(graph_json, str(self.default_graph_path))
        try:
            build_assembly(str(self.default_graph_path), str(self.default_xml_path))
            QMessageBox.information(self, "Export Complete", f"Saved graph to {self.default_graph_path}\nSaved XML to {self.default_xml_path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export Failed", f"Could not build XML:\n{exc}")

    def load_graph_from_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Graph",
            "../graphs",
            "JSON Files (*.json);;All Files (*)",
        )
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