import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from graph_visualizer import draw_graph_to_figure


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output" / "evolution_run"
SCRATCH_DIR = OUTPUT_DIR / "_scratch" / "generated_graphs"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


GRAPH_CARD_WIDTH = 340
PLOT_CARD_WIDTH = 600


class GraphCardWidget(QLabel):
    def __init__(self, title: str, image_path: Path, entry: dict, parent=None):
        super().__init__(parent)
        self.entry = entry
        self.image_path = image_path

        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            pixmap = QPixmap(200, 160)
            pixmap.fill(Qt.GlobalColor.lightGray)

        # Scale down to a fixed width, preserving aspect ratio, so two cards fit per row
        pixmap = pixmap.scaledToWidth(
            GRAPH_CARD_WIDTH, Qt.TransformationMode.SmoothTransformation
        )

        self.setPixmap(pixmap)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(title)

        # Initialize with unselected border
        self.set_selected(False)

    def set_selected(self, selected: bool) -> None:
        """Highlight the image directly using a CSS border."""
        if selected:
            self.setStyleSheet(
                "QLabel { border: 3px solid #1f77b4; border-radius: 4px; padding: 2px; }"
            )
        else:
            self.setStyleSheet(
                "QLabel { border: 1px solid transparent; padding: 4px; }"
            )

class EvolutionResultsVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Evolution Results Visualizer")
        self.resize(1400, 900)

        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        self.graphs_tab = QWidget(self)
        self.plots_tab = QWidget(self)
        self.tabs.addTab(self.graphs_tab, "Graphs")
        self.tabs.addTab(self.plots_tab, "Plots")

        self._build_graphs_tab()
        self._build_plots_tab()

        self.current_generation = None
        self.current_graph_entry = None

        self.load_generations()

    def _build_graphs_tab(self) -> None:
        layout = QVBoxLayout(self.graphs_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Generation:"))
        self.generation_combo = QComboBox()
        self.generation_combo.currentIndexChanged.connect(self.on_generation_changed)
        controls.addWidget(self.generation_combo, 1)
        controls.addStretch(1)
        layout.addLayout(controls)

        content = QHBoxLayout()
        content.setSpacing(12)

        left_panel = QWidget(self.graphs_tab)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.graph_scroll_area = QScrollArea(left_panel)
        self.graph_scroll_area.setWidgetResizable(True)
        self.graph_scroll_area.setMinimumWidth(760)
        self.graph_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.graph_cards_container = QWidget(self.graph_scroll_area)
        self.graph_cards_layout = QGridLayout(self.graph_cards_container)
        self.graph_cards_layout.setContentsMargins(0, 0, 0, 0)
        self.graph_cards_layout.setSpacing(8)
        self.graph_cards_columns = 2
        self.graph_scroll_area.setWidget(self.graph_cards_container)
        left_layout.addWidget(self.graph_scroll_area)

        content.addWidget(left_panel, 3)

        right_panel = QWidget(self.graphs_tab)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.metrics_group = QGroupBox("Metrics")
        self.metrics_group.setMinimumWidth(320)
        metrics_layout = QVBoxLayout(self.metrics_group)
        self.metrics_label = QLabel("Select a graph to see fitness metrics.")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        metrics_layout.addWidget(self.metrics_label)
        right_layout.addWidget(self.metrics_group)
        content.addWidget(right_panel, 1)
        layout.addLayout(content)

    def _build_plots_tab(self) -> None:
        layout = QVBoxLayout(self.plots_tab)
        layout.setContentsMargins(12, 12, 12, 12)

        self.plots_scroll_area = QScrollArea(self.plots_tab)
        self.plots_scroll_area.setWidgetResizable(True)
        self.plots_scroll_area.setMinimumWidth(760)
        self.plots_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        plots_container = QWidget(self.plots_scroll_area)
        self.plots_layout = QGridLayout(plots_container)
        self.plots_layout.setContentsMargins(0, 0, 0, 0)
        self.plots_layout.setSpacing(12)
        self.plots_columns = 1
        self.plots_scroll_area.setWidget(plots_container)
        layout.addWidget(self.plots_scroll_area)

    def load_generations(self) -> None:
        if not OUTPUT_DIR.exists():
            self.generation_combo.addItem("No output directory found")
            return

        generation_files = sorted(OUTPUT_DIR.glob("generation_*_population.json"))
        generations = []
        for path in generation_files:
            try:
                generation_idx = int(path.stem.split("generation_")[1].split("_population")[0])
            except (IndexError, ValueError):
                continue
            generations.append((generation_idx, path))

        generations.sort(key=lambda item: item[0])
        if not generations:
            self.generation_combo.addItem("No population files found")
            return

        for generation_idx, _ in generations:
            self.generation_combo.addItem(f"Generation {generation_idx}", generation_idx)

        self.generation_combo.setCurrentIndex(0)
        self.on_generation_changed(0)
        self.load_plot_images()

    def on_generation_changed(self, index: int) -> None:
        generation_idx = self.generation_combo.itemData(index)
        if generation_idx is None:
            return
        self.current_generation = generation_idx
        self.populate_graph_cards(generation_idx)

    def populate_graph_cards(self, generation_idx: int) -> None:
        self._clear_graph_cards()
        population_path = OUTPUT_DIR / f"generation_{generation_idx}_population.json"
        if not population_path.exists():
            self._set_metrics_text("Population file not found.")
            return

        population = self._load_population(population_path)
        if not population:
            self._set_metrics_text("No population entries found.")
            return

        sorted_entries = sorted(
            population,
            key=lambda item: self._aggregate_fitness(item["objectives"]),
            reverse=True,
        )

        for rank, entry in enumerate(sorted_entries, start=1):
            image_path = self._ensure_graph_preview(generation_idx, rank - 1, entry)
            card = GraphCardWidget(
                f"#{rank} | {self._format_objective_summary(entry['objectives'])}",
                image_path,
                entry,
            )
            card.mousePressEvent = lambda event, card=card: self.select_graph_card(card)  # type: ignore[assignment]
            row, col = divmod(rank - 1, self.graph_cards_columns)
            self.graph_cards_layout.addWidget(card, row, col)

        last_row = (len(sorted_entries) - 1) // self.graph_cards_columns + 1
        self.graph_cards_layout.setRowStretch(last_row, 1)
        self._set_metrics_text("Select a graph to view metrics.")

    def _clear_graph_cards(self) -> None:
        while self.graph_cards_layout.count():
            item = self.graph_cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def select_graph_card(self, card: GraphCardWidget) -> None:
        for index in range(self.graph_cards_layout.count()):
            widget = self.graph_cards_layout.itemAt(index).widget()
            if isinstance(widget, GraphCardWidget):
                widget.set_selected(widget is card)

        self.current_graph_entry = card.entry
        self._set_metrics_text(self._format_metrics_text(card.entry["objectives"]))

    def _set_metrics_text(self, text: str) -> None:
        self.metrics_label.setText(text)

    def _load_population(self, population_path: Path) -> List[dict]:
        with population_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload

    def _ensure_graph_preview(self, generation_idx: int, entry_idx: int, entry: dict) -> Path:
        SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
        preview_path = SCRATCH_DIR / f"generation_{generation_idx}_individual_{entry_idx}.png"
        if preview_path.exists():
            return preview_path

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis("off")
        draw_graph_to_figure(entry["graph"], figure=fig, title=f"Generation {generation_idx} Individual {entry_idx}", aspect_equal=False)
        fig.savefig(preview_path, dpi=140)
        plt.close(fig)
        return preview_path

    def load_plot_images(self) -> None:
        self._clear_plot_cards()
        if not OUTPUT_DIR.exists():
            return

        image_paths = sorted(
            path for path in OUTPUT_DIR.glob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        for index, image_path in enumerate(image_paths):
            card = self._build_plot_card(image_path)
            row, col = divmod(index, self.plots_columns)
            self.plots_layout.addWidget(card, row, col)

        last_row = (len(image_paths) - 1) // self.plots_columns + 1 if image_paths else 0
        self.plots_layout.setRowStretch(last_row, 1)

    def _clear_plot_cards(self) -> None:
        while self.plots_layout.count():
            item = self.plots_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _build_plot_card(self, image_path: Path) -> QWidget:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            pixmap = QPixmap(200, 160)
            pixmap.fill(Qt.GlobalColor.lightGray)

        pixmap = pixmap.scaledToWidth(
            PLOT_CARD_WIDTH, Qt.TransformationMode.SmoothTransformation
        )

        label = QLabel(pixmap=pixmap, parent=self.plots_tab)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    def _aggregate_fitness(self, objectives: dict) -> float:
        return sum(float(value) for value in objectives.values())

    def _format_objective_summary(self, objectives: dict) -> str:
        values = [f"{name}={value:.4f}" for name, value in objectives.items()]
        return ", ".join(values)

    def _format_metrics_text(self, objectives: dict) -> str:
        lines = ["Fitness values:"]
        for name, value in objectives.items():
            lines.append(f"{name}: {float(value):.4f}")
        return "\n".join(lines)


def main() -> None:
    app = QApplication(sys.argv)
    window = EvolutionResultsVisualizer()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
