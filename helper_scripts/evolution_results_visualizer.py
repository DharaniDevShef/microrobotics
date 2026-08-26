import json
import os
import sys
from pathlib import Path
from typing import Optional

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
_SRC_DIR = str(ROOT_DIR / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import objectives_api  # noqa: E402 - the single source of truth for "combined
# fitness", also used for the RL reward signal (moo_api.py) and now
# main.py's per-generation "Individual index" log line. _aggregate_fitness
# below calls this instead of its own sum() so all three never drift apart
# again - see scalarize()'s docstring for why raw summation is wrong once
# a "minimize" objective (like f4) stops being a placeholder zero.
OUTPUT_DIR = ROOT_DIR / "output" / "evolution_run"
SCRATCH_DIR = OUTPUT_DIR / "_scratch" / "generated_graphs"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


GRAPH_CARD_WIDTH = 340
PLOT_CARD_WIDTH = 600
LINEAGE_THUMB_WIDTH = 200
SCREENSHOT_CARD_WIDTH = 260
SCREENSHOTS_COLUMNS = 3


def generation_dir(generation_idx: int) -> Path:
    """moo_api.run_generation's per-generation work_dir (XMLs, stats.json,
    screenshots, breeding_events.json) - see main.py."""
    return OUTPUT_DIR / f"generation_{generation_idx}"


def screenshot_path(generation_idx: int, ind_idx: int) -> Path:
    """Matches roblet_simulator.run_headless's `screenshot_{model_name}.png`
    naming, where model_name is the XML's basename - moo_api.py always
    names an individual's XML `ind{i}_assembly.xml`."""
    return generation_dir(generation_idx) / f"screenshot_ind{ind_idx}_assembly.png"


def load_population_history() -> dict:
    """out_dir/population_history.json (plotting_api.append_generation_population)
    - one growing file for the whole run instead of a separate
    generation_{gen}_population.json per generation. Returns
    {generation_idx: [records...]}, {} if the run hasn't written one yet.

    Every record's objectives dict is run through
    objectives_api.migrate_legacy_objectives() here, so this visualizer
    can open an OLDER run's output folder (pre-renumbering f1..f7 objective
    names) exactly the same as a current one - _aggregate_fitness()/
    _format_metrics_text() below only ever need to know about the CURRENT
    OBJECTIVE_NAMES schema."""
    path = OUTPUT_DIR / "population_history.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        history = json.load(f)
    result = {}
    for entry in history:
        population = entry["population"]
        for record in population:
            record["objectives"] = objectives_api.migrate_legacy_objectives(record["objectives"])
        result[entry["generation"]] = population
    return result


def load_breeding_events(generation_idx: int) -> Optional[dict]:
    """The lineage log moo_api.run_generation writes via
    _write_breeding_events() - None if this generation predates the
    feature (or hasn't run yet)."""
    path = generation_dir(generation_idx) / "breeding_events.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
        self.population_history = {}

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

        self.graphs_subtabs = QTabWidget(self.graphs_tab)
        layout.addWidget(self.graphs_subtabs)

        population_subtab = QWidget()
        self.graphs_subtabs.addTab(population_subtab, "Population")
        self._build_population_subtab(population_subtab)

        screenshots_subtab = QWidget()
        self.graphs_subtabs.addTab(screenshots_subtab, "3D Screenshots")
        self._build_screenshots_subtab(screenshots_subtab)

        mutations_subtab = QWidget()
        self.graphs_subtabs.addTab(mutations_subtab, "Mutations")
        self.mutations_layout = self._build_lineage_subtab(mutations_subtab)

        crossover_subtab = QWidget()
        self.graphs_subtabs.addTab(crossover_subtab, "Crossover")
        self.crossover_layout = self._build_lineage_subtab(crossover_subtab)

    def _build_population_subtab(self, tab: QWidget) -> None:
        """The original single-generation graph-topology cards + fitness
        metrics view, now living in its own sub-tab alongside the new
        Screenshots/Mutations/Crossover ones."""
        content = QHBoxLayout(tab)
        content.setSpacing(12)

        left_panel = QWidget(tab)
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

        right_panel = QWidget(tab)
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

    def _build_screenshots_subtab(self, tab: QWidget) -> None:
        """3D Screenshots: grid of the individuals newly bred THIS
        generation only - carried-over survivors from the previous
        generation are excluded (they were already shown when they were
        new)."""
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)

        scroll = QScrollArea(tab)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget(scroll)
        self.screenshots_grid = QGridLayout(container)
        self.screenshots_grid.setContentsMargins(0, 0, 0, 0)
        self.screenshots_grid.setSpacing(10)
        scroll.setWidget(container)
        layout.addWidget(scroll)

    def _build_lineage_subtab(self, tab: QWidget) -> QVBoxLayout:
        """Shared scaffold for Mutations/Crossover: a scrollable vertical
        stack of parent(s) -> offspring(s) rows. Returns the layout rows
        get added to."""
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)

        scroll = QScrollArea(tab)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget(scroll)
        rows_layout = QVBoxLayout(container)
        rows_layout.setContentsMargins(0, 0, 0, 0)
        rows_layout.setSpacing(10)
        scroll.setWidget(container)
        layout.addWidget(scroll)
        return rows_layout

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

        self.population_history = load_population_history()
        if not self.population_history:
            self.generation_combo.addItem("No population_history.json found")
            return

        for generation_idx in sorted(self.population_history):
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
        self.populate_screenshots_tab(generation_idx)
        self.populate_mutations_tab(generation_idx)
        self.populate_crossover_tab(generation_idx)

    def populate_graph_cards(self, generation_idx: int) -> None:
        self._clear_graph_cards()
        population = self.population_history.get(generation_idx)
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

    # ------------------------------------------------------------------
    # 3D Screenshots / Mutations / Crossover sub-tabs
    # ------------------------------------------------------------------

    def populate_screenshots_tab(self, generation_idx: int) -> None:
        """Mirrors the Population tab exactly: the SAME individuals, in
        the SAME fitness-sorted order (see populate_graph_cards's
        sorted_entries) - just rendered as the real MuJoCo screenshot
        instead of a graph-topology diagram.

        Previously this only showed newly-bred offspring (filtered by
        ind_id), in a separate ad-hoc order - a different set AND a
        different order than the Population tab, so #N here didn't
        correspond to #N there. Reading the exact same population.json
        list, with the exact same sort key, guarantees both tabs agree.

        A gray placeholder means that survivor's own simulation failed
        (roblet_simulator.run_headless only saves a screenshot when
        success=True) - moo_api.run_generation's survival selection is now
        feasibility-first, so a failed individual should only ever show up
        here if there weren't enough feasible ones to fill the population.
        """
        self._clear_layout(self.screenshots_grid)
        population = self.population_history.get(generation_idx)
        if not population:
            self._add_placeholder(self.screenshots_grid, "No population entries found.")
            return

        sorted_entries = sorted(
            population, key=lambda item: self._aggregate_fitness(item["objectives"]), reverse=True
        )

        for position, entry in enumerate(sorted_entries, start=1):
            ind_id = entry.get("ind_id")
            path = screenshot_path(generation_idx, ind_id) if ind_id is not None else None
            card = self._image_card(path, width=SCREENSHOT_CARD_WIDTH, caption=f"#{position}")
            row, col = divmod(position - 1, SCREENSHOTS_COLUMNS)
            self.screenshots_grid.addWidget(card, row, col)

    def populate_mutations_tab(self, generation_idx: int) -> None:
        self._populate_lineage_tab(generation_idx, self.mutations_layout, event_type="mutation")

    def populate_crossover_tab(self, generation_idx: int) -> None:
        self._populate_lineage_tab(generation_idx, self.crossover_layout, event_type="crossover")

    def _populate_lineage_tab(self, generation_idx: int, layout: QVBoxLayout, event_type: str) -> None:
        self._clear_layout(layout)
        data = load_breeding_events(generation_idx)
        if data is None:
            self._add_placeholder(layout, "No breeding_events.json for this generation.")
            return

        events = [e for e in data.get("events", []) if e.get("type") == event_type]
        if not events:
            self._add_placeholder(layout, f"No {event_type} events recorded this generation.")
            return

        for event in events:
            row = self._build_lineage_row(
                generation_idx, event.get("parent_ids", []), event.get("child_ids", []), event.get("action", "")
            )
            layout.addWidget(row)
        layout.addStretch(1)

    def _build_lineage_row(self, generation_idx: int, parent_ids: list, child_ids: list, action_label: str) -> QWidget:
        """One parent(s) -> offspring(s) row: 1 parent for a mutation, 2
        parents for a crossover (which itself has 1 child for
        GRAFT_SUBTREE, or 2 for SWAP_SUBTREES - both render fine here,
        just with more thumbnails after the arrow). Captions are named by
        ROLE + POSITION in this row ("Parent 1", "Offspring 2", ...), not
        by the raw ind{i} batch index, which was confusing on its own."""
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        row = QHBoxLayout(frame)
        row.setSpacing(10)

        for position, parent_id in enumerate(parent_ids, start=1):
            row.addWidget(self._image_card(
                self._graph_topology_image(generation_idx, parent_id),
                width=LINEAGE_THUMB_WIDTH, caption=f"Parent {position}",
            ))

        action_taken = QLabel(f"{action_label}" if action_label else "---")
        action_taken.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(action_taken)

        for position, child_id in enumerate(child_ids, start=1):
            row.addWidget(self._image_card(
                self._graph_topology_image(generation_idx, child_id),
                width=LINEAGE_THUMB_WIDTH, caption=f"Offspring {position}",
            ))

        row.addStretch(1)
        return frame

    def _graph_topology_image(self, generation_idx: int, ind_idx: int) -> Optional[Path]:
        """Renders ind{ind_idx}'s graph topology (same graph_visualizer
        rendering the Population tab's cards use) from the per-generation
        `ind{i}_graph.json` moo_api.py writes for EVERY individual it
        evaluates - not just the survivors in
        population_history.json, which is all breeding_events.json's
        ids can otherwise point to (a bred-but-not-selected offspring, or a
        parent that didn't survive, won't be in that file). Cached inside
        that generation's own folder so it only renders once."""
        graph_json_path = generation_dir(generation_idx) / f"ind{ind_idx}_graph.json"
        if not graph_json_path.exists():
            return None

        cache_path = generation_dir(generation_idx) / "_graph_previews" / f"ind{ind_idx}.png"
        if cache_path.exists():
            return cache_path

        with graph_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.axis("off")
        draw_graph_to_figure(data, figure=fig, title=f"ind{ind_idx}", aspect_equal=False)
        fig.savefig(cache_path, dpi=140)
        plt.close(fig)
        return cache_path

    def _image_card(self, path: Optional[Path], width: int, caption: str = "") -> QWidget:
        """A thumbnail (or a gray placeholder if `path` is missing/None -
        e.g. the individual failed to build/simulate) with a caption
        underneath."""
        wrapper = QWidget()
        column = QVBoxLayout(wrapper)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)

        pixmap = QPixmap(str(path)) if path is not None and path.exists() else QPixmap()
        if pixmap.isNull():
            pixmap = QPixmap(width, int(width * 0.75))
            pixmap.fill(Qt.GlobalColor.lightGray)
        pixmap = pixmap.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)

        image_label = QLabel()
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(image_label)

        if caption:
            caption_label = QLabel(caption)
            caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            column.addWidget(caption_label)
        return wrapper

    def _clear_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _add_placeholder(self, layout, text: str) -> None:
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if isinstance(layout, QGridLayout):
            layout.addWidget(label, 0, 0)
        else:
            layout.addWidget(label)

    def select_graph_card(self, card: GraphCardWidget) -> None:
        for index in range(self.graph_cards_layout.count()):
            widget = self.graph_cards_layout.itemAt(index).widget()
            if isinstance(widget, GraphCardWidget):
                widget.set_selected(widget is card)

        self.current_graph_entry = card.entry
        self._set_metrics_text(self._format_metrics_text(card.entry["objectives"]))

    def _set_metrics_text(self, text: str) -> None:
        self.metrics_label.setText(text)

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
        """objectives_api.scalarize(), not a raw sum: it sign-corrects
        "minimize" objectives (per objectives_api.MAXIMIZE) before summing,
        so this stays correct once f3/f4/f5 stop being placeholder zeros -
        and stays identical to what main.py logs as "Individual index" and
        what moo_api.py uses for the RL reward, since all three now call
        this same function instead of each reimplementing it."""
        return objectives_api.scalarize(objectives)

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
