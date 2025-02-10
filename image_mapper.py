import sys
import os
import shutil
from PyQt5 import uic
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel,
                             QFileDialog, QSlider,
                             QGraphicsView, QGraphicsScene, QWidget,
                             QTableWidget, QTableWidgetItem, QPushButton)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPointF, pyqtSignal
import cv2
import numpy as np
from matplotlib import colormaps


class ClickableGraphicsView(QGraphicsView):
    point_clicked = pyqtSignal(QPointF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        self.point_clicked.emit(scene_pos)
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(zoom_factor, zoom_factor)


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('image_mapper.ui', self)

        self.reference_image_path = None
        self.deformed_image_path = None
        self.reference_image = None
        self.deformed_image = None
        self.reference_points = []
        self.deformed_points = []
        self.affine_matrix = None
        self.overlay_image = None
        self.current_opacity = 0.5

        # Initialize result window, view and scene at startup
        self.result_window = None
        self.result_scene = QGraphicsScene()
        self.result_view = QGraphicsView()
        self.result_view.setScene(self.result_scene)

        self.reference_scene = QGraphicsScene()
        self.reference_view.setScene(self.reference_scene)
        self.reference_view.point_clicked.connect(self.handle_reference_click)

        self.deformed_scene = QGraphicsScene()
        self.deformed_view.setScene(self.deformed_scene)
        self.deformed_view.point_clicked.connect(self.handle_deformed_click)

        # Connect existing buttons from UI
        self.calculate_button.clicked.connect(self.calculate_affine_map)
        self.view_result_button.clicked.connect(self.view_result)
        self.opacity_slider.valueChanged.connect(self.update_overlay_opacity)
        self.save_all_button.clicked.connect(self.save_all)
        
        # Connect menu actions
        self.load_reference_action.triggered.connect(self.load_reference_image)
        self.load_deformed_action.triggered.connect(self.load_deformed_image)
        self.save_transformed_action.triggered.connect(self.save_transformed_image)

        # Connect the existing load buttons from UI
        self.load_reference_button.clicked.connect(self.load_reference_image)
        self.load_deformed_button.clicked.connect(self.load_deformed_image)

        self.update_ui_state()
        self.setup_toolbars()

    def setup_toolbars(self):
        # Toolbar for the Reference view
        self.ref_toolbar = self.addToolBar("Reference Controls")
        self.ref_toolbar.setObjectName("ref_toolbar")
        zoom_in_ref_action = self.ref_toolbar.addAction("Zoom In (Ref)")
        zoom_out_ref_action = self.ref_toolbar.addAction("Zoom Out (Ref)")
        reset_ref_action = self.ref_toolbar.addAction("Reset (Ref)")
        pan_ref_action = self.ref_toolbar.addAction("Pan Mode (Ref)")
        pan_ref_action.setCheckable(True)
        zoom_in_ref_action.triggered.connect(lambda: self.zoom_in(self.reference_view))
        zoom_out_ref_action.triggered.connect(lambda: self.zoom_out(self.reference_view))
        reset_ref_action.triggered.connect(lambda: self.reset_view(self.reference_view))
        pan_ref_action.toggled.connect(lambda checked: self.set_pan_mode(self.reference_view, checked))

        # Toolbar for the Deformed view
        self.def_toolbar = self.addToolBar("Deformed Controls")
        self.def_toolbar.setObjectName("def_toolbar")
        zoom_in_def_action = self.def_toolbar.addAction("Zoom In (Def)")
        zoom_out_def_action = self.def_toolbar.addAction("Zoom Out (Def)")
        reset_def_action = self.def_toolbar.addAction("Reset (Def)")
        pan_def_action = self.def_toolbar.addAction("Pan Mode (Def)")
        pan_def_action.setCheckable(True)
        zoom_in_def_action.triggered.connect(lambda: self.zoom_in(self.deformed_view))
        zoom_out_def_action.triggered.connect(lambda: self.zoom_out(self.deformed_view))
        reset_def_action.triggered.connect(lambda: self.reset_view(self.deformed_view))
        pan_def_action.toggled.connect(lambda checked: self.set_pan_mode(self.deformed_view, checked))

    def zoom_in(self, view):
        view.scale(1.15, 1.15)

    def zoom_out(self, view):
        view.scale(1 / 1.15, 1 / 1.15)

    def reset_view(self, view):
        view.resetTransform()
        if view.scene() is not None:
            view.fitInView(view.scene().itemsBoundingRect(), Qt.KeepAspectRatio)

    def set_pan_mode(self, view, enabled):
        if enabled:
            view.setDragMode(QGraphicsView.ScrollHandDrag)
        else:
            view.setDragMode(QGraphicsView.NoDrag)

    def load_reference_image(self):
        self.reference_image_path, _ = QFileDialog.getOpenFileName(
            self, "Open Reference Image", "", "Image Files (*.png *.jpg *.bmp *.tif)"
        )
        if self.reference_image_path:
            self.reference_image = cv2.imread(self.reference_image_path, cv2.IMREAD_GRAYSCALE)
            if self.reference_image is not None:
                self.display_image(self.reference_image, self.reference_scene, self.reference_view)
                self.reference_points = []
                self.update_ui_state()

    def load_deformed_image(self):
        self.deformed_image_path, _ = QFileDialog.getOpenFileName(
            self, "Open Deformed Image", "", "Image Files (*.png *.jpg *.bmp *.tif)"
        )
        if self.deformed_image_path:
            self.deformed_image = cv2.imread(self.deformed_image_path, cv2.IMREAD_GRAYSCALE)
            if self.deformed_image is not None:
                self.display_image(self.deformed_image, self.deformed_scene, self.deformed_view)
                self.deformed_points = []
                self.update_ui_state()

    def display_image(self, image, scene, view):
        height, width = image.shape
        qimage = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        scene.clear()
        scene.addPixmap(pixmap)
        view.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        view.setSceneRect(scene.itemsBoundingRect())
        view.update()

    def handle_reference_click(self, point):
        if self.reference_view.dragMode() == QGraphicsView.ScrollHandDrag:
            return
        if self.reference_image is not None and len(self.reference_points) < 3:
            self.reference_points.append(point)
            self.draw_points(self.reference_scene, self.reference_points, "red")
            self.update_ui_state()

    def handle_deformed_click(self, point):
        if self.deformed_view.dragMode() == QGraphicsView.ScrollHandDrag:
            return
        if self.deformed_image is not None and len(self.deformed_points) < 3:
            self.deformed_points.append(point)
            self.draw_points(self.deformed_scene, self.deformed_points, "blue")
            self.update_ui_state()

    def draw_points(self, scene, points, color):
        scene_pen = QPen(QColor(color))
        scene_pen.setWidth(4)
        for point in points:
            scene.addEllipse(point.x() - 5, point.y() - 5, 3, 2, scene_pen, QColor(color))

    def update_ui_state(self):
        enable_calculate = (
            self.reference_image is not None and
            self.deformed_image is not None and
            len(self.reference_points) == 3 and
            len(self.deformed_points) == 3
        )
        self.calculate_button.setEnabled(enable_calculate)

        enable_view = (
            self.affine_matrix is not None and
            self.overlay_image is not None
        )
        self.view_result_button.setEnabled(enable_view)

    def calculate_affine_map(self):
        src_pts = np.float32([[p.x(), p.y()] for p in self.reference_points])
        dst_pts = np.float32([[p.x(), p.y()] for p in self.deformed_points])
        try:
            self.affine_matrix = cv2.getAffineTransform(src_pts, dst_pts)
            self.overlay_image = self.create_overlay()
            self.update_affine_table()
            print("Affine map calculated")
            self.update_ui_state()
        except Exception as e:
            print(f"Error calculating affine transform: {e}")

    def update_affine_table(self):
        if self.affine_matrix is None:
            return
        A = self.affine_matrix[:, :2]
        b = self.affine_matrix[:, 2]
        for i in range(2):
            for j in range(2):
                item = QTableWidgetItem(f"{A[i, j]:.4f}")
                self.affine_table.setItem(i, j, item)
            item = QTableWidgetItem(f"{b[i]:.4f}")
            self.affine_table.setItem(i, 2, item)

    def create_overlay(self):
        if self.reference_image is None or self.affine_matrix is None or self.deformed_image is None:
            return None
        height, width = self.deformed_image.shape
        warped_reference = cv2.warpAffine(self.reference_image, self.affine_matrix, (width, height))
        return warped_reference

    def update_overlay_opacity(self, value):
        self.current_opacity = value / 100.0
        self.opacity_label.setText(f"Opacity: {self.current_opacity:.2f}")
        if self.overlay_image is not None:
            self.display_result(self.overlay_image, self.deformed_image)

    def display_result(self, overlay, background):
        height, width = background.shape
        combined_image = np.zeros((height, width, 4), dtype=np.uint8)

        colormap = colormaps['gray']
        colored_background = (colormap(background / 255) * 255).astype(np.uint8)
        combined_image[:, :, :3] = colored_background[:, :, :3]
        combined_image[:, :, 3] = 255

        colored_overlay = (colormap(overlay / 255) * 255).astype(np.uint8)
        alpha_channel = np.full((height, width), int(self.current_opacity * 255), dtype=np.uint8)
        combined_image[:, :, :3] = (
            (1 - self.current_opacity) * combined_image[:, :, :3] +
            (self.current_opacity) * colored_overlay[:, :, :3]
        ).astype(np.uint8)
        combined_image[:, :, 3] = (
            (1 - self.current_opacity) * combined_image[:, :, 3] +
            (self.current_opacity) * alpha_channel
        ).astype(np.uint8)

        qimage = QImage(combined_image.data, width, height, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimage)
        self.result_scene.clear()
        self.result_scene.addPixmap(pixmap)
        self.result_view.fitInView(self.result_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        self.result_view.setSceneRect(self.result_scene.itemsBoundingRect())
        self.result_view.update()

    def view_result(self):
        if self.overlay_image is None or self.deformed_image is None:
            print("No overlay or deformed image to show")
            return

        if self.result_window is None:
            self.result_window = QMainWindow(self)
            self.result_window.setWindowTitle("Overlay View")
            self.result_window.setGeometry(100, 100, 800, 600)
            self.result_window.setCentralWidget(self.result_view)

        self.display_result(self.overlay_image, self.deformed_image)

        self.result_window.setCentralWidget(self.result_view)
        self.result_window.show()

    def save_transformed_image(self):
        if self.overlay_image is None:
            print("No transformed image to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Transformed Image", "", "Image Files (*.png *.jpg *.bmp *.tif)"
        )
        if file_path:
            try:
                cv2.imwrite(file_path, self.overlay_image)
                print(f"Transformed image saved to: {file_path}")
            except Exception as e:
                print(f"Error saving transformed image: {e}")

    def save_all(self):
        """
        Creates a folder called "processed" in the folder where the reference image was loaded.
        If it exists, deletes it without warning, creates a new one, then:
          - Copies the original deformed image into the folder as "2.extension"
          - Saves the transformed reference image (overlay) as "1.extension"
          - Saves the affine mapping (A and b) into a text file "manual_mapping.txt"
        """
        if (self.reference_image_path is None or 
            self.deformed_image_path is None or 
            self.affine_matrix is None or 
            self.overlay_image is None):
            print("Cannot save all. Missing required images or affine mapping.")
            return

        # Folder where the reference image was loaded from
        ref_dir = os.path.dirname(self.reference_image_path)
        processed_folder = os.path.join(ref_dir, "processed")

        # If "processed" exists, remove it without warning
        if os.path.exists(processed_folder):
            shutil.rmtree(processed_folder)
        os.makedirs(processed_folder)

        # Use the deformed image file extension for naming
        _, ext = os.path.splitext(self.deformed_image_path)

        # Copy the original deformed image as "2.extension"
        dest_deformed = os.path.join(processed_folder, "2" + ext)
        shutil.copy(self.deformed_image_path, dest_deformed)

        # Save the transformed (mapped) reference image as "1.extension"
        dest_transformed = os.path.join(processed_folder, "1" + ext)
        cv2.imwrite(dest_transformed, self.overlay_image)

        # Save the affine mapping parameters into "manual_mapping.txt"
        A = self.affine_matrix[:, :2]
        b = self.affine_matrix[:, 2]
        txt_path = os.path.join(processed_folder, "manual_mapping.txt")
        with open(txt_path, "w") as f:
            for row in A:
                f.write(" ".join([f"{v:.4f}" for v in row]) + "\n")
            f.write(" ".join([f"{v:.4f}" for v in b]) + "\n")

        print("All files saved in folder:", processed_folder)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
