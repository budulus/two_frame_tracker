import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel,
                             QFileDialog, QHBoxLayout, QVBoxLayout, QSlider,
                             QMainWindow, QGraphicsView, QGraphicsScene, QAction,
                             QScrollBar)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPointF, pyqtSignal, QRectF
import cv2
import numpy as np
from matplotlib import cm


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

        self.reference_image_path = None
        self.deformed_image_path = None
        self.reference_image = None
        self.deformed_image = None
        self.reference_points = []
        self.deformed_points = []
        self.affine_matrix = None
        self.overlay_image = None
        self.current_opacity = 0.5

        self.setWindowTitle("Image Affine Mapping Tool")
        self.setGeometry(100, 100, 1000, 800)

        self.create_menu()
        self.init_ui()

    def create_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        load_reference_action = QAction("Load Reference Image", self)
        load_reference_action.triggered.connect(self.load_reference_image)
        file_menu.addAction(load_reference_action)

        load_deformed_action = QAction("Load Deformed Image", self)
        load_deformed_action.triggered.connect(self.load_deformed_image)
        file_menu.addAction(load_deformed_action)

        save_transformed_action = QAction("Save Transformed Image As",self)
        save_transformed_action.triggered.connect(self.save_transformed_image)
        file_menu.addAction(save_transformed_action)

    def init_ui(self):
        main_layout = QHBoxLayout()

        # Image Display Layout
        image_layout = QVBoxLayout()

        self.reference_label = QLabel("Reference Image")
        self.reference_view = ClickableGraphicsView()
        self.reference_scene = QGraphicsScene()
        self.reference_view.setScene(self.reference_scene)
        self.reference_view.point_clicked.connect(self.handle_reference_click)


        self.deformed_label = QLabel("Deformed Image")
        self.deformed_view = ClickableGraphicsView()
        self.deformed_scene = QGraphicsScene()
        self.deformed_view.setScene(self.deformed_scene)
        self.deformed_view.point_clicked.connect(self.handle_deformed_click)


        image_layout.addWidget(self.reference_label)
        image_layout.addWidget(self.reference_view)
        image_layout.addWidget(self.deformed_label)
        image_layout.addWidget(self.deformed_view)


        # Control Layout
        control_layout = QVBoxLayout()

        self.calculate_button = QPushButton("Calculate Affine Map")
        self.calculate_button.clicked.connect(self.calculate_affine_map)
        self.calculate_button.setEnabled(False)  # Initially disabled


        self.view_result_button = QPushButton("View Result")
        self.view_result_button.clicked.connect(self.view_result)
        self.view_result_button.setEnabled(False)  # Initially disabled


        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(int(self.current_opacity * 100))
        self.opacity_slider.valueChanged.connect(self.update_overlay_opacity)
        self.opacity_label = QLabel(f"Opacity: {self.current_opacity:.2f}")


        control_layout.addWidget(self.calculate_button)
        control_layout.addWidget(self.view_result_button)
        control_layout.addWidget(QLabel("Overlay Opacity:"))
        control_layout.addWidget(self.opacity_slider)
        control_layout.addWidget(self.opacity_label)


        main_layout.addLayout(image_layout)
        main_layout.addLayout(control_layout)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def load_reference_image(self):
        self.reference_image_path, _ = QFileDialog.getOpenFileName(self, "Open Reference Image", "", "Image Files (*.png *.jpg *.bmp *.tif)")
        if self.reference_image_path:
            self.reference_image = cv2.imread(self.reference_image_path, cv2.IMREAD_GRAYSCALE)
            if self.reference_image is not None:
                self.display_image(self.reference_image, self.reference_scene, self.reference_view)
                self.reference_points = [] # clear the points
                self.update_ui_state()

    def load_deformed_image(self):
        self.deformed_image_path, _ = QFileDialog.getOpenFileName(self, "Open Deformed Image", "", "Image Files (*.png *.jpg *.bmp *.tif)")
        if self.deformed_image_path:
            self.deformed_image = cv2.imread(self.deformed_image_path, cv2.IMREAD_GRAYSCALE)
            if self.deformed_image is not None:
                self.display_image(self.deformed_image, self.deformed_scene, self.deformed_view)
                self.deformed_points = [] # clear the points
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
        if self.reference_image is not None and len(self.reference_points) < 3:
            self.reference_points.append(point)
            self.draw_points(self.reference_scene, self.reference_points, "red")
            self.update_ui_state()

    def handle_deformed_click(self, point):
        if self.deformed_image is not None and len(self.deformed_points) < 3:
             self.deformed_points.append(point)
             self.draw_points(self.deformed_scene, self.deformed_points, "blue")
             self.update_ui_state()

    def draw_points(self, scene, points, color):
        scene_pen = QPen(QColor(color))
        scene_pen.setWidth(4)

        for point in points:
          scene.addEllipse(point.x()-5, point.y()-5, 10, 10, scene_pen, QColor(color))

    def update_ui_state(self):
        enable_calculate = (
            self.reference_image is not None
            and self.deformed_image is not None
            and len(self.reference_points) == 3
            and len(self.deformed_points) == 3
        )
        self.calculate_button.setEnabled(enable_calculate)

        enable_view = (
            self.affine_matrix is not None
            and self.overlay_image is not None
        )
        self.view_result_button.setEnabled(enable_view)


    def calculate_affine_map(self):
        src_pts = np.float32([[p.x(), p.y()] for p in self.reference_points])
        dst_pts = np.float32([[p.x(), p.y()] for p in self.deformed_points])
        try:
           self.affine_matrix = cv2.getAffineTransform(src_pts, dst_pts)
           self.overlay_image = self.create_overlay()
           print("Affine map calculated")
           self.update_ui_state()
        except Exception as e:
            print(f"Error calculating affine transform:{e}")

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
          self.display_result(self.overlay_image,self.deformed_image)

    def display_result(self, overlay, background):
        height, width = background.shape
        combined_image = np.zeros((height, width, 4), dtype=np.uint8)

        # Convert grayscale background to RGBA
        colormap = cm.get_cmap('gray')
        colored_background = (colormap(background/255)*255).astype(np.uint8)
        combined_image[:, :, :3] = colored_background[:,:,:3]
        combined_image[:, :, 3] = 255 # full alpha for background

        # Convert grayscale overlay to RGBA
        colored_overlay = (colormap(overlay / 255) * 255).astype(np.uint8)
        alpha_channel = np.full((height, width), int(self.current_opacity * 255), dtype=np.uint8)
        combined_image[:, :, :3] = (
             (1 - self.current_opacity) * combined_image[:, :, :3]
            +  (self.current_opacity) * colored_overlay[:, :, :3]
        ).astype(np.uint8)
        combined_image[:, :, 3] = ( (1 - self.current_opacity) * combined_image[:, :, 3]
            +  (self.current_opacity) * alpha_channel
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

        self.result_window = QMainWindow(self)
        self.result_window.setWindowTitle("Overlay View")
        self.result_window.setGeometry(100, 100, 800, 600)

        self.result_view = QGraphicsView(self.result_window)
        self.result_scene = QGraphicsScene()
        self.result_view.setScene(self.result_scene)

        self.display_result(self.overlay_image,self.deformed_image)

        self.result_window.setCentralWidget(self.result_view)
        self.result_window.show()

    def save_transformed_image(self):
        if self.overlay_image is None:
             print("No transformed image to save.")
             return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Transformed Image", "", "Image Files (*.png *.jpg *.bmp *.tif)")
        if file_path:
            try:
                cv2.imwrite(file_path, self.overlay_image)
                print(f"Transformed image saved to: {file_path}")
            except Exception as e:
                print(f"Error saving transformed image: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
