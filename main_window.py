from PySide2.QtCore import Qt, Slot, QUrl
from PySide2.QtGui import QStandardItemModel
from PySide2.QtWidgets import QMainWindow, QTableWidget, QWidget, QAction, QMessageBox, QDockWidget, QPushButton, QLabel, QDateTimeEdit, QTabelView
from libs import folium_offline as folium

from webview import HtmlView


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle("雷电定位Demo")

        # Menu
        self.menu = self.menuBar()
        self.fileMenu = self.menu.addMenu("File")
        self.editMenu = self.menu.addMenu("Edit")
        self.selectMenu = self.menu.addMenu("Select")
        self.runMenu = self.menu.addMenu("Run")
        self.helpMenu = self.menu.addMenu("Help")

        # Action
        # runAction = QAction("Run", self)
        # runAction.triggered.connect(self.updateData)
        # self.runMenu.addAction(runAction)  # TODO mod or del
        # self.runMenu.addSeparator()
        manualAction = QAction("Manual", self)
        manualAction.triggered.connect(self.manual)
        self.helpMenu.addAction(manualAction)

        # CentralWidget
        self.webview = HtmlView()
        self.setCentralWidget(self.webview)

        # RightDockWidget
        dockWidget = QDockWidget()
        self.addDockWidget(Qt.RightDockWidgetArea, dockWidget)

        begLabel = QLabel("起始时间", dockWidget)
        begLabel.setGeometry(0, 60, 110, 22)
        endLabel = QLabel("终止时间", dockWidget)
        endLabel.setGeometry(0, 90, 110, 22)
        self.begTimeEdit = QDateTimeEdit(dockWidget)
        self.begTimeEdit.setCalendarPopup(True)
        self.begTimeEdit.setGeometry(90, 60, 180, 22)
        self.endTimeEdit = QDateTimeEdit(dockWidget)
        self.endTimeEdit.setCalendarPopup(True)
        self.endTimeEdit.setGeometry(90, 90, 180, 22)
        selButton = QPushButton("筛选", dockWidget)
        selButton.setGeometry(170, 120, 100, 22)
        selButton.connect(self.updateData)

        self.tableView = QTabelView(dockWidget)
        self.itemModel = QStandardItemModel()


        # Status Bar
        self.statusbar = self.statusBar()
        self.statusbar.showMessage("运行中")

    @Slot()
    def updateData(self):
        folium_map = folium.Map(location=[28, 104], zoom_start = 8)
        latitudes = [30, 32, 34]
        longitudes = [100, 102, 102]

        labels = [[1,23324529], [2,23424389], [3, 123423]]

        for lat, lng, label in zip(latitudes, longitudes, labels):
            folium.Marker([lat, lng], popup=label).add_to(folium_map)
        # folium_map.add_child(folium.ClickForMarker())
        # folium_map.add_child(folium.LatLngPopup())
        folium_map.save("./static/map.html")
        _url = QUrl.fromLocalFile(r"./static/map.html")
        print(_url)
        self.webview.load("./static/map.html")

    @Slot()
    def manual(self):
        QMessageBox.about(self, "使用说明", "这是使用说明")
