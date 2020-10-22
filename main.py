from PySide2.QtWidgets import QApplication
from main_window import MainWindow


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    app.exec_()
