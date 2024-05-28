import sys
from PyQt5.QtWidgets import QApplication
from gui import MainWindow


def main(captcha_data):
    app = QApplication(sys.argv)
    window = MainWindow(captcha_data)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    captcha_data = {"name": "website_name", "ramdom_str": True}
    main(captcha_data)
