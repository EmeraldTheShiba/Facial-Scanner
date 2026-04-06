#Face Recognition Scanner GUI (Beta)

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTabWidget, QTextEdit, QTableWidget,
    QTableWidgetItem
)
from PyQt6.QtCore import Qt, QTimer

class FaceScannerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition Scanner")
        self.setGeometry(100, 100, 1100, 650)

        self.last_scan = "None"
        self.students = {}

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        #Header
        header_layout = QHBoxLayout()
        title_layout = QVBoxLayout()

        self.title_label = QLabel("Face Recognition Scanner")
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold;")

        self.version_label = QLabel("Beta")
        self.version_label.setStyleSheet("font-size: 12px; color: gray;")

        title_layout.addWidget(self.title_label)
        title_layout.addWidget(self.version_label)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()

        main_layout.addLayout(header_layout)

        #Tabs
        self.tabs = QTabWidget()

        #Camera Tab
        self.camera_tab = QTextEdit("Camera Feed Placeholder")

        #Classroom Tab (TABLE)
        self.classroom_tab = QWidget()
        classroom_layout = QVBoxLayout()

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Name", "Status", "Emotion"])

        classroom_layout.addWidget(self.table)
        self.classroom_tab.setLayout(classroom_layout)

        #Logs Tab
        self.logs_tab = QTextEdit()
        self.logs_tab.setReadOnly(True)

        #Summary Tab
        self.summary_tab = QTextEdit()
        self.summary_tab.setReadOnly(True)
        self.update_summary()

        self.tabs.addTab(self.camera_tab, "Camera")
        self.tabs.addTab(self.classroom_tab, "Classroom")
        self.tabs.addTab(self.logs_tab, "Logs")
        self.tabs.addTab(self.summary_tab, "Summary")

        main_layout.addWidget(self.tabs)

        #Bottom Section
        bottom_layout = QHBoxLayout()

        self.settings_button = QPushButton("Settings")
        bottom_layout.addWidget(self.settings_button, alignment=Qt.AlignmentFlag.AlignLeft)

        bottom_layout.addStretch()

        self.last_scan_label = QLabel(f"Last Scan: {self.last_scan}")
        bottom_layout.addWidget(self.last_scan_label, alignment=Qt.AlignmentFlag.AlignRight)

        main_layout.addLayout(bottom_layout)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        #Timer (50 min limit)
        self.session_timer = QTimer()
        self.session_timer.timeout.connect(self.clear_session_data)
        self.session_timer.start(3000000)

    #Classroom Tab
    def mark_attendance(self, name, emotion):
        status = "Present"
        self.students[name] = emotion

        self.refresh_table()
        self.update_last_scan(f"{name} - {emotion}")
        self.add_log(f"{name} detected ({emotion})")
        self.update_summary()

    def refresh_table(self):
        self.table.setRowCount(len(self.students))

        for row, (name, emotion) in enumerate(self.students.items()):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem("Present"))
            self.table.setItem(row, 2, QTableWidgetItem(emotion))

    #Log System
    def add_log(self, message):
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs_tab.append(f"[{timestamp}] {message}")

    #Summary Dashboard
    def update_summary(self):
        total = len(self.students)
        emotions = {}

        for e in self.students.values():
            emotions[e] = emotions.get(e, 0) + 1

        summary_text = f"Total Present: {total}\n\nEmotion Breakdown:\n"

        for emotion, count in emotions.items():
            summary_text += f"- {emotion}: {count}\n"

        self.summary_tab.setText(summary_text)

    #Core Functions
      #def update_last_scan(self, result):
        #self.last_scan = result
        #self.last_scan_label.setText(f"Last Scan: {self.last_scan}")

    def clear_session_data(self):
        self.logs_tab.append("[SYSTEM] Session expired. Data cleared.")
        self.students.clear()
        self.table.setRowCount(0)
        self.update_summary()
        self.last_scan = "None"
        self.last_scan_label.setText("Last Scan: None")

#Run
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceScannerGUI()
    window.show()
    
    sys.exit(app.exec())
