# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\dtoot\Desktop\HOME FOLDER\Python files\untitledmassspec4.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1707, 1265)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setAutoFillBackground(True)
        MainWindow.setIconSize(QtCore.QSize(30, 30))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(490, 360, 651, 401))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableWidget.sizePolicy().hasHeightForWidth())
        self.tableWidget.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.tableWidget.setFont(font)
        self.tableWidget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tableWidget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tableWidget.setWordWrap(True)
        self.tableWidget.setRowCount(10)
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setObjectName("tableWidget")
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(200)
        self.tableWidget.horizontalHeader().setHighlightSections(False)
        self.tableWidget.horizontalHeader().setMinimumSectionSize(50)
        self.tableWidget.horizontalHeader().setStretchLastSection(False)
        self.tableWidget.verticalHeader().setVisible(True)
        self.tableWidget.verticalHeader().setCascadingSectionResizes(False)
        self.tableWidget.verticalHeader().setDefaultSectionSize(35)
        self.tableWidget.verticalHeader().setMinimumSectionSize(35)
        self.tableWidget.verticalHeader().setStretchLastSection(False)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(60, 240, 321, 41))
        self.pushButton.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.pushButton.setObjectName("pushButton")
        self.spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox.setGeometry(QtCore.QRect(280, 140, 101, 31))
        self.spinBox.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.spinBox.setMaximum(6000)
        self.spinBox.setSingleStep(50)
        self.spinBox.setProperty("value", 600)
        self.spinBox.setObjectName("spinBox")
        self.spinBox_2 = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_2.setGeometry(QtCore.QRect(290, 190, 91, 31))
        self.spinBox_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.spinBox_2.setMaximum(50000)
        self.spinBox_2.setSingleStep(200)
        self.spinBox_2.setProperty("value", 2700)
        self.spinBox_2.setObjectName("spinBox_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(140, 140, 121, 31))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(30, 190, 231, 31))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(70, 310, 351, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_res = QtWidgets.QLabel(self.centralwidget)
        self.label_res.setGeometry(QtCore.QRect(410, 140, 111, 31))
        self.label_res.setText("")
        self.label_res.setObjectName("label_res")
        self.label_MRP = QtWidgets.QLabel(self.centralwidget)
        self.label_MRP.setGeometry(QtCore.QRect(400, 190, 111, 31))
        self.label_MRP.setText("")
        self.label_MRP.setObjectName("label_MRP")
        self.pushButton_start_em = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_start_em.setEnabled(False)
        self.pushButton_start_em.setGeometry(QtCore.QRect(650, 40, 291, 41))
        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.pushButton_start_em.setFont(font)
        self.pushButton_start_em.setObjectName("pushButton_start_em")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(140, 80, 111, 41))
        self.comboBox.setObjectName("comboBox")
        self.label_connect = QtWidgets.QLabel(self.centralwidget)
        self.label_connect.setGeometry(QtCore.QRect(10, 10, 471, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_connect.setFont(font)
        self.label_connect.setText("")
        self.label_connect.setObjectName("label_connect")
        self.pushButton_stop_em = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_stop_em.setEnabled(False)
        self.pushButton_stop_em.setGeometry(QtCore.QRect(800, 100, 141, 41))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_stop_em.setFont(font)
        self.pushButton_stop_em.setObjectName("pushButton_stop_em")
        self.pushButton_comports = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_comports.setGeometry(QtCore.QRect(280, 80, 101, 41))
        self.pushButton_comports.setObjectName("pushButton_comports")
        self.CollectorTableView = QtWidgets.QTableView(self.centralwidget)
        self.CollectorTableView.setGeometry(QtCore.QRect(50, 360, 431, 401))
        self.CollectorTableView.setMinimumSize(QtCore.QSize(0, 0))
        self.CollectorTableView.setFrameShape(QtWidgets.QFrame.Panel)
        self.CollectorTableView.setFrameShadow(QtWidgets.QFrame.Raised)
        self.CollectorTableView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.CollectorTableView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.CollectorTableView.setShowGrid(False)
        self.CollectorTableView.setGridStyle(QtCore.Qt.NoPen)
        self.CollectorTableView.setObjectName("CollectorTableView")
        self.CollectorTableView.horizontalHeader().setCascadingSectionResizes(False)
        self.CollectorTableView.horizontalHeader().setDefaultSectionSize(100)
        self.CollectorTableView.horizontalHeader().setMinimumSectionSize(50)
        self.CollectorTableView.verticalHeader().setDefaultSectionSize(25)
        self.CollectorTableView.verticalHeader().setStretchLastSection(False)
        self.label_com = QtWidgets.QLabel(self.centralwidget)
        self.label_com.setGeometry(QtCore.QRect(410, 80, 81, 31))
        self.label_com.setText("")
        self.label_com.setObjectName("label_com")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(480, 310, 91, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.label_mass = QtWidgets.QLabel(self.centralwidget)
        self.label_mass.setGeometry(QtCore.QRect(570, 320, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_mass.setFont(font)
        self.label_mass.setText("")
        self.label_mass.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_mass.setObjectName("label_mass")
        self.label_timer = QtWidgets.QLabel(self.centralwidget)
        self.label_timer.setGeometry(QtCore.QRect(1000, 320, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_timer.setFont(font)
        self.label_timer.setText("")
        self.label_timer.setObjectName("label_timer")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(790, 310, 211, 31))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.Peaks_tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.Peaks_tableWidget.setGeometry(QtCore.QRect(50, 770, 1091, 401))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.Peaks_tableWidget.setFont(font)
        self.Peaks_tableWidget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.Peaks_tableWidget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.Peaks_tableWidget.setRowCount(10)
        self.Peaks_tableWidget.setColumnCount(4)
        self.Peaks_tableWidget.setProperty("align", "")
        self.Peaks_tableWidget.setObjectName("Peaks_tableWidget")
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(0, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(0, 2, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(0, 3, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(1, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(1, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(1, 2, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(1, 3, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(2, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(2, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(2, 2, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(2, 3, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(3, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(3, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(3, 2, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(3, 3, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(4, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(4, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(4, 2, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(4, 3, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(5, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(5, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(5, 2, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(5, 3, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(6, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(6, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(6, 2, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(6, 3, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(7, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(7, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(7, 2, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(7, 3, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(8, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(8, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(8, 2, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(8, 3, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(9, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(9, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(9, 2, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.Peaks_tableWidget.setItem(9, 3, item)
        self.Peaks_tableWidget.horizontalHeader().setCascadingSectionResizes(True)
        self.Peaks_tableWidget.horizontalHeader().setDefaultSectionSize(265)
        self.Peaks_tableWidget.horizontalHeader().setMinimumSectionSize(100)
        self.Peaks_tableWidget.verticalHeader().setCascadingSectionResizes(True)
        self.Peaks_tableWidget.verticalHeader().setDefaultSectionSize(35)
        self.Peaks_tableWidget.verticalHeader().setMinimumSectionSize(35)
        self.pushButton_peak_gen = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_peak_gen.setGeometry(QtCore.QRect(1160, 770, 161, 51))
        self.pushButton_peak_gen.setObjectName("pushButton_peak_gen")
        self.checkBoxDirft = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBoxDirft.setGeometry(QtCore.QRect(1190, 50, 291, 31))
        self.checkBoxDirft.setObjectName("checkBoxDirft")
        self.checkBoxSourceConsump = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBoxSourceConsump.setGeometry(QtCore.QRect(1190, 100, 361, 31))
        self.checkBoxSourceConsump.setObjectName("checkBoxSourceConsump")
        self.textEditMagDrift = QtWidgets.QTextEdit(self.centralwidget)
        self.textEditMagDrift.setGeometry(QtCore.QRect(1030, 40, 140, 41))
        self.textEditMagDrift.setObjectName("textEditMagDrift")
        self.textEditSourceConsump = QtWidgets.QTextEdit(self.centralwidget)
        self.textEditSourceConsump.setGeometry(QtCore.QRect(1030, 100, 140, 41))
        self.textEditSourceConsump.setObjectName("textEditSourceConsump")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1707, 38))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "SUPER MASS SPECTREMULATOR"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Faraday A"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Faraday B"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Ion Counting"))
        self.pushButton.setText(_translate("MainWindow", "Initialise Mass Spec Optics"))
        self.label.setText(_translate("MainWindow", "Resolution"))
        self.label_2.setText(_translate("MainWindow", "Mass Resolving Power"))
        self.label_3.setText(_translate("MainWindow", "Collector Configuration"))
        self.pushButton_start_em.setText(_translate("MainWindow", "START EMULATION"))
        self.pushButton_stop_em.setText(_translate("MainWindow", "STOP"))
        self.pushButton_comports.setText(_translate("MainWindow", "COMPort"))
        self.label_4.setText(_translate("MainWindow", "Mass: "))
        self.label_5.setText(_translate("MainWindow", "Integration Time:"))
        item = self.Peaks_tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Mass"))
        item = self.Peaks_tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Ion beam signal"))
        item = self.Peaks_tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Collector channel"))
        item = self.Peaks_tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Detector Type"))
        __sortingEnabled = self.Peaks_tableWidget.isSortingEnabled()
        self.Peaks_tableWidget.setSortingEnabled(False)
        self.Peaks_tableWidget.setSortingEnabled(__sortingEnabled)
        self.pushButton_peak_gen.setText(_translate("MainWindow", "Read Peaks"))
        self.checkBoxDirft.setText(_translate("MainWindow", "Magnet drift ppm/30min"))
        self.checkBoxSourceConsump.setText(_translate("MainWindow", "Source consumption % per min"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

