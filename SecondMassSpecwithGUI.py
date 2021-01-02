import serial
import sys, time
import random
import math
import numpy as np
from scipy.interpolate import UnivariateSpline
import scipy
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, QObject
import second_massspec_gui
import COM_ports
import CollectorImport
import PyQt5
import gc
import requests



class Resolution:

    def __init__(self,resolution,MRP):
        self.resolution = resolution
        self.MRP = MRP

    def delta_mass(self):
        self.delta_mass = 1.0/self.resolution
        return self.delta_mass


class Peak:
    list_of_all_peaks = []

    def __init__(self, *args,deltaM=0.1):#mass,ion_beam_intensity,collector_ch,collector_type,deltaM=0.1):
        self.list_of_all_peaks.append(self)
        self.mass = float(args[0])
        self.lowmass = self.mass - (deltaM * self.mass)
        self.highmass = self.mass + (deltaM * self.mass)
        self.ion_beam_intensity = float(args[1])
        self.collector_ch = int(args[2])
        self.collector_type = args[3]

    def drift(self,drift_rate):
        self.mass = self.mass - (self.mass*(drift_rate*(1e-6/60)))
        self.lowmass = self.mass - (deltaM * self.mass)
        self.highmass = self.mass + (deltaM * self.mass)
        return self.mass, self.lowmass, self.highmass

    def __del__(self):
        print("PEAK Instance Deleted")


def detector_array():
    # Specify Faraday amp for each channel...
    # 'F11' as 1E11 ohm resistor amp
    # 'F12' as 1E12 ohm resistor amp
    # 'ATONA' as Atona amp
    # detector_type ={'0':'F11','1':'ATONA','2':'F11','3':'F11','4':'F11','5':'F11','6':'F11','7':'F12','8':'ATONA','9':'F11'}
    # detector_type = {0: 'F11', 1: 'ATONA', 2: 'F11', 3: 'F11', 4: 'F11', 5: 'F11', 6: 'F11', 7: 'F12',8: 'ATONA', 9: 'F11'}
    detector_type = ('F11','ATONA','F11','F11','F11','F11','F11','F12','ATONA','F11')

    return detector_type

def Ion_counting_data(int_time):

    IC_noise = [0] * 8
    # ion_beam = 100000 * int_time
    # ion_count_stats = math.sqrt(ion_beam)
    for i in range(0,8,1):
        # ion_noise[i] = int(ion_beam*(i+1) + random.gauss(100,ion_count_stats/10.0))
        count_prob = random.random()
        if count_prob > 0.833333: #10 CPM or 0.167 CPS (assuming single count noise only)
            IC_noise[i] = 1

    #IC_data = 'IC:' + ','.join(str(e) for e in ion_noise)

    return IC_noise

def Faraday_baseline_noise(int_time,detector_type):

    offset = 0.0#-0.003/resistor
    kb = 1.38e-23 # J per Kelvin
    Temp = 289 # Kelvin
    J_N_noise_F11 = math.sqrt((4*kb*Temp*(1/(2*int_time)))/1e11) # Johnson-Nyquist thermal noise in Amp adjusted for Nyquist sample theorem
    J_N_noise_F12 = math.sqrt((4*kb*Temp*(1/(2*int_time)))/1e12) # Johnson-Nyquist thermal noise in Amp adjusted for Nyquist sample theorem
    ATONA_LT30 = 6.05E-17*(int_time**(-0.95)) # function generated from real noise data
    ATONA_GT30 = math.sqrt((4*kb*Temp*(1/int_time))/1e14) # Approx noise of ATONA over 30 secs
    #print 'Johnson Nyquist Noise = ', J_N_noise, ' A'
    collector_data = [0] * 10


    for i in range(0,10,1):
        if detector_type[i] == 'F11':
            collector_data[i] =  random.gauss(offset,J_N_noise_F11)
        if detector_type[i] == 'F12':
            collector_data[i] =  random.gauss(offset,J_N_noise_F12)
        if detector_type[i] == 'ATONA':
            if int_time < 30.0:
                collector_data[i] = random.gauss(offset,ATONA_LT30)
            else:
                collector_data[i] = random.gauss(offset,ATONA_GT30)

    return collector_data


def Multicollector_data(int_time,ionBeamSignal,collector_data_A,collector_data_B,B_data,B_data_ints,IC_data,detector_type,collector_ch,collector_type):

    if collector_type == 'F':
        if detector_type[collector_ch] == 'F11' or detector_type[collector_ch] == 'ATONA':

            ion_stats = math.sqrt((ionBeamSignal * 1.6e-19 * 1e11)/int_time)
            collector_data_A[collector_ch] = collector_data_A[collector_ch] + ((ionBeamSignal + random.gauss(0,ion_stats))/1e11)
            #print('A data ion stats = ',ion_stats)
            if B_data == True:
                ion_stats = math.sqrt((ionBeamSignal * 1.6e-19 * 1e11)/(int_time * B_data_ints))
                collector_data_B[collector_ch] = collector_data_B[collector_ch] + ((ionBeamSignal + random.gauss(0,ion_stats))/1e11)
                #print('B data ion stats = ', ion_stats)

        if detector_type[collector_ch] == 'F12':
            ion_stats = math.sqrt((ionBeamSignal * 1.6e-19 * 1e12)/int_time)
            collector_data_A[collector_ch] = collector_data_A[collector_ch] + ((ionBeamSignal + random.gauss(0,ion_stats))/1e12)
            if B_data == True:
                ion_stats = math.sqrt((ionBeamSignal * 1.6e-19 * 1e12)/(int_time * B_data_ints))
                collector_data_B[collector_ch] = collector_data_B[collector_ch] + ((ionBeamSignal + random.gauss(0,ion_stats))/1e12)

    if collector_type == 'M':
        #print ('Ion beam IC = ', ionBeamSignal)
        ion_stats = math.sqrt(ionBeamSignal)
        ion_beam_counts = int(ionBeamSignal + random.gauss(0,ion_stats))
        IC_data[collector_ch] = IC_data[collector_ch] + (ion_beam_counts * int_time)

    return collector_data_A,collector_data_B,IC_data


def final_data_string(collector_data_A,collector_data_B,IC_data):

    Faraday_A = 'A:' + ','.join(str(e) for e in collector_data_A) # convert list collector_data into comma separated string and add A: to front
    Faraday_AB = Faraday_A + ';' +'B:' + ','.join(str(e) for e in collector_data_B)
    Faraday_AB_IC = Faraday_AB + ';' + 'IC:' + ','.join(str(e) for e in IC_data) + '\r'

    return Faraday_AB_IC



def peak_shape_definition(requestMass,peak):
    peakSide = peak.mass / optics.MRP
    flat_top_region = (peak.mass/optics.resolution)-(2 * peakSide)

    lowMassTop = peak.mass - (flat_top_region/2.0)
    highMassTop = peak.mass + (flat_top_region/2.0)


    x1 = np.linspace((lowMassTop - peakSide),lowMassTop,9)
    y1 = (0.00, 0.05, 0.18, 0.33, 0.50, 0.69, 0.84, 0.95, 1.00)
    #y1 = (0.00,0.002473,0.017986,0.119203,0.5,0.880797,0.982014,0.997527,1.00)

    x2 = np.linspace(highMassTop,(highMassTop + peakSide),9)
    y2 = (1.00, 0.95, 0.84, 0.69, 0.50, 0.33, 0.18, 0.05, 0.00)
    #y2 = ( 1.00, 0.997527, 0.982014,0.880797,0.5,0.119203,0.017986,0.002473,0.00)


    Low_spl = UnivariateSpline(x1,y1)
    High_spl = UnivariateSpline(x2,y2)

    if requestMass >= lowMassTop and requestMass <= highMassTop:
        ionBeamSignal = 1.0
    elif requestMass < lowMassTop and requestMass > lowMassTop - peakSide:
        ionBeamSignal = Low_spl(requestMass)
    elif requestMass > highMassTop and requestMass < highMassTop + peakSide:
        ionBeamSignal = High_spl(requestMass)
    else:
        ionBeamSignal = 0.0

    return ionBeamSignal


def initialise(res,MRP):
    global optics,deltaM
    optics = Resolution(res,MRP) # resolution, Mass resolving power
    deltaM = optics.delta_mass()

    print('Mass spectrometer initialised')


    #global a_peak, b_peak,c,d
    # a_peak = Peak(37.964, deltaM, 1.27, 6, 'F')
    # b_peak = Peak(37.964, deltaM, 3.6, 7, 'F')
    # c = Peak(37.964, deltaM, 2,8, 'F')
    # d = Peak(39.9624, deltaM, 1235,3, 'M')# Ax 1e11
    # e = Peak(40.0313,deltaM,170, 3, 'M')
    # jj = Peak(3.0160293191, deltaM, 250, 1, 'M')  # Ax 1e11
    # kk = Peak(3.021927, deltaM, 800, 1, 'M')  # Ax 1e11
    # ll = Peak(3.023475, deltaM, 300, 1, 'M')
    # mm = Peak(3.998, deltaM, 5.0, 8, 'F')




    return optics,deltaM

def COMPort_connect():
    global ser

    try:
        ser = serial.Serial(port=com_port, baudrate=19200, timeout=0.1, write_timeout=0.1)
        print(com_port)
    except serial.serialutil.SerialException:
        print(com_port,"cannot connect; try different COM Port number")
    return ser

# optics, deltaM = initialise()

# detector_type = detector_array()
#
# print (detector_type)
#define peaks
#a=Peak(37.964,deltaM,1.6e-1,6,'F')#Ax 1e11
# b=Peak(37.964,deltaM,1.6e-1,7,'F')#H1 1e12
# c=Peak(37.964,deltaM,1.6e-2,8,'F')#H2 ATONA
# #d=Peak(37.990,deltaM,2.96e-2,6,'F')
# #e=Peak(37.964,deltaM,9.92e-4,3,'F')
# f=Peak(37.964,deltaM,1000000,3,'M')

#

class InitialisationMassSpec(QtWidgets.QMainWindow,second_massspec_gui.Ui_MainWindow):

    COM_check = False
    Initialise_check = False

    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon(r'C:\Users\dtoot\Pictures\peaks.png'))
        self.pushButton.clicked.connect(self.initialise_mass_spec)
        self.pushButton_start_em.clicked.connect(self.data_stream)
        self.pushButton_stop_em.clicked.connect(self.data_stream_stop)
        self.pushButton_comports.clicked.connect(self.COMPort_assign)
        self.pushButton_peak_gen.clicked.connect(self.peak_generator)
        self.comboBox.addItems(COM_ports.COM_ports())
        self.CollectorTableView.setModel(CollectorImport.create_model())
        row_cnt = self.Peaks_tableWidget.rowCount()
        coll_type = ['','Faraday','Multiplier']
        for j in range(row_cnt):
            self.Peaks_tableWidget.setItem(j, 3, QtWidgets.QTableWidgetItem())
            self.coll_comboBox = QtWidgets.QComboBox()
            self.coll_comboBox.addItems(coll_type)
            self.Peaks_tableWidget.setCellWidget(j, 3, self.coll_comboBox)

        Ar40V = 2.0318
        self.a_peak = Peak(35.964,Ar40V/298.5, 6, 'F')#,deltaM=0.1)
        # #self.b_peak = Peak(35.964, 729551, 6, 'F')#,deltaM=0.1)
        # self.c = Peak(37.964, Ar40V/1585, 6, 'F')#,deltaM=0.1)
        # self.d = Peak(39.964, Ar40V, 6, 'F')#,deltaM=0.1)
        # self.e = Peak(39.964, Ar40V/1585, 5, 'F')  # ,deltaM=0.1)
        # self.f = Peak(39.964, Ar40V/298.5, 4, 'F')  # ,deltaM=0.1)


    def initialise_mass_spec(self):

        self.label_res.setText(str(self.spinBox.value()))
        self.label_MRP.setText(str(self.spinBox_2.value()))
        self.optics,self.deltaM = initialise(self.spinBox.value(),self.spinBox_2.value())
        print("Resolution:",self.optics.resolution,"  MRP:",self.optics.MRP,"  deltaM:",self.deltaM)
        self.Initialise_check = True
        if self.COM_check == True and self.Initialise_check == True:
            self.pushButton_start_em.setEnabled(True)
            self.COM_check = False


        # k = 0
        #
        # for peak in Peak.list_of_all_peaks:
        #     ID_item = QtWidgets.QTableWidgetItem(str(peak.mass))
        #     ID_item.setTextAlignment(QtCore.Qt.AlignHCenter)
        #     ionB_item = QtWidgets.QTableWidgetItem(str(peak.ion_beam_intensity))
        #     ionB_item.setTextAlignment(QtCore.Qt.AlignHCenter)
        #     coll_ch_item = QtWidgets.QTableWidgetItem(str(peak.collector_ch))
        #     coll_ch_item.setTextAlignment(QtCore.Qt.AlignHCenter)
        #     coll_typ_item = QtWidgets.QTableWidgetItem(str(peak.collector_type))
        #     coll_typ_item.setTextAlignment(QtCore.Qt.AlignHCenter)
        #     self.tableWidget.setItem(k,0,ID_item)
        #     self.tableWidget.setItem(k,1,ionB_item)
        #     self.tableWidget.setItem(k,2,coll_ch_item)
        #     self.tableWidget.setItem(k,3,coll_typ_item)
        #     k+=1


    def display_data(self,collector_data_A,collector_data_B,IC_data,int_time,fmtRMass,cycle_time):
        k=0
        l=0
        m=0

        self.label_mass.setStyleSheet('color:blue')
        self.label_mass.setText(fmtRMass)
        self.label_timer.setStyleSheet('color:blue')
        self.label_timer.setText(str(cycle_time))

        for channel in collector_data_A:
            test_data_format = collector_data_A[k] * 1e11 # TODO link to collector config resistor value.
            formatted_data = '%7.4f' % test_data_format
            first_num = QtWidgets.QTableWidgetItem(formatted_data)
            first_num.setTextAlignment(QtCore.Qt.AlignHCenter)
            self.tableWidget.setItem(k, 0, first_num)
            k+=1

        for channel in collector_data_B:
            test_data_format = collector_data_B[l] * 1e11 # TODO link to collector config resistor value.
            formatted_data = '%7.4f' % test_data_format
            first_num = QtWidgets.QTableWidgetItem(formatted_data)
            first_num.setTextAlignment(QtCore.Qt.AlignHCenter)
            self.tableWidget.setItem(l, 1, first_num)

            l+=1

        for channel in IC_data:
            test_data_format = IC_data[m] * (1/int_time)
            formatted_data = '%7.0f' % test_data_format
            first_num = QtWidgets.QTableWidgetItem(formatted_data)
            first_num.setTextAlignment(QtCore.Qt.AlignHCenter)
            self.tableWidget.setItem(m, 2, first_num)

            m+=1

    def delete_peaks(self):
        print("....trying to delete peaks....")
        if Peak.list_of_all_peaks:
            for peak in Peak.list_of_all_peaks:
                print("*******  deleting peak  *******", peak)
                del peak
            Peak.list_of_all_peaks = []

        else:
            print("No Peaks to delete!")

    def show_peaks(self,peak_data):
        print('running show peaks')
        peak_data.append('F')
        peak_inst = Peak(*peak_data)

    def peak_generator(self):
        self.peak_workthread = PeakWorkThread()
        self.peak_workthread.peak_signal.connect(self.show_peaks)
        self.peak_workthread.peak_del_signal.connect(self.delete_peaks)
        print("peak gen button clicked")
        #self.QtWidgets.QMessageBox.information(self, 'JESUS!','Jesus!')
        self.peak_workthread.start()


    def data_stream(self):
        self.pushButton_start_em.setEnabled(False)
        self.pushButton_stop_em.setEnabled(True)
        self.label_connect.setStyleSheet('color:red')
        self.label_connect.setText("Waiting for connection...")
        self.workthread = WorkThread()
        self.workthread.data_signal.connect(self.display_data)
        print ('Waiting for connection.......Instrument control connect')
        self.workthread.start()

    def data_stream_stop(self):
        self.workthread.terminate()
        self.pushButton_start_em.setEnabled(True)
        self.pushButton_stop_em.setEnabled(False)
        print ("**** Stopped data stream *****")

    def COMPort_assign(self):
        global com_port

        com_port = str(self.comboBox.currentText())
        self.label_com.setText(com_port)
        self.COM_check = True
        self.ser = COMPort_connect()

        return self.ser


class PeakWorkThread(QtCore.QThread):

    peak_signal = pyqtSignal(list)
    peak_del_signal = pyqtSignal()

    def __init__(self):
        #super(WorkThread,self).__init__(parent)
        QtCore.QThread.__init__(self) #,parent)
        #self.ser = ser


    def __del__(self):
        self.wait()

    def run(self):

        self.peak_del_signal.emit()

        m = mass_spec_app.Peaks_tableWidget.rowCount()
        n = mass_spec_app.Peaks_tableWidget.columnCount()
        peak_data = []
        for i in range(m):
            for j in range(n):
                cell_item = mass_spec_app.Peaks_tableWidget.item(i,j)
                if cell_item != None:
                    cell_text = cell_item.text()
                    if j == (n - 1):
                        cell_text = mass_spec_app.coll_comboBox.currentText()
                    if cell_text != '':
                        peak_data.append(cell_text)
                else:
                    continue
            if peak_data:
                self.peak_signal.emit(peak_data)
            peak_data = []



class WorkThread(QtCore.QThread):

    data_signal = pyqtSignal(list,list,list,float,str,str)

    def __init__(self):
        #super(WorkThread,self).__init__(parent)
        QtCore.QThread.__init__(self) #,parent)
        #self.ser = ser


    def __del__(self):
        self.wait()

    def run(self):

        #ser = COMPort_connect()
        toc = 0.0001
        tic = 0.0
        int_time = 0.2
        t_start = time.time()

        while True:
            calc_time = (toc - tic)*1000.0
            cycle_time = "{:.2f}ms".format(calc_time)

            out = mass_spec_app.ser.read_until(b'\r', 1024)
            out = str(out.decode('utf-8'))
            out_parse = out.split(',')

            if out != '':
                #print (out)
                if out[0] == 'P':
                    int_time = int(out[1], 16) / 10.0
                    print ('Integration time = ', int_time)
                    ser.write('0\r'.encode(errors="strict"))
                    time.sleep(0.01)
                    #TODO configure a check box to determine ion beam consumption timer
                    if out[1] == 'A':
                         t_start = time.time()
                         print('Reset ION BEAM Ar40')
                    continue

                if out[0] == 'U':
                    ser.write('Isotopx Emulator\r'.encode(errors="strict"))
                    print ('Integration time = ', int_time)
                    ser.write('0\r'.encode(errors="strict"))
                    time.sleep(0.01)
                    continue

                if out[0] == 'M' or out[0] == 'T':
                    mass_spec_app.label_connect.setStyleSheet('color:green')
                    mass_spec_app.label_connect.setText("**** CONNECTED ****")
                    ser.write('0\r'.encode(errors="strict"))
                    time.sleep(0.001)
                    continue

                if out == 'X1\r':
                    mass_spec_app.label_connect.setStyleSheet('color:green')
                    mass_spec_app.label_connect.setText("**** CONNECTED ****")

                    print ('************ CONNECTED ****************')
                    ser.write('0\r'.encode(errors="strict"))
                    time.sleep(0.01)
                    continue
                # out_parse[0] = R3
                # out_parse[1] = ScanType
                # out_parse[2] = Owner
                # out_parse[3] = integration number
                # out_parse[4] = total number of integrations (-1 means infinite)
                # out_parse[5] = Magnet mass
                t_end = time.time()
                elapsed_time = t_end - t_start
                # print 'Timer (', elapsed_time, ' s')

                if out_parse[0] == 'R3':
                    #print(out)
                    tic = time.perf_counter()
                    requestMass = float(out_parse[5])
                    fmtRMass = '%7.4f' % requestMass
                    int_time_B = float(out_parse[4]) * int_time
                    # print('Int time B = ', int_time_B, 's')
                    time.sleep(int_time)
                    ionBeamSignal = 0.0
                    ionBeam_alpha = 0.0
                    collector_data_A = Faraday_baseline_noise(int_time, detector_type)
                    IC_data = Ion_counting_data(int_time)
                    if out_parse[3] == out_parse[4]:
                        collector_data_B = Faraday_baseline_noise(int_time_B, detector_type)
                        B_data = True
                        B_data_ints = float(out_parse[4])
                    else:
                        collector_data_B = [0.0] * 10
                        B_data = False
                    for peak in Peak.list_of_all_peaks:
                        # peak.drift(1)
                        # print(peak.mass)

                        if requestMass >= peak.lowmass and requestMass <= peak.highmass:
                            ionBeam_alpha = peak_shape_definition(requestMass, peak) * (
                                peak.ion_beam_intensity)  * (0.995**(elapsed_time/60.0))

                        ionBeamSignal = ionBeamSignal + ionBeam_alpha
                        ionBeam_alpha = 0.0

                        # print ('Ion beam = ', ionBeamSignal)
                        if ionBeamSignal < 0.0:
                            ionBeamSignal = 0.0
                        Multicollector_data(int_time, ionBeamSignal, collector_data_A, collector_data_B, B_data,
                                            B_data_ints, IC_data, detector_type, peak.collector_ch,
                                            peak.collector_type)
                        ionBeamSignal = 0.0


                    data_stream = final_data_string(collector_data_A, collector_data_B, IC_data)
                    #print(collector_data_A)

                    self.data_signal.emit(collector_data_A,collector_data_B,IC_data,int_time,fmtRMass,cycle_time)

                    ser.write(data_stream.encode(encoding="utf-8", errors="strict"))


                else:
                    ser.write('0\r'.encode(errors="strict"))



            time.sleep(0.010)
            out = ''
            toc=time.perf_counter()


        return



if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')

    mass_spec_app = InitialisationMassSpec()
    # p = QtGui.QPalette()
    # gradient = QtGui.QLinearGradient(0, 0, 0, 400)
    # gradient.setColorAt(0.0,QtGui.QColor(129, 139, 175))
    # gradient.setColorAt(1.0, QtGui.QColor(129, 139, 175))
    # p.setBrush(QtGui.QPalette.Window, QtGui.QBrush(gradient))
    # mass_spec_app.setPalette(p)

    detector_type = detector_array()
    # MainWindow = QtGui.QMainWindow()
    # ui = Ui_MainWindow()
    # ui.setupUi(MainWindow)
    mass_spec_app.show()


    sys.exit(app.exec_())





#C:\Python27\Lib\site-packages\PyQt4>pyuic4 -x input.ui -o output.py
#pyuic5 -x "C:\Users\dtoot\Desktop\HOME FOLDER\Python files\untitledmassspec4.ui" -o "C:\Users\dtoot\Desktop\HOME FOLDER\Python files\second_massspec_gui.py"

