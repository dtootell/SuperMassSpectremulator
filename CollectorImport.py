import xml.etree.ElementTree as ET
import pandas as pd
from PyQt5.QtCore import QAbstractTableModel,Qt

collector_ID = []
collector_Type = []
collector_ResistorType = []
collector_HWChannel = []

def get_collector_table():

    root = ET.parse(r'C:\ProgramData\Isotopx\Isolinx\Collector.MSCFG')

    for collector in root.iter('ID'):
        collector_ID.append(collector.text)

    for collector in root.iter('Type'):
        collector_Type.append(collector.text)

    for collector in root.iter('ResistorType'):
        collector_ResistorType.append(collector.text)

    for collector in root.iter('HWChannel'):
        collector_HWChannel.append(collector.text)

    column_length = len(collector_Type)

    collector_df = pd.DataFrame(collector_ID,columns = ['ID'])
    collector_df['Type'] = collector_Type
    collector_df['Resistor'] = collector_ResistorType
    collector_df['Channel'] = collector_HWChannel

    print(collector_df)
    return collector_df

class pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
            if role == Qt.TextAlignmentRole:
                return Qt.AlignCenter
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None

def create_model():
    return pandasModel(get_collector_table())



