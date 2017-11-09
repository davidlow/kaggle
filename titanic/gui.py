
from PyQt5 import QtWidgets
from PyQt5 import QtCore

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import matplotlib.pyplot as plt
import multiprocessing
import sys
import functools
import time
import numpy as np
from importlib import reload

class WorkerObject(QtCore.QObject):
    def __init__(self, func, parent=None):
        super(self.__class__, self).__init__(parent)
        self.func = func

    @QtCore.pyqtSlot()
    def startWork(self):
        self.func()

class Window(QtWidgets.QDialog):
    datatypes = [-1,1]
    datamarker = ['x','o']
    datacolor  = ['b', 'r']

    def __init__(self, thisapp, parent=None):
        super(Window, self).__init__(parent)

        self.figure = plt.figure()
        self.ax     = self.figure.add_subplot(111)
        self.ax.set_xlim([-10,10])
        self.ax.set_ylim([-10,10])
        self.canvas = FigureCanvas(self.figure)
        self.canvas.draw()
        self.app    = thisapp


        self.toolbar = NavigationToolbar(self.canvas, self)

        # button for clearing axes
        self.button_clear = QtWidgets.QPushButton('Clear!')
        self.button_clear.clicked.connect(self.clearaxes)

        # button for chosing datatype
        self._datatype_cntr = 0
        hlayout_datatype = QtWidgets.QHBoxLayout()
        self.button_datatype = QtWidgets.QPushButton('Setting Training data: ')
        self.button_datatype.clicked.connect(self.changedatatype)
        self.label_datatype = QtWidgets.QLabel(
                                str(self.datatypes[self._datatype_cntr]))
        self.marker_datatype = QtWidgets.QLabel(
                                str(self.datamarker[self._datatype_cntr]))
        hlayout_datatype.addWidget(self.button_datatype)
        hlayout_datatype.addWidget(self.label_datatype)
        hlayout_datatype.addWidget(self.marker_datatype)

        # Button for perceptron
        self.button_perceptron = QtWidgets.QPushButton('Perceptron!')
        self.button_perceptron.clicked.connect(self.perceptron)

        # set GUI layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.button_clear)
        layout.addWidget(self.canvas)
        layout.addLayout(hlayout_datatype)
        layout.addWidget(self.button_perceptron)
        self.setLayout(layout)

        # setup training data collection thread
        self.trainingset = []
        self.trainingpipe_gui, self.trainingpipe_worker = multiprocessing.Pipe()
        self.worker        = WorkerObject(
                                functools.partial(self.collecttrainingpts, 
                                    pipe=self.trainingpipe_worker))
        self.worker_thread = QtCore.QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.startWork)
        self.worker_thread.start()

        cid = self.figure.canvas.mpl_connect('button_press_event', 
                                             functools.partial(self.onfigclick,
                                                 pipe=self.trainingpipe_gui))
        
    def clearaxes(self):
        self.ax.cla()
        self.ax.set_xlim([-10,10])
        self.ax.set_ylim([-10,10])
        self.canvas.draw()
        self.trainingset = []

    def changedatatype(self):
        self._datatype_cntr = (self._datatype_cntr + 1) % 2
        self.label_datatype.setText(str(
                    self.datatypes[self._datatype_cntr]))
        self.marker_datatype.setText(str(
                    self.datamarker[self._datatype_cntr]))

    def collecttrainingpts(self, pipe):
        while True:
            if pipe.poll():
                (pt, d_t, d_m, d_c) = pipe.recv()
                self.trainingset.append((pt, d_t))
                try:
                    self.ax.plot(pt[0],pt[1], linestyle='', color=d_c, marker=d_m)
                except:
                    pass
                self.canvas.draw()
            time.sleep(.05)



    def onfigclick(self, event, pipe):
        # Maybe send which datapt is being selected? would need self
        pipe.send( ((event.xdata, event.ydata), 
                     self.datatypes[self._datatype_cntr], 
                     self.datamarker[self._datatype_cntr],
                     self.datacolor[self._datatype_cntr]
                     ))

    def perceptron(self):
        import perceptron
        reload(perceptron)
        trainingset = [{'x': pt[0], 'y': pt[1], 'val':v} 
                       for (pt, v) in self.trainingset]
        p = perceptron.Perceptron(trainingset, ['x', 'y'], 'val', 
                                  targetconv = {1:1, -1:-1})
        p.makeW(timelimit=5)
        xs = np.linspace(-10,10,100)
        ys = (-p.W[0] * xs - p.W[2])/p.W[1]
        self.ax.plot(xs,ys, marker='', linestyle='-', color='k')
        self.canvas.draw()



def main():
    app = QtWidgets.QApplication(sys.argv)
    main = Window(app)
    main.setWindowTitle('Machine Learning Testbed')
    main.show()

    app.exec_()

