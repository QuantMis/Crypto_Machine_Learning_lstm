import numpy as np 
import plotly.offline as py 
import plotly.graph_objs as go  
from scipy.signal import savgol_filter
import sys

class GenLabels(object):
    def __init__(self, data, window, polyorder=3, graph=False):
        # check for valid parameters
        try:
            if window%2 == 0: raise ValueError('window length must be odd and positive')
            if polyorder >= window: raise ValueError('polyorder must less than window')
        except ValueError as error:
            sys.exit(f'error: {error}')
        
        # load historic data from file
        self.hist = data
        self.window = window
        self.polyorder = polyorder

        # filter data and generate labels
        self.savgol = self.apply_filter(deriv=0)
        self.savgol_deriv = self.apply_filter(deriv=1)
        self.labels = self.cont_to_disc()
        if graph: self.graph()
    
    def apply_filter(self, deriv):
        # apply Savitzky-Golay filter to historical prices
        return savgol_filter(self.hist, self.window, self.polyorder, deriv=deriv)

    def cont_to_disc(self):
        # encode label as binary (up/down)
        label = []
        for value in self.savgol_deriv:
            if value >= 0: label.append(1)
            else: label.append(0)
        return np.array(label) 

    
    def graph(self):
        # graph the labels
        trace0 = go.Scatter(y=self.hist, name='Price')
        trace1 = go.Scatter(y=self.savgol, name='Filter')
        trace2 = go.Scatter(y=self.savgol_deriv, name='Derivative', yaxis='y2')
        data = [trace0, trace1, trace2]

        layout = go.Layout(
            title = 'Labels',
            yaxis = dict(
                title = 'USDT value'
            ),
            yaxis2 = dict(
                title='Derivative of Filter',
                overlaying = 'y',
                side = 'right'
            )
        )

        fig = go.Figure(data=data, layout=layout)
        py.plot(fig, filename='label.html')



