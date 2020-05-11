import time
import multiprocessing as mp
from visualize.standart_plotter import StandartPlotter


class Plotter():
    def __init__(self, plotter_class=StandartPlotter):
        plot_rcv, plot_send = mp.Pipe(False)
        self.plot_rcv = plot_rcv
        self.plot_send = plot_send
        self.plotter = plotter_class()
        self.process = mp.Process(target=self.plotter.start,
                                  args=(self.plot_rcv,),
                                  daemon=True)
        self.process.start()

    def send(self, cmd, data):
        assert cmd is not None, 'Command cannot be None'
        if self.process.is_alive():
            self.plot_send.send([cmd, data])
            if cmd == 'save' or cmd == 'close':
                time.sleep(2)
            return True
        else:
            return False


if __name__ == "__main__":
    import numpy as np

    plotter = Plotter()
    time.sleep(3)
    for i in range(100):
        data = {'session': np.random.randint(1, 10),
                'current_epoch': np.random.randint(1, 10),
                'total_epoch': 10,
                'current_iter': np.random.randint(1, 10000),
                'total_iter': 10000,
                'lr': np.random.random(),
                'time_cost': np.random.random() * 1000,
                'loss': [np.random.random() / ((i+1)/10) if i < 50
                         else np.random.random() * (i/50),
                         np.random.random(),
                         np.random.random(),
                         np.random.random(),
                         np.random.random()]}
        if not plotter.send('data', data):
            break
        time.sleep(0.5)
    #plotter.send('save', 'test')
    plotter.send('close', None)
    time.sleep(2)
