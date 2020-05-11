import numpy as np
import matplotlib.pyplot as plt


class StandartPlotter():
    def __init__(self, figsize=(12, 7.5), grid_size_y=7):
        self.figsize = figsize
        assert (grid_size_y - 1) % 2 == 0 and grid_size_y >= 3, \
            'Grid size of Y axis must be odd number'
        self.grid_size_y = grid_size_y

        self.x = []
        self.step = -1

        self.main_y = []
        self.rpn_cls_y = []
        self.rpn_bbox_y = []
        self.rcnn_cls_y = []
        self.rcnn_bbox_y = []

    def start(self, pipe):
        self.pipe = pipe
        self.fig = plt.figure(figsize=self.figsize)
        self.grid_spec = self.fig.add_gridspec(self.grid_size_y, 4)
        _half_grid = int((self.grid_size_y - 1) / 2) + 1

        self.ax_info = self.fig.add_subplot(self.grid_spec[0, :])
        self.ax_main_loss = self.fig.add_subplot(self.grid_spec[1:, :2])
        self.ax_rpn_cls = self.fig.add_subplot(self.grid_spec[1:_half_grid, 2])
        self.ax_rpn_bbox = self.fig.add_subplot(self.grid_spec[1:_half_grid, 3])
        self.ax_rcnn_cls = self.fig.add_subplot(self.grid_spec[_half_grid:, 2])
        self.ax_rcnn_bbox = self.fig.add_subplot(self.grid_spec[_half_grid:, 3])

        self._prepare_axes()
        self.grid_spec.tight_layout(self.fig)
        self._init_plotter(show=True)

    def _init_plotter(self, show=False):
        self.cid = self.fig.canvas.mpl_connect('close_event', self._reopen_fig)
        self.fig.canvas.set_window_title('Faster R-CNN Network train process')
        self.timer = self.fig.canvas.new_timer(interval=1000)
        self.timer.add_callback(self._callback)
        self.timer.start()
        if show:
            plt.show()

    def _callback(self):
        while self.pipe.poll():
            msg = self.pipe.recv()
            if msg[0] == 'data':
                self._update_data(msg[1])
            elif msg[0] == 'save':
                self._save_figure(msg[1])
            elif msg[0] == 'close':
                self._terminate()
            else:
                raise ValueError('Cannot recognize command: "%s".' % (msg[0]))

        self.fig.canvas.draw()

    def _update_data(self, data):
        self.step += 1
        self.x.append(self.step)
        self.main_y.append(data['loss'][0])
        self.rpn_cls_y.append(data['loss'][1])
        self.rpn_bbox_y.append(data['loss'][2])
        self.rcnn_cls_y.append(data['loss'][3])
        self.rcnn_bbox_y.append(data['loss'][4])

        self._prepare_axes()

        if len(self.x) > 4:
            r = int(len(self.x) / 2)
            p = np.poly1d(np.polyfit(self.x[-r:], self.main_y[-r:], 1))
            hx = np.array([self.x[0], self.x[-1]])
            hy = np.array([p(self.x[-1]), p(self.x[-1])])
            self.ax_main_loss.plot(self.x, self.main_y, 'r-',
                                   self.x[-r:], p(self.x[-r:]), 'b-',
                                   hx, hy, '--k')
        else:
            self.ax_main_loss.plot(self.x, self.main_y, 'r-')
        self._update_value(self.ax_main_loss, self.main_y[-1], 16)

        self.ax_rpn_cls.plot(self.x, self.rpn_cls_y, 'r-')
        self._update_value(self.ax_rpn_cls, self.rpn_cls_y[-1], 12)

        self.ax_rpn_bbox.plot(self.x, self.rpn_bbox_y, 'r-')
        self._update_value(self.ax_rpn_bbox, self.rpn_bbox_y[-1], 12)

        self.ax_rcnn_cls.plot(self.x, self.rcnn_cls_y, 'r-')
        self._update_value(self.ax_rcnn_cls, self.rcnn_cls_y[-1], 12)

        self.ax_rcnn_bbox.plot(self.x, self.rcnn_bbox_y, 'r-')
        self._update_value(self.ax_rcnn_bbox, self.rcnn_bbox_y[-1], 12)

        text_size = (self.figsize[1] / self.grid_size_y) * 16
        self.ax_info.text(1, 33, 'Session: {:2d}'.format(
            data['session']), fontsize=text_size)
        self.ax_info.text(30, 33, 'Epoch: {:2d}/{:2d}'.format(
            data['current_epoch'], data['total_epoch']), fontsize=text_size)
        self.ax_info.text(62, 33, 'Iteration: {:4d}/{:4d}'.format(
            data['current_iter'], data['total_iter']), fontsize=text_size)
        self.ax_info.text(120, 33, 'LR: {:.2e}'.format(
            data['lr']), fontsize=text_size)
        self.ax_info.text(158, 33, 'Time cost: {:4.2f}s'.format(
            data['time_cost']), fontsize=text_size)

    def _save_figure(self, path):
        plt.savefig(path)

    def _terminate(self):
        self.fig.canvas.mpl_disconnect(self.cid)
        self.timer.stop()
        plt.close(self.fig)

    def _update_value(self, ax, data, text_size):
        ax.text(0.98, 0.98, '{:.4f}'.format(data),
                fontsize=text_size,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)

    def _reopen_fig(self, event):
        dummy = plt.figure(figsize=self.figsize)
        new_manager = dummy.canvas.manager
        new_manager.canvas.figure = self.fig
        self.fig.set_canvas(new_manager.canvas)
        plt.pause(0.2)
        self._init_plotter()

    def _prepare_axes(self):
        self.ax_main_loss.cla()
        self.ax_main_loss.set_title('Main loss of network')

        self.ax_rpn_cls.cla()
        self.ax_rpn_cls.set_title('RPN class loss')

        self.ax_rpn_bbox.cla()
        self.ax_rpn_bbox.set_title('RPN bbox loss')

        self.ax_rcnn_cls.cla()
        self.ax_rcnn_cls.set_title('RCNN class loss')

        self.ax_rcnn_bbox.cla()
        self.ax_rcnn_bbox.set_title('RCNN bbox loss')

        self.ax_info.cla()
        self.ax_info.set_xlim(0, 200)
        self.ax_info.set_ylim(0, 100)
        self.ax_info.set_xticks([])
        self.ax_info.set_yticks([])
