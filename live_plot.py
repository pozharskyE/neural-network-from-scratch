import matplotlib.pyplot as plt

class live_plot():
  def __init__(self, slice_size=10):
    self.slice_size = slice_size
    
    self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))


    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.setGeometry(fig_manager.canvas.width() - 50, 0, 1050, 1030)

    self.line1_1, = self.ax1.plot([], [], label='J_history')
    self.line1_2,  = self.ax1.plot([], [], label='J_cv_history')


    self.line2_1, = self.ax2.plot([], [], label=f'J_history (last {self.slice_size} steps)')
    self.line2_2, = self.ax2.plot([], [], label=f'J_cv_history (last {self.slice_size} steps)')


    self.line3, = self.ax3.plot([], [])
    self.line4, = self.ax4.plot([], [], c='orange')


    self.ax1.set_title('J_history and J_cv_history together')

    for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
      ax.set_xlabel('# Step')
      ax.set_ylabel('J (cost function value)')

  def update(self, J_history, J_cv_history):
    self.slice_size = len(J_history) // 10 + 5

    self.ax2.set_title(f'J_history and J_cv_history together (last {self.slice_size} steps)')
    self.ax3.set_title(f'J_history (last {self.slice_size} steps)')
    self.ax4.set_title(f'J_cv_history (last {self.slice_size} steps)')

    self.line1_1.set_data(range(len(J_history)), J_history)
    self.line1_2.set_data(range(len(J_cv_history)), J_cv_history)

    self.line2_1.set_data(range(len(J_history[-self.slice_size:])), J_history[-self.slice_size:])
    self.line2_2.set_data(range(len(J_cv_history[-self.slice_size:])), J_cv_history[-self.slice_size:])

    self.line3.set_data(range(len(J_history[-self.slice_size:])), J_history[-self.slice_size:])
    self.line4.set_data(range(len(J_cv_history[-self.slice_size:])), J_cv_history[-self.slice_size:])

    for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
      ax.relim()
      ax.autoscale_view()
    plt.pause(0.1)
