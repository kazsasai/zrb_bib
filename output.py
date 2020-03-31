import numpy as np
import zaraba as zr
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Simulation(object):
    def __init__(self, number):
        self.sr = np.zeros(number)
        self.ts = np.arange(-1 * number, 0, 1)
        self.model = zr.ZarabaModel(1000, 0.02, 0.0, 1, 10, 0, 5)

    def update(self, t):
        result = self.model.update()
        self.sr = np.roll(self.sr, -1)
        self.sr[np.size(self.sr) - 1] = result[1]
        self.ts = np.roll(self.ts, -1)
        self.ts[np.size(self.ts) - 1] = t
        print(str(result[0]) + ", " + str(result[1]) + ", " + str(result[2]) + ", " + str(result[3]))

    def plot(self, t):
        plt.cla()  # 現在描写されているグラフを消去
        im = plt.plot(self.ts, self.sr)  # グラフを生成
        self.update(t)


class TimeSeries(object):
    def __init__(self, t_max):
        self.sr = np.zeros(t_max)
        self.ts = np.zeros(t_max)
        self.model = zr.ZarabaModel(1000, 1., 1, 10, 0, 5)

        for t in range(t_max):
            result = self.model.update()
            self.sr[t] = result[1]
            self.ts[t] = t

        plt.plot(self.ts, self.sr)


if __name__ == '__main__':
    fig = plt.figure()
    data = Simulation(1000)
    ani = animation.FuncAnimation(fig, data.plot, interval=1)
    # ts = TimeSeries(10000)
    plt.show()
