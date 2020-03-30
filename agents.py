import numpy as np


class Buyers(object):
    def __init__(self, num, p_max, p_min, v_min, v_max):
        self.base_price = np.round(np.random.rand(num) * (p_max - p_min)) + p_min
        self.volatility = np.round(np.random.rand(num) * (v_max - v_min)) + v_min
        self.success = np.zeros(num)

    def update_volatility(self):
        next_price = self.base_price + (1 - 2 * self.success) * self.volatility

        if np.size(np.nonzero(self.success)) > 0:
            success_max = np.max(self.base_price * self.success)
            next_price[next_price > success_max] = success_max
        if np.size(np.nonzero(1 - self.success)) > 0:
            failure_min = np.min(self.base_price * ((1-self.success) > 0))
            next_price[next_price < failure_min] = failure_min

        self.volatility = next_price - self.base_price
        # self.volatility[self.volatility < 1] = 1
        # print(self.base_price)
        # print(self.volatility)
        self.base_price += self.volatility

    def bid(self):
        self.success *= 0
        return self.base_price

    def perturbation(self, noise):
        self.base_price *= (1 - noise * np.random.rand(self.base_price.shape[0]))


class Sellers(object):
    def __init__(self, num, p_max, p_min, v_min, v_max):
        self.base_price = np.round(np.random.rand(num) * (p_max - p_min)) + p_min
        self.volatility = np.round(np.random.rand(num) * (v_max - v_min)) + v_min
        self.success = np.zeros(num)

    def update_volatility(self):
        next_price = self.base_price + (2 * self.success - 1) * self.volatility

        if np.size(np.nonzero(self.success)) > 0:
            success_min = np.min(self.base_price * (self.success > 0))
            next_price[next_price < success_min] = success_min
        if np.size(np.nonzero(1 - self.success)) > 0:
            failure_max = np.max(self.base_price * (1 - self.success))
            next_price[next_price > failure_max] = failure_max

        self.volatility = next_price - self.base_price
        # self.volatility[self.volatility < 1] = 1
        self.base_price += self.volatility

    def ask(self):
        self.success *= 0
        return self.base_price

    def perturbation(self, noise):
        self.base_price *= (1 - noise * np.random.rand(self.base_price.shape[0]))
