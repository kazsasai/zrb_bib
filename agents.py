import numpy as np

class Buyers(object):
    def __init__(self, num, p_max, p_min, v_min, v_max):
        self.base_price = np.random.randint(p_min, p_max+1, num)
        self.volatility = np.random.randint(v_min, v_max+1, num)
        self.success = np.zeros(num)

    def update_volatility(self):
        success = self.success * self.base_price
        success_max = np.max(success[success > 0])
        failure = (1 - self.success) * self.base_price
        failure_min = np.min(failure[failure > 0])

        success[success > 0] -= self.volatility
        success[success < failure_min] = failure_min
        failure[failure > 0] += self.volatility
        failure[failure > success_max] = success_max

        success += failure
        success[success < 2] = 2
        self.volatility = success - self.base_price
        self.base_price = success

    def bid(self):
        self.success *= 0
        return self.base_price + self.volatility * np.random.rand()


class Sellers(object):
    def __init__(self, num, p_max, p_min, v_min, v_max):
        self.base_price = np.random.randint(p_min, p_max+1, num)
        self.volatility = np.random.randint(v_min, v_max+1, num)
        self.success = np.zeros(num)

    def update_volatility(self):
        success = self.success * self.base_price
        success_min = np.min(success[success > 0])
        failure = (1 - self.success) * self.base_price
        failure_max = np.max(failure[failure > 0])

        success[success > 0] += self.volatility
        success[success < failure_max] = failure_max
        failure[failure > 0] -= self.volatility
        failure[failure > success_min] = success_min

        success += failure
        success[success < 2] = 2
        self.volatility = success - self.base_price
        self.base_price = success

    def ask(self):
        self.success *= 0
        return self.base_price
