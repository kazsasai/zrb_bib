import numpy as np
import agents as ag
import orderbook as ob


class ZarabaModel(object):
    def __init__(self, num, priority, noise, v_min, v_max, p_min, p_max):
        self.buyer = ag.Buyers(num, p_max, p_min, v_min, v_max)
        self.seller = ag.Sellers(num, p_max + 10, p_min + 10, v_min, v_max)
        self.book = ob.OrderBook()
        self.t = 0
        self.priority = priority
        self.noise = noise

    def update(self):
        success = self.book.transaction(self.buyer.bid(), self.seller.ask(), self.priority)
        if np.size(success) > 0:
            self.buyer.success[success[0]] = 1
            self.seller.success[success[1]] = 1
        self.buyer.update_volatility()
        self.seller.update_volatility()
        self.buyer.perturbation(self.noise)
        self.seller.perturbation(self.noise)
        self.t += 1
        return [self.t, self.book.return_prices()[0], self.book.return_prices()[1], self.book.return_prices()[2]]
