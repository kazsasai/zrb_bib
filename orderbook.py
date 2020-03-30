import numpy as np


class OrderBook(object):
    def __init__(self):
        self.market_price = 0
        self.center_price = 0
        self.mean_price = 0
        self.max_price = 0

    def transaction(self, bid, ask, priority):
        select_bid = np.where(np.random.rand(np.size(bid)) < priority)[0]
        select_bid = select_bid[np.argsort(bid[select_bid])[::-1]]
        select_ask = np.where(np.random.rand(np.size(ask)) < priority)[0]
        select_ask = select_ask[np.argsort(ask[select_ask])]
        min_size = np.min((np.size(select_bid), np.size(select_ask)))
        # print(bid[select_bid])
        # print(ask[select_ask])
        # print(min_size)
        if min_size > 0:
            deal = np.where(bid[:min_size] - ask[:min_size] >= 0)[0]
            successful_buyers = select_bid[deal]
            successful_sellers = select_ask[deal]
            deal_num = np.size(deal)
            if deal_num > 0:
                self.market_price = np.mean(np.append(bid[successful_buyers], ask[successful_sellers]))
                self.center_price = np.mean([bid[select_bid][-1], ask[select_ask][-1]])
                self.mean_price = np.mean(np.append(bid[select_bid], ask[select_ask]))
            else:
                self.market_price = 0
            return [successful_buyers, successful_sellers]
        else:
            return [[], []]

    def return_prices(self):
        return [self.market_price, self.center_price, self.mean_price]
