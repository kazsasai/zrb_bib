import numpy as np


class OrderBook(object):
    def __init__(self):
        self.market_price = 0

    def transaction(self, bid, ask, priority):
        select_bid = np.where(np.random.rand(np.size(bid)) < priority)[0]
        select_bid = select_bid[np.argsort(bid[select_bid])[::-1]]
        select_ask = np.where(np.random.rand(np.size(ask)) < priority)[0]
        select_ask = select_ask[np.argsort(ask[select_ask])]

        deal = np.where(bid[select_bid] - ask[select_ask] >= 0)[0]
        successful_buyers = select_bid[deal]
        successful_sellers = select_ask[deal]
        deal_num = np.size(deal)
        if deal_num > 0:
            self.market_price =

        return [successful_buyers, successful_sellers]
