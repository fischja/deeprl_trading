from collections import deque


class AccountState:
    def __init__(self, fixed=0.0, floating=0.0, market_price=0.0, units=0, long_positions=0, short_positions=None):
        if short_positions is None:
            short_positions = []

        if units >= 0 and abs(abs(units) * market_price - floating) > 1e-5:
            raise ValueError()
        if units > 0 and (long_positions == 0 or len(short_positions) > 0):
            raise ValueError()
        if units < 0 and (long_positions > 0 or len(short_positions) == 0):
            raise ValueError()
        if units == 0 and (long_positions != 0 or len(short_positions) != 0):
            raise ValueError()
        if long_positions != 0 and len(short_positions) != 0:
            raise ValueError()

        self.fixed = fixed
        self.floating = floating
        self.units = units
        self.long_positions = long_positions
        self.short_positions = short_positions
        self.market_price = market_price

    def __eq__(self, other):
        if not isinstance(other, AccountState):
            return False

        return (self.fixed == other.fixed and self.floating == other.floating and self.units == other.units and
                self.long_positions == other.long_positions and self.short_positions == other.short_positions and
                self.market_price == other.market_price)

    def total(self):
        return self.fixed + self.floating

    def alloc(self):
        if self.total() > 0.0:
            return self.floating / self.total() if self.long_positions > 0 else -self.floating / (self.total())
        else:
            return 0.0

    def update(self, market_price):
        floating = 0
        if self.long_positions > 0:
            floating = self.long_positions * market_price
        else:
            for p in self.short_positions:
                floating += p + (p - market_price)

        new_state = AccountState(fixed=self.fixed,
                                 floating=floating,
                                 units=self.units,
                                 long_positions=self.long_positions,
                                 short_positions=self.short_positions.copy(),
                                 market_price=market_price)
        return new_state

    def trade(self, units):
        if units > 0:
            return self._buy(units=units)
        else:
            return self._sell(units=abs(units))

    def _buy(self, units):
        fixed = self.fixed
        floating = self.floating
        long_positions = self.long_positions
        short_positions = deque(self.short_positions)

        # close all short positions first
        while len(short_positions) > 0 and units > 0:
            entry_price = short_positions.popleft()
            trade_value = entry_price + (entry_price - self.market_price)
            fixed += trade_value
            floating -= trade_value
            units -= 1

        # open long positions
        while units > 0 and fixed >= self.market_price:
            long_positions += 1
            fixed -= self.market_price
            floating += self.market_price
            units -= 1

        new_state = AccountState(fixed=fixed,
                                 floating=floating,
                                 units=long_positions - len(short_positions),
                                 long_positions=long_positions,
                                 short_positions=list(short_positions),
                                 market_price=self.market_price)
        return new_state

    def _sell(self, units):
        fixed = self.fixed
        floating = self.floating
        long_positions = self.long_positions
        short_positions = deque(self.short_positions)

        # close all long positions first
        while long_positions > 0 and units > 0:
            fixed += self.market_price
            floating -= self.market_price
            long_positions -= 1
            units -= 1

        # open short positions
        while units > 0 and fixed >= self.market_price:
            short_positions.append(self.market_price)
            fixed -= self.market_price
            floating += self.market_price
            units -= 1

        new_state = AccountState(fixed=fixed,
                                 floating=floating,
                                 units=long_positions - len(short_positions),
                                 long_positions=long_positions,
                                 short_positions=list(short_positions),
                                 market_price=self.market_price)
        return new_state

    def get_units_to_trade(self, target_alloc):
        if len(self.short_positions) > 0:
            if target_alloc <= self.alloc():
                return -round((abs(target_alloc) * self.total() - self.floating) / self.market_price)
            elif target_alloc >= 0:
                return round(self.total() * target_alloc / self.market_price) + abs(self.units)
            else:
                units = 0
                state = self
                while state.alloc() < target_alloc:
                    state = state._buy(1)
                    units += 1
                return units
        else:
            return round(self.total() * target_alloc / self.market_price) - self.units

    @staticmethod
    def perc_value_change(state1, state2):
        return (state2.total() - state1.total()) / state1.total() if state1.total() != 0 else 0
