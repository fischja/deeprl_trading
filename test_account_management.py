import unittest
from account_management import AccountState


class TestAccountState(unittest.TestCase):
    def test_bad_ctor(self):
        self.assertRaises(ValueError, lambda: AccountState(floating=2.0))
        self.assertRaises(ValueError, lambda: AccountState(units=2.0))
        self.assertRaises(ValueError, lambda: AccountState(long_positions=2))
        self.assertRaises(ValueError, lambda: AccountState(long_positions=-2))
        self.assertRaises(ValueError, lambda: AccountState(short_positions=[2]))
        self.assertRaises(ValueError, lambda: AccountState(long_positions=2, short_positions=[2]))
        self.assertRaises(ValueError, lambda: AccountState(floating=10, long_positions=1))
        self.assertRaises(ValueError, lambda: AccountState(floating=10, long_positions=1, market_price=5))
        self.assertRaises(ValueError, lambda: AccountState(floating=10, units=-1, long_positions=1, market_price=10))
        self.assertRaises(ValueError, lambda: AccountState(floating=10, units=1, long_positions=0, market_price=10))
        self.assertRaises(ValueError, lambda: AccountState(floating=10, units=2, long_positions=0, market_price=10))

    def test_total(self):
        a = AccountState()
        self.assertEquals(a.total(), 0.0)

        a = AccountState(fixed=100)
        self.assertEquals(a.total(), 100)

        a = AccountState(floating=10, units=1, long_positions=1, market_price=10)
        self.assertEquals(a.total(), 10)

        a = AccountState(fixed=1, floating=10, units=1, long_positions=1, market_price=10)
        self.assertEquals(a.total(), 11)

    def test_alloc(self):
        a = AccountState()
        self.assertEquals(a.alloc(), 0.0)

        a = AccountState(floating=10, units=1, long_positions=1, market_price=10)
        self.assertEquals(a.alloc(), 1.0)

        a = AccountState(fixed=90, floating=10, units=1, long_positions=1, market_price=10)
        self.assertEquals(a.alloc(), 0.1)

    def test_eq(self):
        a = AccountState()
        b = AccountState()
        self.assertTrue(a == b)

        a = AccountState(fixed=90, floating=10, units=1, long_positions=1, market_price=10)
        b = AccountState(fixed=90, floating=10, units=1, long_positions=1, market_price=10)
        self.assertTrue(a == b)

        a = AccountState(fixed=90, floating=10, units=1, long_positions=1, market_price=10)
        b = AccountState(fixed=91, floating=10, units=1, long_positions=1, market_price=10)
        self.assertFalse(a == b)

    def test_update(self):
        a = AccountState(floating=10, units=1, long_positions=1, market_price=10)
        b = a.update(market_price=a.market_price)
        self.assertTrue(a == b)

        a = AccountState(floating=10, units=1, long_positions=1, market_price=10)
        b = a.update(market_price=11)
        self.assertFalse(a == b)
        self.assertEquals(b.floating, 11)

    def test_example(self):
        pre = AccountState(fixed=100, floating=0, units=0, market_price=10, long_positions=0)
        self.assertEquals(pre.total(), 100)
        self.assertEquals(pre.alloc(), 0)

        units = pre.get_units_to_trade(target_alloc=0.12)

        post = pre.trade(units=units)
        self.assertEquals(post.fixed, 90)
        self.assertEquals(post.floating, 10)
        self.assertEquals(post.units, 1)
        self.assertEquals(post.long_positions, 1)
        self.assertEquals(post.short_positions, pre.short_positions)
        self.assertEquals(post.market_price, pre.market_price)
        self.assertEquals(post.total(), pre.total())
        self.assertEquals(post.alloc(), 0.1)

        pre = post.update(market_price=14)
        self.assertEquals(pre.total(), 104)
        self.assertEquals(pre.market_price, 14)

        units = pre.get_units_to_trade(target_alloc=0.21)
        self.assertEquals(units, 1)

        post = pre.trade(units=units)
        self.assertEquals(post.fixed, 76)
        self.assertEquals(post.floating, 28)
        self.assertEquals(post.units, 2)
        self.assertEquals(post.long_positions, 2)
        self.assertEquals(post.short_positions, pre.short_positions)
        self.assertEquals(post.market_price, pre.market_price)
        self.assertEquals(post.total(), pre.total())
        self.assertAlmostEquals(post.alloc(), 0.27, places=2)

        pre = post.update(market_price=13)
        self.assertEquals(pre.total(), 102)
        self.assertEquals(pre.market_price, 13)

        units = pre.get_units_to_trade(target_alloc=0.40)
        self.assertEquals(units, 1)

        post = pre.trade(units=units)
        self.assertEquals(post.fixed, 63)
        self.assertEquals(post.floating, 39)
        self.assertEquals(post.units, 3)
        self.assertEquals(post.long_positions, 3)
        self.assertEquals(post.short_positions, pre.short_positions)
        self.assertEquals(post.market_price, pre.market_price)
        self.assertEquals(post.total(), pre.total())
        self.assertAlmostEquals(post.alloc(), 0.38, places=2)

        pre = post.update(market_price=15)
        self.assertEquals(pre.total(), 108)
        self.assertEquals(pre.market_price, 15)

        units = pre.get_units_to_trade(target_alloc=0.20)
        self.assertEquals(units, -2)

        post = pre.trade(units=units)
        self.assertEquals(post.fixed, 93)
        self.assertEquals(post.floating, 15)
        self.assertEquals(post.units, 1)
        self.assertEquals(post.long_positions, 1)
        self.assertEquals(post.short_positions, pre.short_positions)
        self.assertEquals(post.market_price, pre.market_price)
        self.assertEquals(post.total(), pre.total())
        self.assertAlmostEquals(post.alloc(), 0.14, places=2)

        pre = post.update(market_price=14)
        self.assertEquals(pre.total(), 107)
        self.assertEquals(pre.market_price, 14)

        units = pre.get_units_to_trade(target_alloc=-0.1)
        self.assertEquals(units, -2)

        post = pre.trade(units=units)
        self.assertEquals(post.fixed, 93)
        self.assertEquals(post.floating, 14)
        self.assertEquals(post.units, -1)
        self.assertEquals(post.long_positions, 0)
        self.assertEquals(post.short_positions, [14])
        self.assertEquals(post.market_price, pre.market_price)
        self.assertEquals(post.total(), pre.total())
        self.assertAlmostEquals(post.alloc(), -0.13, places=2)

        pre = post.update(market_price=12)
        self.assertEquals(pre.total(), 109)
        self.assertEquals(pre.market_price, 12)

        units = pre.get_units_to_trade(target_alloc=-0.5)
        self.assertEquals(units, -3)

        post = pre.trade(units=units)
        self.assertEquals(post.fixed, 57)
        self.assertEquals(post.floating, 52)
        self.assertEquals(post.units, -4)
        self.assertEquals(post.long_positions, 0)
        self.assertEquals(post.short_positions, [14, 12, 12, 12])
        self.assertEquals(post.market_price, pre.market_price)
        self.assertEquals(post.total(), pre.total())
        self.assertAlmostEquals(post.alloc(), -0.48, places=2)

        pre = post.update(market_price=10)
        self.assertEquals(pre.total(), 117)
        self.assertEquals(pre.market_price, 10)

        units = pre.get_units_to_trade(target_alloc=-0.54)
        self.assertEquals(units, 0)

        post = pre.trade(units=units)
        self.assertEquals(post.fixed, 57)
        self.assertEquals(post.floating, 60)
        self.assertEquals(post.units, -4)
        self.assertEquals(post.long_positions, 0)
        self.assertEquals(post.short_positions, [14, 12, 12, 12])
        self.assertEquals(post.market_price, pre.market_price)
        self.assertEquals(post.total(), pre.total())
        self.assertAlmostEquals(post.alloc(), -0.51, places=2)

        pre = post.update(market_price=21)
        self.assertEquals(pre.total(), 73)
        self.assertEquals(pre.market_price, 21)

        units = pre.get_units_to_trade(target_alloc=-0.1)
        self.assertEquals(units, 2)

        post = pre.trade(units=units)
        self.assertEquals(post.fixed, 67)
        self.assertEquals(post.floating, 6)
        self.assertEquals(post.units, -2)
        self.assertEquals(post.long_positions, 0)
        self.assertEquals(post.short_positions, [12, 12])
        self.assertEquals(post.market_price, pre.market_price)
        self.assertEquals(post.total(), pre.total())
        self.assertAlmostEquals(post.alloc(), -0.08, places=2)

        pre = post.update(market_price=18)
        self.assertEquals(pre.total(), 79)
        self.assertEquals(pre.market_price, 18)

        units = pre.get_units_to_trade(target_alloc=0.32)
        self.assertEquals(units, 3)

        post = pre.trade(units=units)
        self.assertEquals(post.fixed, 61)
        self.assertEquals(post.floating, 18)
        self.assertEquals(post.units, 1)
        self.assertEquals(post.long_positions, 1)
        self.assertEquals(post.short_positions, [])
        self.assertEquals(post.market_price, pre.market_price)
        self.assertEquals(post.total(), pre.total())
        self.assertAlmostEquals(post.alloc(), 0.23, places=2)

        pre = post.update(market_price=5)
        self.assertEquals(pre.total(), 66)
        self.assertEquals(pre.market_price, 5)

        units = pre.get_units_to_trade(target_alloc=-1.3)
        self.assertEquals(units, -18)

        post = pre.trade(units=units)
        self.assertEquals(post.fixed, 1)
        self.assertEquals(post.floating, 65)
        self.assertEquals(post.units, -13)
        self.assertEquals(post.long_positions, 0)
        self.assertEquals(post.short_positions, [5] * 13)
        self.assertEquals(post.market_price, pre.market_price)
        self.assertEquals(post.total(), pre.total())
        self.assertAlmostEquals(post.alloc(), -0.98, places=2)

        pre = post.update(market_price=4)
        self.assertEquals(pre.total(), 79)
        self.assertEquals(pre.market_price, 4)

        units = pre.get_units_to_trade(target_alloc=1.5)
        self.assertEquals(units, 43)
        post = pre.trade(units=units)
        self.assertEquals(post.fixed, 3)
        self.assertEquals(post.floating, 76)
        self.assertEquals(post.units, 19)
        self.assertEquals(post.long_positions, 19)
        self.assertEquals(post.short_positions, [])
        self.assertEquals(post.market_price, pre.market_price)
        self.assertEquals(post.total(), pre.total())
        self.assertAlmostEquals(post.alloc(), 0.96, places=2)
