from __future__ import annotations

import unittest

from server.math_tool import add, divide, multiply, round


class MathToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_basic_operations(self) -> None:
        self.assertEqual(await add(2, 12), 14)
        self.assertEqual(await multiply(3, 4), 12)
        self.assertEqual(await divide(10, 4), 2.5)
        self.assertEqual(await round(2.555, 2), 2.56)


if __name__ == "__main__":
    unittest.main()
