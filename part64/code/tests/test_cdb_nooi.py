import unittest
import ctypes
from code.world_web.c_double_buffer_backend import _get_engine, _CDB_MAX_PRESENCE_SLOTS


class TestCDBNooiIntegration(unittest.TestCase):
    def test_nooi_update(self):
        engine = _get_engine(count=100, seed=123)

        # Create a dummy field: 64*64*8*2 floats
        size = 64 * 64 * 8 * 2
        dummy_field = [0.1] * size

        # Test update
        try:
            engine.update_nooi(dummy_field)
        except Exception as e:
            self.fail(f"update_nooi raised exception: {e}")

        # Verify snapshot still works
        result = engine.snapshot()
        self.assertEqual(result[0], 100)  # count

        engine.close()


if __name__ == "__main__":
    unittest.main()
