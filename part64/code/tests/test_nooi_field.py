import unittest
from array import array
from code.world_web.nooi import NooiField, NOOI_GRID_COLS, NOOI_GRID_ROWS, NOOI_LAYERS


class TestNooiField(unittest.TestCase):
    def test_initialization(self):
        field = NooiField()
        self.assertEqual(field.cols, NOOI_GRID_COLS)
        self.assertEqual(field.rows, NOOI_GRID_ROWS)
        self.assertEqual(len(field.layers), NOOI_LAYERS)
        self.assertEqual(len(field.layers[0]), NOOI_GRID_COLS * NOOI_GRID_ROWS * 2)

    def test_deposit_and_snapshot(self):
        field = NooiField(cols=10, rows=10)
        # Deposit a vector at center (0.5, 0.5) moving right (1, 0)
        field.deposit(0.5, 0.5, 1.0, 0.0)

        snapshot = field.get_grid_snapshot()
        self.assertEqual(snapshot["cols"], 10)
        self.assertEqual(snapshot["rows"], 10)
        self.assertTrue(len(snapshot["cells"]) > 0)

        # Check center cell
        center_cell = None
        for cell in snapshot["cells"]:
            if cell["col"] == 5 and cell["row"] == 5:
                center_cell = cell
                break

        self.assertIsNotNone(center_cell)
        if center_cell:
            self.assertGreater(center_cell["vx"], 0.0)
            self.assertAlmostEqual(center_cell["vy"], 0.0, places=4)

    def test_decay(self):
        field = NooiField(cols=10, rows=10)
        field.deposit(0.5, 0.5, 1.0, 0.0)

        snapshot1 = field.get_grid_snapshot()
        mag1 = 0.0
        for cell in snapshot1["cells"]:
            if cell["col"] == 5 and cell["row"] == 5:
                mag1 = cell["vector_magnitude"]
                break

        # Decay for 1 second
        field.decay(1.0)

        snapshot2 = field.get_grid_snapshot()
        mag2 = 0.0
        for cell in snapshot2["cells"]:
            if cell["col"] == 5 and cell["row"] == 5:
                mag2 = cell["vector_magnitude"]
                break

        self.assertLess(mag2, mag1)

    def test_sample_vector_returns_aggregated_direction(self):
        field = NooiField(cols=10, rows=10)
        for _ in range(4):
            field.deposit(0.4, 0.4, 1.0, 0.0)
        vx, vy = field.sample_vector(0.4, 0.4)
        self.assertGreater(vx, 0.0)
        self.assertAlmostEqual(vy, 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
