from problems.neighborhood import NeighborhoodProblem
from problems.construction import IndependenceSystemProblem
import numpy as np

NUM_BOX_COLS = 4


class RectanglePackingProblem(NeighborhoodProblem, IndependenceSystemProblem):
    def __init__(self, box_length, num_rects, w_min, w_max, h_min, h_max, **kwargs):
        super(RectanglePackingProblem, self).__init__(is_max=False, **kwargs)
        self.box_length = box_length
        self.num_rects = num_rects
        self.rectangles = None
        self.__generate(box_length, num_rects, w_min, w_max, h_min, h_max)

    def __generate(self, box_length, num_rects, w_min, w_max, h_min, h_max):
        """Generates a new problem instance.

        :param box_length: L
        :param num_rects: number of rectangles
        :param w_min: minimum rectangle width
        :param w_max: maximum rectangle width
        :param h_min: minimum rectangle height
        :param h_max: maximum rectangle height
        """

        assert w_min <= box_length and w_max <= box_length
        assert h_min <= box_length and h_max <= box_length

        # Generate rectangles with uniformly random side lengths (width and height)
        widths = np.random.randint(w_min, w_max, size=num_rects)
        heights = np.random.randint(h_min, h_max, size=num_rects)
        self.rectangles = np.stack([widths, heights], axis=1)

    def rotate_rect(self, idx):
        """Rotates the rectangle with index idx."""
        rect = self.rectangles[idx]
        self.rectangles[idx] = [rect[1], rect[0]]

    def f(self, x):
        """Returns the number of boxes occupied in the current solution.
        Warning: Only defined for feasible solutions! Unfeasible solutions will yield
        invalid values."""
        locations, _ = x
        box_coords = locations // self.box_length
        return len(set(tuple(map(tuple, box_coords))))

    def is_feasible(self, x):
        # Collect rectangle properties
        locations, rotations = x
        rects = self.rectangles.copy()

        # Consider all rotations
        rects[rotations, 0] = self.rectangles[rotations, 1]
        rects[rotations, 1] = self.rectangles[rotations, 0]

        # ---- First, check that all coordinates are non-negative ----
        if np.any(locations < 0):
            return False

        # ---- Second, check for any box violation (each rect must lie inside a single box) ----
        locations_rel = locations % self.box_length
        rect_extensions = locations_rel + rects
        if np.any(rect_extensions > self.box_length):
            return False

        # ---- Third, check that no two rects intersect ----
        begins = locations
        ends = locations + rects

        # Construct virtual grid world
        x_max = np.max(ends[:, 0])
        y_max = np.max(ends[:, 1])
        grid = np.zeros((x_max, y_max), dtype=np.bool)

        # Place each single rect into the grid world
        for begin, end in zip(begins, ends):
            region_to_place = grid[begin[0]:end[0], begin[1]:end[1]]

            # Check if region to place is already occupied
            if np.any(region_to_place):
                # There is already some other rect occupying the region to place
                return False
            else:
                # Place rect into grid world
                region_to_place[:] = 1

        return True

    def get_arbitrary_solution(self):
        """Returns a solution where each rectangle is placed into an own box (not rotated)."""
        x_locations = np.arange(self.num_rects) % NUM_BOX_COLS
        y_locations = np.arange(self.num_rects) // NUM_BOX_COLS
        locations = np.stack([x_locations, y_locations], axis=1) * self.box_length
        rotations = np.zeros(self.num_rects, dtype=np.bool)
        return locations, rotations

    def get_neighborhood(self, x):
        raise NotImplementedError

    def construct(self, x, i):
        raise NotImplementedError

    def get_elements(self):
        pass

    def c(self, e):
        pass

    def is_independent(self, x):
        pass
