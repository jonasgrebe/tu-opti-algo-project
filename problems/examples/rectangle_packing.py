from problems.neighborhood import NeighborhoodProblem
from problems.construction import IndependenceSystemProblem
import numpy as np

NUM_BOX_COLS = 4


class RectanglePackingProblem(NeighborhoodProblem, IndependenceSystemProblem):
    def __init__(self, box_length, num_rects, w_min, w_max, h_min, h_max,
                 neighborhood_relation="geometry_based", **kwargs):
        super(RectanglePackingProblem, self).__init__(is_max=False, **kwargs)
        self.box_length = box_length
        self.num_rects = num_rects
        self.neighborhood_relation = neighborhood_relation
        self.__generate(box_length, num_rects, w_min, w_max, h_min, h_max)

        # Compute the lower bound for the minimum
        oversize = self.box_length // 2
        top_dog = np.all(self.sizes > oversize, axis=1)
        num_top_dogs = np.sum(top_dog)  # each top dog rectangle requires an own box (no two top dogs in one rectangle)
        min_box_required = np.ceil(np.sum(self.sizes[:, 0] * self.sizes[:, 1]) / self.box_length ** 2)
        self.minimum_lower_bound = max(min_box_required, num_top_dogs)

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
        self.sizes = np.stack([widths, heights], axis=1)

    def rotate_rect(self, idx):
        """Rotates the rectangle with index idx."""
        rect = self.sizes[idx]
        self.sizes[idx] = [rect[1], rect[0]]

    def f(self, x):
        """Returns the number of boxes occupied in the current solution.
        Warning: Only defined for feasible solutions! Unfeasible solutions will yield
        invalid values."""
        return len(self.get_occupied_boxes(x))

    def is_feasible(self, x):
        # Collect rectangle properties
        locations, rotations = x
        sizes = self.sizes.copy()

        # Consider all rotations
        sizes[rotations, 0] = self.sizes[rotations, 1]
        sizes[rotations, 1] = self.sizes[rotations, 0]

        # ---- First, check for any box violation (each rect must lie inside a single box) ----
        locations_rel = locations % self.box_length
        ends_rel = locations_rel + sizes
        if np.any(ends_rel > self.box_length):
            return False

        # ---- Second, check that no two rects intersect ----
        begins = locations.copy()
        ends = locations + sizes

        # Construct virtual grid world
        x_min = np.min(begins[:, 0])
        y_min = np.min(begins[:, 1])
        x_max = np.max(ends[:, 0])
        y_max = np.max(ends[:, 1])
        grid = np.zeros((x_max - x_min, y_max - y_min), dtype=np.bool)

        begins -= np.array([x_min, y_min])
        ends -= np.array([x_min, y_min])

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
        num_cols = int(np.ceil(np.sqrt(self.num_rects)))
        x_locations = np.arange(self.num_rects) % num_cols
        y_locations = np.arange(self.num_rects) // num_cols
        locations = np.stack([x_locations, y_locations], axis=1) * self.box_length
        rotations = np.zeros(self.num_rects, dtype=np.bool)
        return locations, rotations

    def get_neighborhood(self, x):
        if self.neighborhood_relation == "geometry_based":
            return self.__get_geometry_based_neighborhood(x)
        else:
            raise NotImplementedError

    def __get_geometry_based_neighborhood(self, x):
        # TODO: define place() method for another geometry based neighborhood relation
        neighbors = []

        locations, rotations = x

        # Rotations
        for rect_idx in range(self.num_rects):
            rotations_mod = rotations.copy()
            rotations_mod[rect_idx] = ~rotations_mod[rect_idx]
            solution = (locations.copy(), rotations_mod)
            if self.is_feasible(solution):
                neighbors += [solution]

        # Rect movement (by 1, 4 or box_length in any direction)
        directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        for rect_idx in range(self.num_rects):  # for each rectangle
            for direction in directions:  # for each direction
                for distance in [1, 4, self.box_length]:  # for each distance option
                    locations_mod = locations.copy()
                    locations_mod[rect_idx] += direction * distance

                    solution = (locations_mod, rotations.copy())
                    if self.is_feasible(solution):
                        neighbors += [solution]

        return neighbors

    def get_occupied_boxes(self, x):
        """Returns the coordinates of all occupied boxes as a set for a given solution x."""
        locations, _ = x
        box_coords = locations // self.box_length
        return set(tuple(map(tuple, box_coords)))

    def is_optimal(self, x):
        """Returns true if the solution is optimal."""
        return self.f(x) <= self.minimum_lower_bound

    def get_elements(self):
        pass

    def c(self, e):
        pass

    def is_independent(self, x):
        pass
