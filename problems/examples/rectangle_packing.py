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
        self.w_min, self.w_max = w_min, w_max
        self.h_min, self.h_max = h_min, h_max
        self.neighborhood_relation = neighborhood_relation
        self.generate()

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

    def generate(self):
        self.__generate(self.box_length, self.num_rects, self.w_min, self.w_max, self.h_min, self.h_max)

    def rotate_rect(self, idx):
        """Rotates the rectangle with index idx."""
        rect = self.sizes[idx]
        self.sizes[idx] = [rect[1], rect[0]]

    def f(self, x):
        """Returns the number of boxes occupied in the current solution.
        Warning: Only defined for feasible solutions! Unfeasible solutions will yield
        invalid values."""
        return len(self.get_occupied_boxes(x))

    def h(self, x):
        return self.box_occupancy_heuristic(x)

    def rect_cnt_heuristic(self, x):
        """Depends on rectangle count per box."""
        boxes = self.get_occupied_boxes(x)
        locations, _ = x
        box_coords = locations // self.box_length

        # Count rectangles per box
        rect_cnt = {}
        for box in boxes:
            rect_cnt[box] = 0
        for b in box_coords:
            rect_cnt[tuple(b)] += 1

        cost = 0
        for box in boxes:
            cost += 1 - 0.5 ** rect_cnt[box]
        return cost

    def box_occupancy_heuristic(self, x):
        """Depends on occupancy inside box."""
        boxes = self.get_occupied_boxes(x)
        locations, _ = x
        box_coords = locations // self.box_length

        # Count occupancy per box
        occupancy = {}
        for box in boxes:
            occupancy[box] = 0
        for rect_idx, b in enumerate(box_coords):
            occupancy[tuple(b)] += self.sizes[rect_idx, 0] * self.sizes[rect_idx, 1]

        cost = 0
        box_capacity = self.box_length ** 2
        for box in boxes:
            cost += 1 + (occupancy[box] / box_capacity - 1) ** 3
        return cost

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

        # Rect placement inside other boxes
        boxes = self.get_occupied_boxes(x)
        for rect_idx in range(self.num_rects):  # for each rectangle
            for box in boxes:
                new_solution = self.__place(x, rect_idx, box)
                if new_solution is not None:
                    neighbors += [new_solution]

        # Topological rect movement
        directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        for rect_idx in range(self.num_rects):  # for each rectangle
            for direction in directions:  # for each direction
                for distance in [1, 2, 4]:  # for each distance option
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

    def __place(self, solution, rect_idx, target_box):
        """Takes a solution, places the rectangle with the given index into the specified box
        and returns it as a new solution. Returns None if placement is impossible."""

        # Fetch box info
        locations, rotations = solution
        sizes = self.sizes.copy()

        # Consider all rotations
        sizes[rotations, 0] = self.sizes[rotations, 1]
        sizes[rotations, 1] = self.sizes[rotations, 0]

        # Get all rects inside specified box
        box_coords = locations // self.box_length
        inside_target_box = np.all(box_coords == target_box, axis=1)
        inside_target_box[rect_idx] = False  # Consider the rect to place outside the target box
        locations_rel = locations[inside_target_box] % self.box_length

        # Create virtual box
        virtual_box = np.zeros((self.box_length, self.box_length), dtype=np.bool)

        # Place each single rect into the virtual box
        for (x, y), (w, h) in zip(locations_rel, sizes[inside_target_box]):
            region_to_place = virtual_box[x:(x + w), y:(y + h)]
            region_to_place[:] = 1

        # Try to place the new rect into that box
        w, h = self.sizes[rect_idx]
        for rotated in [False, True]:
            if rotated:
                w, h = h, w

            for x in range(self.box_length - w + 1):
                for y in range(self.box_length - h + 1):
                    region_to_place = virtual_box[x:(x + w), y:(y + h)]

                    if not np.any(region_to_place):
                        # We found a valid placement
                        new_locations = locations.copy()
                        new_locations[rect_idx] = [target_box[0] * self.box_length + x,
                                                   target_box[1] * self.box_length + y]
                        new_rotations = rotations.copy()
                        new_rotations[rect_idx] = rotated
                        return new_locations, new_rotations

        return None

    def is_optimal(self, x):
        """Returns true if the solution is optimal."""
        return self.f(x) <= self.minimum_lower_bound

    def get_elements(self):
        pass

    def c(self, e):
        pass

    def is_independent(self, x):
        pass
