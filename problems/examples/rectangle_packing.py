from problems.neighborhood import NeighborhoodProblem
from problems.construction import IndependenceSystemProblem
import numpy as np
import time
import itertools

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
        self.areas = self.sizes[:, 0] * self.sizes[:, 1]

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
        return self.__box_occupancy_heuristic(x)

    def __rect_cnt_heuristic(self, x):
        """Depends on rectangle count per box."""
        rect_cnts = np.array(list(self.__get_rect_cnts(x).values()))
        cost = 1 - 0.5 ** rect_cnts
        return np.sum(cost)

    def __box_occupancy_heuristic(self, x):
        """Penalizes comparably low occupied boxes more."""
        occupancies = np.array(list(self.__get_occupancies(x).values()))

        # Convert into cost value
        box_capacity = self.box_length ** 2
        cost = 1 + (occupancies / box_capacity - 1) ** 3
        return np.sum(cost)

    def __get_occupancies(self, x):
        """Identifies and returns the relative load (occupancy) for each box."""
        boxes = self.get_occupied_boxes(x)
        locations, _ = x
        box_coords = locations // self.box_length

        occupancies = {}
        for box in boxes:
            occupancies[box] = 0
        for rect_idx, b in enumerate(box_coords):
            occupancies[tuple(b)] += self.areas[rect_idx]

        return occupancies

    def __get_rect_cnts(self, x):
        """Counts rectangles per box."""
        boxes = self.get_occupied_boxes(x)
        locations, _ = x
        box_coords = locations // self.box_length

        rect_cnts = {}
        for box in boxes:
            rect_cnts[box] = 0
        for rect_idx, b in enumerate(box_coords):
            rect_cnts[tuple(b)] += 1

        return rect_cnts

    def __get_box2rects(self, x):
        """Returns a dict with boxes as keys and lists of rect IDs as values."""
        boxes = self.get_occupied_boxes(x)
        locations, _ = x
        box_coords = locations // self.box_length

        rect_lists = {}
        for box in boxes:
            rect_lists[box] = []
        for rect_idx, b in enumerate(box_coords):
            rect_lists[tuple(b)] += [rect_idx]

        return rect_lists

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
            return list(itertools.chain(*list(self.__place_all(x))))
        else:
            raise NotImplementedError

    def get_next_neighbors(self, x):
        if self.neighborhood_relation == "geometry_based":
            return self.__place_all(x)
        else:
            raise NotImplementedError

    def __get_next_geometry_based_neighbor(self, x):
        # Rect placement inside other boxes
        boxes = self.get_occupied_boxes(x)
        for rect_idx in range(self.num_rects):  # for each rectangle
            for box in boxes:
                new_solution = self.__place(x, rect_idx, box)
                if new_solution is not None:
                    yield new_solution

        """# Topological rect movement
        directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        for rect_idx in range(self.num_rects):  # for each rectangle
            for direction in directions:  # for each direction
                for distance in [1, 2, 4]:  # for each distance option
                    locations_mod = locations.copy()
                    locations_mod[rect_idx] += direction * distance

                    solution = (locations_mod, rotations.copy())
                    if self.is_feasible(solution):
                        yield solution

        # Rotations
        for rect_idx in range(self.num_rects):
            rotations_mod = rotations.copy()
            rotations_mod[rect_idx] = ~rotations_mod[rect_idx]
            solution = (locations.copy(), rotations_mod)
            if self.is_feasible(solution):
                yield solution"""

    def get_occupied_boxes(self, x):
        """Returns the coordinates of all occupied boxes as a set for a given solution x."""
        locations, _ = x
        box_coords = locations // self.box_length
        return set(tuple(map(tuple, box_coords)))

    def __place_all(self, solution):
        """Returns all valid placing coordinates for all rectangles."""
        # Fetch rect info
        locations, rotations = solution
        sizes = self.sizes.copy()

        # Consider all rotations
        sizes[rotations, 0] = self.sizes[rotations, 1]
        sizes[rotations, 1] = self.sizes[rotations, 0]

        # Prepare boxes for grid method
        occupancies = self.__get_occupancies(solution)
        occupancy_values = np.array(list(occupancies.values()))
        occupancy_order = occupancy_values.argsort()
        box_coords = list(occupancies.keys())
        box_ids = {k: v for v, k in enumerate(box_coords)}
        box_coords = np.array(box_coords)
        num_boxes = len(box_coords)
        boxes_grid = np.zeros((num_boxes, self.box_length, self.box_length), dtype=np.bool)

        rect_box_coords = locations // self.box_length
        locations_rel = locations % self.box_length

        begins = locations_rel
        ends = locations_rel + sizes

        for begin, end, box_coord in zip(begins, ends, rect_box_coords):
            box_id = box_ids[tuple(box_coord)]
            boxes_grid[box_id, begin[0]:end[0], begin[1]:end[1]] = 1

        # Determine an efficient rect order
        box2rects = self.__get_box2rects(solution)
        rect_lists = list(box2rects.values())
        rect_cnts = np.array(list(map(len, rect_lists)))
        order = rect_cnts.argsort()
        ordered_rect_lists = np.array(rect_lists)[order]
        ordered_rect_ids = np.concatenate(ordered_rect_lists).astype(np.int)

        # Check placements using sliding window approach
        for rect_idx in ordered_rect_ids:
            size = self.sizes[rect_idx]

            # Select the n most promising boxes
            box_capacity = self.box_length ** 2
            max_occupancy = box_capacity - self.areas[rect_idx]
            promising_boxes = occupancy_values < max_occupancy  # drop boxes which are too full
            sorted_box_ids = occupancy_order
            selected_box_ids = sorted_box_ids[promising_boxes[occupancy_order]]
            selected_box_ids = selected_box_ids[-16:]  # take at most 32 boxes

            # Identify all locations which are allowed for placement
            regions_to_place = np.lib.stride_tricks.sliding_window_view(boxes_grid[selected_box_ids], size, axis=(1, 2))
            b, x, y = np.where(~np.any(regions_to_place, axis=(3, 4)))
            b_id = selected_box_ids[b]
            b_loc = box_coords[b_id] * self.box_length
            loc = b_loc + np.stack([x, y], axis=1)

            # Consider only one placement per box
            b_id_cmp = np.zeros(b_id.shape, dtype=np.int)
            b_id_cmp[1:] = b_id[:-1]
            is_first_valid_placement = b_id_cmp < b_id
            relevant_locs = loc[is_first_valid_placement]

            # Generate new solutions
            solutions = []
            for loc in relevant_locs:
                new_locations = locations.copy()
                new_locations[rect_idx] = loc
                solutions += [(new_locations, rotations)]
            yield solutions

    def __place(self, solution, rect_idx, target_box):
        """Takes a solution, places the rectangle with the given index into the specified box
        and returns it as a new solution. Returns None if placement is impossible."""

        # Fetch rect info
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
