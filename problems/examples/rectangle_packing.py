from problems.neighborhood import NeighborhoodProblem
from problems.optimization import Solution
from problems.construction import IndependenceSystemProblem
import numpy as np
import time
import copy
import itertools


MAX_CONSIDERED_BOXES = 256
MAX_SELECTED_PLACINGS = 4


class RectanglePackingSolution(Solution):
    def __init__(self, problem):
        super(RectanglePackingSolution, self).__init__()

        self.problem = problem
        self.locations = None
        self.rotations = None

        self.box_ids = None
        self.box_coords = None

        self.box_occupancies = None
        self.box_rect_cnts = None
        self.box2rects = None

        self.boxes_grid = None

        self.standalone = True  # If False, attributes of this class require deepcopy before any modification
        self.move_pending = False
        self.pending_move_params = None

    def set_solution(self, locations, rotations):
        self.locations = locations.copy()
        self.rotations = rotations.copy()

        self.__set_box_info(locations, rotations)

    def __set_box_info(self, locations, rotations):
        # Fetch rect info
        rect_box_coords = locations // self.problem.box_length
        box_relative_locations = locations % self.problem.box_length
        sizes = self.problem.sizes.copy()

        # Consider all rotations
        sizes[rotations, 0] = self.problem.sizes[rotations, 1]
        sizes[rotations, 1] = self.problem.sizes[rotations, 0]

        # Identify all occupied boxes
        occupied_boxes = list(set(tuple(map(tuple, rect_box_coords))))

        # Save their IDs and their coordinates
        self.box_ids = {b: idx for idx, b in enumerate(occupied_boxes)}
        self.box_coords = np.array(occupied_boxes, dtype=np.int)

        # Determine box occupancies, rect counts and box to rect relation
        occupancies = np.zeros(len(rect_box_coords), dtype=np.int)
        rect_cnts = np.zeros(len(rect_box_coords), dtype=np.int)
        box2rects = {i: [] for i in range(self.problem.num_rects)}
        for rect_idx, box in enumerate(rect_box_coords):
            box_id = self.box_ids[tuple(box)]
            occupancies[box_id] += self.problem.areas[rect_idx]
            rect_cnts[box_id] += 1
            box2rects[box_id] += [rect_idx]
        self.box_occupancies, self.box_rect_cnts, self.box2rects = occupancies, rect_cnts, box2rects

        # Generate box grid array
        self.boxes_grid = np.zeros((self.problem.num_rects,
                                    self.problem.box_length,
                                    self.problem.box_length), dtype=np.int)
        begins = box_relative_locations
        ends = box_relative_locations + sizes
        for begin, end, box in zip(begins, ends, rect_box_coords):
            box_id = self.box_ids[tuple(box)]
            self.boxes_grid[box_id, begin[0]:end[0], begin[1]:end[1]] += 1

    def move_rect(self, rect_idx, target_pos, rotated):
        """Assumes that this action leads to a feasible solution."""
        if self.move_pending:
            return ValueError("Cannot add another pending move if there is already one.")
        self.move_pending = True
        self.pending_move_params = rect_idx, target_pos, rotated

    def apply_pending_move(self):
        if not self.move_pending:
            return

        rect_idx, target_pos, rotated = self.pending_move_params
        source_box_idx, target_box_idx = self.get_source_target_box_ids(rect_idx, target_pos, update_ids=True)

        self.box_occupancies[source_box_idx] -= self.problem.areas[rect_idx]
        self.box_occupancies[target_box_idx] += self.problem.areas[rect_idx]

        self.box_rect_cnts[source_box_idx] -= 1
        self.box_rect_cnts[target_box_idx] += 1

        self.box2rects[source_box_idx].remove(rect_idx)
        self.box2rects[target_box_idx].append(rect_idx)

        x, y = self.locations[rect_idx] % self.problem.box_length
        w, h = self.problem.sizes[rect_idx]
        if self.rotations[rect_idx]:
            w, h = h, w
        self.boxes_grid[source_box_idx, x:x+w, y:y+h] -= 1

        x, y = target_pos % self.problem.box_length
        w, h = self.problem.sizes[rect_idx]
        if rotated:
            w, h = h, w
        self.boxes_grid[target_box_idx, x:x+w, y:y+h] += 1

        self.locations[rect_idx] = target_pos
        self.rotations[rect_idx] = rotated

        self.move_pending = False
        self.pending_move_params = None

    def get_source_target_box_ids(self, rect_idx, target_pos, update_ids=False):
        source_box = tuple(self.locations[rect_idx] // self.problem.box_length)
        source_box_idx = self.box_ids[source_box]
        target_box = tuple(target_pos // self.problem.box_length)

        if target_box not in self.box_ids.keys():
            if self.box_rect_cnts[source_box_idx] == 1:
                target_box_idx = source_box_idx
            else:
                target_box_idx = np.where(self.box_rect_cnts == 0)[0][0]
            if update_ids:
                self.box_ids.pop(source_box)
                self.box_ids[target_box] = target_box_idx
                self.box_coords[target_box_idx] = target_box
        else:
            target_box_idx = self.box_ids[target_box]

        return source_box_idx, target_box_idx

    def copy(self):
        if self.move_pending:
            self.apply_pending_move()

        new_solution = RectanglePackingSolution(self.problem)

        new_solution.locations = self.locations
        new_solution.rotations = self.rotations
        new_solution.box_ids = self.box_ids
        new_solution.box_coords = self.box_coords
        new_solution.box_occupancies = self.box_occupancies
        new_solution.box_rect_cnts = self.box_rect_cnts
        new_solution.box2rects = self.box2rects
        new_solution.boxes_grid = self.boxes_grid

        new_solution.standalone = False

        return new_solution

    def make_standalone(self):
        if not self.standalone:
            self.locations = self.locations.copy()
            self.rotations = self.rotations.copy()
            self.box_ids = self.box_ids.copy()
            self.box_coords = self.box_coords.copy()
            self.box_occupancies = self.box_occupancies.copy()
            self.box_rect_cnts = self.box_rect_cnts.copy()
            self.box2rects = copy.deepcopy(self.box2rects)
            self.boxes_grid = self.boxes_grid.copy()
            self.standalone = True


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
        num_top_dogs = np.sum(self.top_dogs)  # each top dog rectangle requires an own box (no two top dogs in one rectangle)
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
        oversize = self.box_length // 2
        self.top_dogs = np.all(self.sizes > oversize, axis=1)  # "Platzhirsche"

    def objective_function(self, solution: RectanglePackingSolution):
        """Returns the number of boxes occupied in the current solution. Function symbol f.
        Assumes the solution to be feasible!"""
        if solution.move_pending:
            box_rect_cnts = solution.box_rect_cnts.copy()

            rect_idx, target_pos, rotated = solution.pending_move_params
            source_box_idx, target_box_idx = solution.get_source_target_box_ids(rect_idx, target_pos)

            box_rect_cnts[source_box_idx] -= 1
            box_rect_cnts[target_box_idx] += 1
        else:
            box_rect_cnts = solution.box_rect_cnts
        return np.sum(box_rect_cnts > 0)

    def heuristic(self, solution: RectanglePackingSolution):
        """Assumes the solution to be feasible!"""
        return self.__box_occupancy_heuristic(solution)

    def __rect_cnt_heuristic(self, solution: RectanglePackingSolution):
        """Depends on rectangle count per box."""
        if solution.move_pending:
            raise NotImplementedError
        cost = 1 - 0.5 ** solution.box_rect_cnts
        return np.sum(cost)

    def __box_occupancy_heuristic(self, solution: RectanglePackingSolution):
        """Penalizes comparably low occupied boxes more."""
        if solution.move_pending:
            box_occupancies = solution.box_occupancies.copy()

            rect_idx, target_pos, rotated = solution.pending_move_params
            source_box_idx, target_box_idx = solution.get_source_target_box_ids(rect_idx, target_pos)

            box_occupancies[source_box_idx] -= solution.problem.areas[rect_idx]
            box_occupancies[target_box_idx] += solution.problem.areas[rect_idx]
        else:
            box_occupancies = solution.box_occupancies

        box_capacity = self.box_length ** 2
        cost = 1 + 0.9 * (box_occupancies[box_occupancies > 0] / box_capacity - 1) ** 3
        return np.sum(cost)

    def is_feasible(self, solution: RectanglePackingSolution):
        if solution.move_pending:
            raise NotImplementedError

        locations, rotations = solution.locations, solution.rotations

        # Collect rectangle properties
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
        if np.any(solution.boxes_grid > 1):
            return False

        return True

    def get_arbitrary_solution(self):
        """Returns a solution where each rectangle is placed into an own box (not rotated)."""
        num_cols = int(np.ceil(np.sqrt(self.num_rects)))
        x_locations = np.arange(self.num_rects) % num_cols
        y_locations = np.arange(self.num_rects) // num_cols
        locations = np.stack([x_locations, y_locations], axis=1) * self.box_length
        rotations = np.zeros(self.num_rects, dtype=np.bool)
        solution = RectanglePackingSolution(self)
        solution.set_solution(locations, rotations)
        return solution

    def get_neighborhood(self, solution: RectanglePackingSolution):
        if self.neighborhood_relation == "geometry_based":
            return list(itertools.chain(*list(self.__place_all(solution))))
        else:
            raise NotImplementedError

    def get_next_neighbors(self, solution: RectanglePackingSolution):
        if self.neighborhood_relation == "geometry_based":
            return self.__place_all(solution)
        else:
            raise NotImplementedError

    def __place_all(self, solution: RectanglePackingSolution):
        """Returns all valid placing coordinates for all rectangles."""
        ordered_by_occupancy = solution.box_occupancies.argsort()[::-1]

        # ---- Determine a good rect selection order ----
        # Drop rects which lie in very full boxes
        box_capacity = self.box_length ** 2
        almost_full = solution.box_occupancies / box_capacity > 0.9

        rect_lists = list(solution.box2rects.values())
        rect_cnts = np.array(list(map(len, rect_lists)))[~almost_full]

        # Order rects by rect count per box
        ordered_by_rect_cnt = rect_cnts.argsort()
        ordered_rect_lists = np.array(rect_lists, dtype=np.object)[~almost_full][ordered_by_rect_cnt]

        # Sub-order by rect size
        rect_ids = []
        for rect_cnt in set(list(rect_cnts)):
            have_count = rect_cnts[ordered_by_rect_cnt] == rect_cnt
            rect_selection = np.concatenate(ordered_rect_lists[have_count]).astype(np.int)
            ordered_by_area = self.areas[rect_selection].argsort()
            rect_ids += [rect_selection[ordered_by_area][::-1]]

        rect_ids = np.concatenate(rect_ids).astype(np.int)

        # Throw away all top-dogs
        rect_ids = rect_ids[~self.top_dogs[rect_ids]]

        # ---- Check placements using sliding window approach ----
        for rect_idx in rect_ids:
            # Select the n most promising boxes
            box_capacity = self.box_length ** 2
            max_occupancy = box_capacity - self.areas[rect_idx]
            promising_boxes = solution.box_occupancies <= max_occupancy  # drop boxes which are too full
            sorted_box_ids = ordered_by_occupancy
            selected_box_ids = sorted_box_ids[promising_boxes[ordered_by_occupancy]]
            selected_box_ids = selected_box_ids[:MAX_CONSIDERED_BOXES]  # take at most a certain number of boxes

            solutions = []

            for rotate in [False, True]:
                size = self.sizes[rect_idx]
                if rotate:
                    size = size[::-1]

                # Identify all locations which are allowed for placement
                regions_to_place = np.lib.stride_tricks.sliding_window_view(solution.boxes_grid[selected_box_ids], size, axis=(1, 2))
                b, x, y = np.where(~np.any(regions_to_place, axis=(3, 4)))
                b_id = selected_box_ids[b]
                b_loc = solution.box_coords[b_id] * self.box_length
                locs = b_loc + np.stack([x, y], axis=1)

                # Consider only one placement per box
                b_id_cmp = np.zeros(b_id.shape, dtype=np.int)
                b_id_cmp[1:] = b_id[:-1]
                is_first_valid_placement = b_id_cmp < b_id
                relevant_locs = locs[is_first_valid_placement][:MAX_SELECTED_PLACINGS]

                # Generate new solutions
                for loc in relevant_locs:
                    new_solution = solution.copy()
                    new_solution.move_rect(rect_idx, loc, rotate)
                    # assert self.is_feasible(new_solution)
                    solutions += [new_solution]

            # print("generating %d neighbors took %.3f s" % (len(solutions), time.time() - t))
            yield solutions

    def is_optimal(self, solution: RectanglePackingSolution):
        """Returns true if the solution is optimal."""
        return self.objective_function(solution) <= self.minimum_lower_bound

    def get_elements(self):
        pass

    def c(self, e):
        pass

    def is_independent(self, solution: RectanglePackingSolution):
        pass
