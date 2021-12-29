from abc import ABC

from problems.construction import ConstructionProblem
from problems.neighborhood import NeighborhoodProblem, OptProblem
from problems.rectangle_packing.solution import RectanglePackingSolutionGeometryBased, \
    RectanglePackingSolutionRuleBased, RectanglePackingSolutionGreedy, RectanglePackingSolution

import numpy as np
import itertools

MAX_CONSIDERED_BOXES = 256
MAX_SELECTED_PLACINGS = 4


class RectanglePackingProblem(OptProblem, ABC):
    def __init__(self, box_length, num_rects, w_min, w_max, h_min, h_max):
        super(RectanglePackingProblem, self).__init__(is_max=False)
        self.box_length = box_length
        self.num_rects = num_rects
        self.w_min, self.w_max = w_min, w_max
        self.h_min, self.h_max = h_min, h_max

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
        widths = np.random.randint(w_min, w_max + 1, size=num_rects)
        heights = np.random.randint(h_min, h_max + 1, size=num_rects)
        self.sizes = np.stack([widths, heights], axis=1)
        self.areas = self.sizes[:, 0] * self.sizes[:, 1]
        oversize = self.box_length // 2
        self.top_dogs = np.all(self.sizes > oversize, axis=1)  # "Platzhirsche"

        # Compute a lower bound for the minimum
        # "top dog" := a rectangle that requires an own box (no two top dogs fit together into one box)
        num_top_dogs = np.sum(self.top_dogs)
        min_box_required = np.ceil(np.sum(self.sizes[:, 0] * self.sizes[:, 1]) / self.box_length ** 2)
        self.minimum_lower_bound = max(min_box_required, num_top_dogs)

    def is_optimal(self, solution: RectanglePackingSolution):
        """If True is returned, the solution is optimal
        (otherwise no assertion can be made)."""
        return self.objective_function(solution) <= self.minimum_lower_bound

    def place(self, rect_size, boxes_grid, selected_box_ids, box_coords):
        regions_to_place = np.lib.stride_tricks.sliding_window_view(boxes_grid[selected_box_ids],
                                                                    rect_size, axis=(1, 2))
        b, x, y = np.where(~np.any(regions_to_place, axis=(3, 4)))

        # Consider only one placement per box
        b_cmp = b.copy()
        b_cmp[0] = -1
        b_cmp[1:] = b[:-1]
        first_valid_placement = b_cmp < b
        b, x, y, = b[first_valid_placement], x[first_valid_placement], y[first_valid_placement]

        # Convert into location data
        b_id = selected_box_ids[b]
        b_loc = box_coords[b_id] * self.box_length
        locs = b_loc + np.stack([x, y], axis=1)

        return locs, b

    def get_rect_selection_order(self, box_occupancies, box2rects, occupancy_threshold=0.9, keep_top_dogs=False):
        # Drop rects which lie in very full boxes
        box_capacity = self.box_length ** 2
        almost_full = box_occupancies / box_capacity > occupancy_threshold

        rect_lists = list(box2rects.values())
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

        if not keep_top_dogs:
            # Throw away all top-dogs
            rect_ids = rect_ids[~self.top_dogs[rect_ids]]

        return rect_ids

    def get_rect_area(self, rect_idx):
        return np.prod(self.sizes[rect_idx])

    def get_rect_areas(self):
        return np.prod(self.sizes, axis=1)


class RectanglePackingProblemGeometryBased(RectanglePackingProblem, NeighborhoodProblem):
    def __init__(self, *args, **kwargs):
        super(RectanglePackingProblemGeometryBased, self).__init__(*args, **kwargs)

    def objective_function(self, solution: RectanglePackingSolutionGeometryBased):
        """Returns the number of boxes occupied in the current solution. Function symbol f.
        Assumes the solution to be feasible!"""
        if solution.move_pending:
            box_rect_cnts = solution.box_rect_cnts.copy()

            rect_idx, target_pos, rotated = solution.pending_move_params
            source_box_idx = solution.get_box_idx_by_rect_id(rect_idx)
            target_box_idx = solution.get_box_idx_by_pos(target_pos)

            box_rect_cnts[source_box_idx] -= 1
            box_rect_cnts[target_box_idx] += 1
        else:
            box_rect_cnts = solution.box_rect_cnts
        return np.sum(box_rect_cnts > 0)

    def heuristic(self, solution: RectanglePackingSolutionGeometryBased):
        """Assumes the solution to be feasible!"""
        return self.__box_occupancy_heuristic(solution)

    def __rect_cnt_heuristic(self, solution: RectanglePackingSolutionGeometryBased):
        """Depends on rectangle count per box."""
        if solution.move_pending:
            raise NotImplementedError
        cost = 1 - 0.5 ** solution.box_rect_cnts
        return np.sum(cost)

    def __box_occupancy_heuristic(self, solution: RectanglePackingSolutionGeometryBased):
        """Penalizes comparably low occupied boxes more."""
        if solution.move_pending:
            box_occupancies = solution.box_occupancies.copy()

            rect_idx, target_pos, rotated = solution.pending_move_params
            source_box_idx = solution.get_box_idx_by_rect_id(rect_idx)
            target_box_idx = solution.get_box_idx_by_pos(target_pos)

            box_occupancies[source_box_idx] -= solution.problem.areas[rect_idx]
            box_occupancies[target_box_idx] += solution.problem.areas[rect_idx]
        else:
            box_occupancies = solution.box_occupancies

        box_capacity = self.box_length ** 2
        cost = 1 + 0.9 * (box_occupancies[box_occupancies > 0] / box_capacity - 1) ** 3
        return np.sum(cost)

    def is_feasible(self, solution: RectanglePackingSolutionGeometryBased):
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
        solution = RectanglePackingSolutionGeometryBased(self)
        solution.set_solution(locations, rotations)
        return solution

    def get_neighborhood(self, solution: RectanglePackingSolutionGeometryBased):
        return list(itertools.chain(*list(self.get_next_neighbors(solution))))

    def get_next_neighbors(self, solution: RectanglePackingSolutionGeometryBased):
        """Returns all valid placing coordinates for all rectangles."""
        ordered_by_occupancy = solution.box_occupancies.argsort()[::-1]

        # ---- Preprocessing: Determine a good rect selection order ----
        rect_ids = self.get_rect_selection_order(solution.box_occupancies, solution.box2rects)

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
                relevant_locs, _ = self.place(rect_size=size,
                                              boxes_grid=solution.boxes_grid,
                                              selected_box_ids=selected_box_ids,
                                              box_coords=solution.box_coords)

                # Prune abundant options
                relevant_locs = relevant_locs[:MAX_SELECTED_PLACINGS]

                # Generate new solutions
                for loc in relevant_locs:
                    new_solution = solution.copy()
                    new_solution.move_rect(rect_idx, loc, rotate)
                    # assert self.is_feasible(new_solution)
                    solutions += [new_solution]

            # print("generating %d neighbors took %.3f s" % (len(solutions), time.time() - t))
            yield solutions


class RectanglePackingProblemRuleBased(RectanglePackingProblem, NeighborhoodProblem):
    def __init__(self, *args, **kwargs):
        super(RectanglePackingProblemRuleBased, self).__init__(*args, **kwargs)

    def get_neighborhood(self, sol: RectanglePackingSolutionRuleBased):
        pass

    def get_next_neighbors(self, sol: RectanglePackingSolutionRuleBased):
        if not sol.placed:
            self.put_all_rects(sol)

        rect_selection = self.get_rect_selection_order(sol.box_occupancies, sol.box2rects, occupancy_threshold=0.9,
                                                       keep_top_dogs=True)
        rect_areas = self.get_rect_areas()
        max_area = max(rect_areas)
        min_area = min(rect_areas)

        for rect_idx, rect_area in zip(rect_selection, rect_areas[rect_selection]):
            # Set target order position depending on rectangle area, TODO: Is this a good target order position?
            if min_area == max_area:
                target_order_pos = 0
            else:
                # target_order_pos = int((self.num_rects - 1) * (1 - (rect_area - min_area) /
                #                                                (max_area - min_area)))
                target_order_pos = 0
            new_sol = sol.copy()
            new_sol.move_rect_to_order_pos(rect_idx, target_order_pos)
            yield [new_sol]

    def get_arbitrary_solution(self):
        solution = RectanglePackingSolutionRuleBased(self)
        solution.reset()
        solution.rect_order = np.arange(self.num_rects)
        return solution

    def objective_function(self, sol: RectanglePackingSolutionRuleBased):
        if not sol.placed:
            self.put_all_rects(sol)

        return np.sum(sol.box_rect_cnts > 0)

    def select_boxes_to_place(self, sol: RectanglePackingSolutionRuleBased, rect_idx: int):
        rect_area = np.prod(self.sizes[rect_idx])
        box_capacity = self.box_length ** 2

        box_selection = np.where((sol.box_occupancies > 0) &
                                 (sol.box_occupancies <= box_capacity - rect_area))

        # Add an empty box to selection (such a box always exists ir rect_idx isn't put yet)
        box_selection = np.append(box_selection, [sol.get_empty_box_ids()[0]])

        return box_selection

    def put_all_rects(self, sol: RectanglePackingSolutionRuleBased):
        sol.reset()
        for rect_idx in sol.rect_order:
            box_selection = self.select_boxes_to_place(sol, rect_idx)
            place_options = []
            for rotate in [False, True]:
                rect_size = self.sizes[rect_idx] if not rotate else self.sizes[rect_idx][::-1]
                place_locations, place_b = self.place(rect_size=rect_size,
                                                      boxes_grid=sol.boxes_grid,
                                                      selected_box_ids=box_selection,
                                                      box_coords=sol.box_coords)
                place_options += [(place_locations[0], place_b[0])]  # There always exists at least one placement
            choose_rotated = place_options[0][1] > place_options[1][1]
            target_pos = place_options[1][0] if choose_rotated else place_options[0][0]
            sol.put_rect(rect_idx, target_pos=target_pos, rotated=choose_rotated, update_ids=True)
        sol.placed = True

    def heuristic(self, sol: RectanglePackingSolutionRuleBased):
        if not sol.placed:
            self.put_all_rects(sol)

        box_occupancies = sol.box_occupancies
        box_capacity = self.box_length ** 2
        cost = 1 + 0.9 * (box_occupancies[box_occupancies > 0] / box_capacity - 1) ** 3
        return np.sum(cost)

    def is_feasible(self, sol: RectanglePackingSolutionRuleBased):
        rect_id_set = set(list(sol.rect_order))
        return len(rect_id_set) == self.num_rects \
               and np.all(sol.rect_order >= 0) \
               and np.all(sol.rect_order < self.num_rects)


class RectanglePackingProblemGreedyLargestFirstStrategy(RectanglePackingProblem, ConstructionProblem):
    def __init__(self, *args, **kwargs):
        super(RectanglePackingProblemGreedyLargestFirstStrategy, self).__init__(*args, **kwargs)

    def objective_function(self, sol: RectanglePackingSolutionRuleBased):
        # if not sol.placed:
        #    self.put_all_rects(sol)

        return np.sum(sol.box_rect_cnts > 0)

    def heuristic(self, sol: RectanglePackingSolutionRuleBased):
        # if not sol.placed:
        #    self.put_all_rects(sol)

        box_occupancies = sol.box_occupancies
        box_capacity = self.box_length ** 2
        cost = 1 + 0.9 * (box_occupancies[box_occupancies > 0] / box_capacity - 1) ** 3
        return np.sum(cost)

    def is_feasible(self, sol: RectanglePackingSolutionRuleBased):
        rect_id_set = set(list(sol.rect_order))
        return len(rect_id_set) == self.num_rects \
               and np.all(sol.rect_order >= 0) \
               and np.all(sol.rect_order < self.num_rects)

    def get_empty_solution(self):
        solution = RectanglePackingSolutionGreedy(self)
        solution.reset()
        return solution

    def get_expansion(self, solution: RectanglePackingSolutionGreedy):
        return list(itertools.chain(*list(self.get_next_expansions(solution))))

    def get_next_expansions(self, solution: RectanglePackingSolutionGreedy):
        """Returns expansion (partial solutions obtained by appending an element) of the given (partial) solution."""
        ordered_by_occupancy = solution.box_occupancies.argsort()[::-1]

        # ---- Determine a good rect selection order ----
        remaining_rect_ids = solution.get_remaining_elements()
        print("remaining ids:", remaining_rect_ids)

        remaining_rect_ids = solution.get_remaining_elements()

        # strategy: take largest rectangle first
        areas = np.prod(self.sizes[remaining_rect_ids], axis=1)
        largest_rect_idx = remaining_rect_ids[np.argmax(areas)]

        # Select the n most promising boxes
        box_capacity = self.box_length ** 2
        max_occupancy = box_capacity - self.areas[largest_rect_idx]
        promising_boxes = solution.box_occupancies <= max_occupancy  # drop boxes which are too full

        sorted_box_ids = ordered_by_occupancy
        selected_box_ids = sorted_box_ids[promising_boxes[ordered_by_occupancy]]
        selected_box_ids = selected_box_ids[:MAX_CONSIDERED_BOXES]  # take at most a certain number of boxes

        solutions = []

        for rotate in [False, True]:
            size = self.sizes[largest_rect_idx]
            if rotate:
                size = size[::-1]

            # Identify all locations which are allowed for placement
            relevant_locs, _ = self.place(rect_size=size,
                                          boxes_grid=solution.boxes_grid,
                                          selected_box_ids=selected_box_ids,
                                          box_coords=solution.box_coords)

            # TODO: Think about this part:
            # Prune abundant options
            relevant_locs = relevant_locs[:MAX_SELECTED_PLACINGS]

            # Generate expanded partial solutions
            for loc in relevant_locs:
                new_solution = solution.copy()
                new_solution.put_rect(largest_rect_idx, loc, rotate)
                # assert self.is_feasible(new_solution)
                solutions += [new_solution]

            yield [min(solutions, key=self.heuristic)]


class RectanglePackingProblemGreedySmallestFirstStrategy(RectanglePackingProblem, ConstructionProblem):
    def __init__(self, *args, **kwargs):
        super(RectanglePackingProblemGreedySmallestFirstStrategy, self).__init__(*args, **kwargs)

    def objective_function(self, sol: RectanglePackingSolutionRuleBased):
        # if not sol.placed:
        #    self.put_all_rects(sol)

        return np.sum(sol.box_rect_cnts > 0)

    def heuristic(self, sol: RectanglePackingSolutionRuleBased):
        # if not sol.placed:
        #    self.put_all_rects(sol)

        box_occupancies = sol.box_occupancies
        box_capacity = self.box_length ** 2
        cost = 1 + 0.9 * (box_occupancies[box_occupancies > 0] / box_capacity - 1) ** 3
        return np.sum(cost)

    def is_feasible(self, sol: RectanglePackingSolutionRuleBased):
        rect_id_set = set(list(sol.rect_order))
        return len(rect_id_set) == self.num_rects \
               and np.all(sol.rect_order >= 0) \
               and np.all(sol.rect_order < self.num_rects)

    def get_empty_solution(self):
        solution = RectanglePackingSolutionGreedy(self)
        solution.reset()
        return solution

    def get_expansion(self, solution: RectanglePackingSolutionGreedy):
        return list(itertools.chain(*list(self.get_next_expansions(solution))))

    def get_next_expansions(self, solution: RectanglePackingSolutionGreedy):
        """Returns expansion (partial solutions obtained by appending an element) of the given (partial) solution."""
        ordered_by_occupancy = solution.box_occupancies.argsort()[::-1]

        # ---- Determine a good rect selection order ----
        remaining_rect_ids = solution.get_remaining_elements()
        print("remaining ids:", remaining_rect_ids)

        remaining_rect_ids = solution.get_remaining_elements()

        # strategy: take largest rectangle first
        areas = np.prod(self.sizes[remaining_rect_ids], axis=1)
        smallest_rect_idx = remaining_rect_ids[np.argmin(areas)]

        # Select the n most promising boxes
        box_capacity = self.box_length ** 2
        max_occupancy = box_capacity - self.areas[smallest_rect_idx]
        promising_boxes = solution.box_occupancies <= max_occupancy  # drop boxes which are too full

        sorted_box_ids = ordered_by_occupancy
        selected_box_ids = sorted_box_ids[promising_boxes[ordered_by_occupancy]]
        selected_box_ids = selected_box_ids[:MAX_CONSIDERED_BOXES]  # take at most a certain number of boxes

        solutions = []

        for rotate in [False, True]:
            size = self.sizes[smallest_rect_idx]
            if rotate:
                size = size[::-1]

            # Identify all locations which are allowed for placement
            relevant_locs, _ = self.place(rect_size=size,
                                          boxes_grid=solution.boxes_grid,
                                          selected_box_ids=selected_box_ids,
                                          box_coords=solution.box_coords)

            # TODO: Think about this part:
            # Prune abundant options
            relevant_locs = relevant_locs[:MAX_SELECTED_PLACINGS]

            # Generate expanded partial solutions
            for loc in relevant_locs:
                new_solution = solution.copy()
                new_solution.put_rect(smallest_rect_idx, loc, rotate)
                # assert self.is_feasible(new_solution)
                solutions += [new_solution]

            yield [min(solutions, key=self.heuristic)]
