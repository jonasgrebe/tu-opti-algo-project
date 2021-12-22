from abc import ABC

from problems.neighborhood import NeighborhoodProblem, OptProblem
from problems.rectangle_packing.solution import RectanglePackingSolutionGeometryBased, \
    RectanglePackingSolutionRuleBased, RectanglePackingSolution
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

        return locs

    def get_good_rect_selection_order(self, box_occupancies, box2rects):
        # Drop rects which lie in very full boxes
        box_capacity = self.box_length ** 2
        almost_full = box_occupancies / box_capacity > 0.9

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

        # Throw away all top-dogs
        return rect_ids[~self.top_dogs[rect_ids]]


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

        # ---- Determine a good rect selection order ----
        rect_ids = self.get_good_rect_selection_order(solution.box_occupancies, solution.box2rects)

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
                relevant_locs = self.place(rect_size=size,
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

        rect_ids = self.get_good_rect_selection_order(sol.box_occupancies, sol.box2rects)

        for rect_idx in rect_ids:
            target_order_pos = 0  # TODO

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

    def put_all_rects(self, sol: RectanglePackingSolutionRuleBased):
        sol.reset()
        for rect_idx in sol.rect_order:
            selected_box_ids = np.arange(self.num_rects)  # TODO: reduce selection
            for rotate in [False, True]:  # TODO
                rect_size = self.sizes[rect_idx] if not rotate else self.sizes[rect_idx][::-1]
                place_locations = self.place(rect_size=rect_size,
                                             boxes_grid=sol.boxes_grid,
                                             selected_box_ids=selected_box_ids,
                                             box_coords=sol.box_coords)
            target_pos = place_locations[0]
            sol.put_rect(rect_idx, target_pos=target_pos, rotated=rotate, update_ids=True)
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
