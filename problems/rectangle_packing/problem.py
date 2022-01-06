from abc import ABC

from problems.construction import ConstructionProblem
from problems.neighborhood import NeighborhoodProblem, OptProblem
from problems.rectangle_packing.solution import RectanglePackingSolutionGeometryBased, \
    RectanglePackingSolutionRuleBased, RectanglePackingSolutionOverlap, RectanglePackingSolutionGreedy, \
    RectanglePackingSolution

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

    def place(self, rect_size, boxes_grid, selected_box_ids, box_coords, one_per_box=True):
        regions_to_place = np.lib.stride_tricks.sliding_window_view(boxes_grid[selected_box_ids],
                                                                    rect_size, axis=(1, 2))

        b, x, y = np.where(~np.any(regions_to_place, axis=(3, 4)))

        if one_per_box:
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

    def is_feasible(self, sol: RectanglePackingSolutionGeometryBased):
        if sol.move_pending:
            # Remember current configuration and apply pending move
            rect_idx, target_pos, rotated = sol.pending_move_params
            orig_pos = sol.locations[rect_idx]
            sol.apply_pending_move()

            feasible = rects_correctly_placed(sol)

            # Re-construct original solution
            sol.move_rect(rect_idx, orig_pos, rotated)
            sol.apply_pending_move()
            sol.move_rect(rect_idx, target_pos, rotated)

            return feasible
        else:
            return rects_correctly_placed(sol)

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
                # Identify all locations which are allowed for placement
                relevant_locs, _ = self.place(rect_size=self.sizes[rect_idx] if not rotate else self.sizes[rect_idx][::-1],
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
        if not sol.all_rects_put():
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
        if not sol.all_rects_put():
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
        # sol.reset()
        rects_to_put = sol.rect_order[~sol.is_put[sol.rect_order]]
        for rect_idx in rects_to_put:
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

    def heuristic(self, sol: RectanglePackingSolutionRuleBased):
        if not sol.all_rects_put():
            self.put_all_rects(sol)
        return occupancy_heuristic(sol)

    def is_feasible(self, sol: RectanglePackingSolutionRuleBased):
        rect_id_set = set(list(sol.rect_order))
        return len(rect_id_set) == self.num_rects \
               and np.all(sol.rect_order >= 0) \
               and np.all(sol.rect_order < self.num_rects)


class RectanglePackingProblemOverlap(RectanglePackingProblem, NeighborhoodProblem):
    def __init__(self, *args, **kwargs):
        super(RectanglePackingProblemOverlap, self).__init__(*args, **kwargs)
        self.allowed_overlap = 0.0  # 1 = rects are allowed to overlap completely, 0 = no overlap allowed
        self.penalty_factor = 0.0

    def objective_function(self, sol: RectanglePackingSolutionOverlap):
        """Returns the number of boxes occupied in the current solution. Function symbol f.
        Assumes the solution to be feasible!"""
        if sol.move_pending:
            box_rect_cnts = sol.box_rect_cnts.copy()

            rect_idx, target_pos, rotated = sol.pending_move_params
            source_box_idx = sol.get_box_idx_by_rect_id(rect_idx)
            target_box_idx = sol.get_box_idx_by_pos(target_pos)

            box_rect_cnts[source_box_idx] -= 1
            box_rect_cnts[target_box_idx] += 1
        else:
            box_rect_cnts = sol.box_rect_cnts
        return np.sum(box_rect_cnts > 0)

    def heuristic(self, sol: RectanglePackingSolutionOverlap):
        """Assumes the solution to be feasible!"""
        return self.__position_heuristic(sol) + self.penalty(sol)

    def penalty(self, sol: RectanglePackingSolutionOverlap):
       return np.sum(sol.boxes_grid[sol.boxes_grid > 1] - 1) * self.penalty_factor

    def __position_heuristic(self, sol: RectanglePackingSolutionOverlap):
        pos_sum = sol.locations.sum()
        box_pos_sum = (sol.locations % self.box_length).sum()
        if sol.move_pending:
            rect_idx, target_pos, _ = sol.pending_move_params
            target_box_pos = target_pos % self.box_length

            source_pos = sol.locations[rect_idx]
            source_box_pos = source_pos % self.box_length

            pos_sum += target_pos.sum() - source_pos.sum()
            box_pos_sum += target_box_pos.sum() - source_box_pos.sum()

        return pos_sum + box_pos_sum

    def __box_occupancy_heuristic(self, sol: RectanglePackingSolutionOverlap):
        """Penalizes comparably low occupied boxes more."""
        if sol.move_pending:
            box_occupancies = sol.box_occupancies.copy()

            rect_idx, target_pos, _ = sol.pending_move_params
            source_box_idx = sol.get_box_idx_by_rect_id(rect_idx)
            target_box_idx = sol.get_box_idx_by_pos(target_pos)

            box_occupancies[source_box_idx] -= sol.problem.areas[rect_idx]
            box_occupancies[target_box_idx] += sol.problem.areas[rect_idx]
        else:
            box_occupancies = sol.box_occupancies

        box_capacity = self.box_length ** 2
        cost = 1 + 0.9 * (box_occupancies[box_occupancies > 0] / box_capacity - 1) ** 3
        return np.sum(cost)

    def is_feasible(self, sol: RectanglePackingSolutionOverlap):
        if sol.move_pending:
            # Remember current configuration and apply pending move
            rect_idx, target_pos, rotated = sol.pending_move_params
            orig_pos = sol.locations[rect_idx]

            sol.apply_pending_move()

            feasible = self.__rect_overlap_tolerable(sol) and rects_respect_boxes(sol)

            # Re-construct original solution
            sol.move_rect(rect_idx, orig_pos, rotated)
            sol.apply_pending_move()
            sol.move_rect(rect_idx, target_pos, rotated)

            return feasible
        else:
            return self.__rect_overlap_tolerable(sol) and rects_respect_boxes(sol)

    def get_arbitrary_solution(self):
        """Returns a solution where each rectangle is placed into an own box (not rotated)."""
        num_cols = int(np.ceil(np.sqrt(self.num_rects)))
        x_locations = np.arange(self.num_rects) % num_cols
        y_locations = np.arange(self.num_rects) // num_cols
        locations = np.stack([x_locations, y_locations], axis=1) * self.box_length
        rotations = np.zeros(self.num_rects, dtype=np.bool)
        sol = RectanglePackingSolutionOverlap(self)
        sol.set_solution(locations, rotations)
        return sol

    def get_neighborhood(self, solution: RectanglePackingSolutionOverlap):
        return list(itertools.chain(*list(self.get_next_neighbors(solution))))

    def get_next_neighbors(self, solution: RectanglePackingSolutionOverlap):
        """Returns all valid placing coordinates for all rectangles."""
        ordered_by_occupancy = solution.box_occupancies.argsort()[::-1]

        # ---- Preprocessing: Determine a good rect selection order ----
        rect_ids = self.get_rect_selection_order(solution.box_occupancies, solution.box2rects, occupancy_threshold=1.0, keep_top_dogs=True)
        print("RECT_IDS", len(rect_ids))

        # ---- Check placements using sliding window approach ----
        for rect_idx in rect_ids:
            # Select the n most promising boxes
            box_capacity = self.box_length ** 2
            max_occupancy = box_capacity - self.areas[rect_idx] * (1 - self.allowed_overlap)
            selected_box_ids = ordered_by_occupancy
            #selected_box_ids = selected_box_ids[:MAX_CONSIDERED_BOXES]  # take at most a certain number of boxes

            solutions = []

            for rotate in [False, True]:
                # Identify all locations which are allowed for placement
                relevant_locs = self.__place_with_overlap(rect_size=self.sizes[rect_idx] if not rotate else self.sizes[rect_idx][::-1],
                                              boxes_grid=solution.boxes_grid,
                                              selected_box_ids=selected_box_ids,
                                              box_coords=solution.box_coords,
                                              rectangle_fields=solution.rectangle_fields,
                                              box2rects=solution.box2rects
                                              )

                # Prune abundant options
                #relevant_locs = relevant_locs[:MAX_SELECTED_PLACINGS]

                # Generate new solutions
                for loc in relevant_locs:
                    new_solution = solution.copy()
                    new_solution.move_rect(rect_idx, loc, rotate)
                    # assert self.is_feasible(new_solution)
                    solutions += [new_solution]

            # print("generating %d neighbors took %.3f s" % (len(solutions), time.time() - t))
            yield solutions


    def __place_with_overlap(self, rect_size, boxes_grid, selected_box_ids, rectangle_fields, box2rects, box_coords):

        if self.allowed_overlap == 0:
            #  handle the simple case (can also be handles with the 'else' case but probably takes more time)
            regions_to_place = np.lib.stride_tricks.sliding_window_view(boxes_grid[selected_box_ids], rect_size, axis=(1, 2))
            b, x, y = np.where(~np.any(regions_to_place, axis=(3, 4)))
        else:

            # -------- quick filter for rectangles in the selected boxes -------
            #contained_rectangles = []
            #for box_id in selected_box_ids:
            #    contained_rectangles.extend(box2rects[box_id])
            #contained_rectangles = np.array(contained_rectangles)

            # (b, rects, L, L) -> (b, rects, x, y, w, h)
            regions_to_place_per_rect = np.lib.stride_tricks.sliding_window_view(rectangle_fields[selected_box_ids], rect_size, axis=(2, 3))

            # (b, rects, x, y, w, h) -> (b, x, y, rects, w, h)
            regions_to_place_per_rect = regions_to_place_per_rect.transpose((0, 2, 3, 1, 4, 5))

            # --------- get only the rects per region that overlap -------------
            # (b, x, y, rects, w, h) -> (b, x, y, rects)
            r_overlaps = regions_to_place_per_rect.sum(axis=(4, 5))

            # (b, x, y, rects)
            b, x, y, r = np.where(r_overlaps >= 0)

            # get areas of involved rectangles
            r1_area = np.prod(rect_size)
            r2_areas = self.areas[r]

            # focus on the pairwise maxima and compute ratios of overlaps
            max_areas = np.maximum(r1_area, r2_areas)
            overlap_ratios = np.divide(r_overlaps.reshape(1, -1), max_areas)

            # is the overlap not exceeding the allowed overlap?
            valid = overlap_ratios <= self.allowed_overlap
            # reshape to the convenient (b, x, y, r) shape
            valid = valid.reshape(r_overlaps.shape)

            # all (b, x, y) triplets are valid for which all r values are True (no rectangle invalidates this placement)
            b, x, y = np.where(np.all(valid, axis=3))

            #m = overlap_ratios.reshape(r_overlaps.shape)[b, x, y].max()
            #print("\nPLACE:", overlap_ratios.reshape(r_overlaps.shape)[b, x, y].max(), self.allowed_overlap)

        if len(b) == 0:
            return []

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


    def __rect_overlap_tolerable(self, sol: RectanglePackingSolutionOverlap):
        # Identify boxes with overlap
        overlap_box_ids = list(set(list(np.where(sol.boxes_grid > 1)[0])))

        sizes = sol.get_rect_sizes()

        max_seen_overlap = 0

        # For each box check the overlaps of its contained rects
        for box_idx in overlap_box_ids:
            contained_rects_ids = sol.box2rects[box_idx]
            for i, rect_1 in enumerate(contained_rects_ids[:-1]):  # O(n^2) with n = len(contained_rects_ids)
                for rect_2 in contained_rects_ids[i+1:]:
                    x_1, y_1 = sol.locations[rect_1]
                    x_2, y_2 = sol.locations[rect_2]
                    w_1, h_1 = sizes[rect_1]
                    w_2, h_2 = sizes[rect_2]

                    l_1, r_1, t_1, b_1 = x_1, x_1 + w_1, y_1, y_1 + h_1
                    l_2, r_2, t_2, b_2 = x_2, x_2 + w_2, y_2, y_2 + h_2

                    x_overlap = max(r_2 - l_1, 0) - max(l_2 - l_1, 0) - max(r_2 - r_1, 0) + max(l_2 - r_1, 0)
                    y_overlap = max(b_2 - t_1, 0) - max(t_2 - t_1, 0) - max(b_2 - b_1, 0) + max(t_2 - b_1, 0)

                    overlap_area = x_overlap * y_overlap

                    total_area = max(sol.problem.areas[rect_1], sol.problem.areas[rect_2])
                    overlap = overlap_area / total_area

                    if overlap > max_seen_overlap:
                        max_seen_overlap = overlap

                    if overlap > self.allowed_overlap:
                        print("[FEASIBLE_CHECK] max_seen_overlap:", max_seen_overlap)
                        return False

        return True



class RectanglePackingProblemGreedyStrategy(RectanglePackingProblem, ConstructionProblem):
    def __init__(self, *args, **kwargs):
        super(RectanglePackingProblemGreedyStrategy, self).__init__(*args, **kwargs)

    def objective_function(self, sol: RectanglePackingSolutionGreedy):
        return np.sum(sol.box_rect_cnts > 0)

    def heuristic(self, sol: RectanglePackingSolutionGreedy):
        return occupancy_heuristic(sol)

    def costs(self, e):
        raise NotImplementedError

    def get_elements(self, sol):
        """Returns a list of elements"""

        elements = []
        for rect_idx in range(self.num_rects):

            for rotate in [False, True]:
                # Identify all locations which are allowed for placement
                relevant_locs, _ = self.place(rect_size=self.sizes[rect_idx] if not rotate else self.sizes[rect_idx][::-1],
                                              boxes_grid=sol.boxes_grid,
                                              selected_box_ids=np.arange(self.num_rects, dtype=int),
                                              box_coords=sol.box_coords,
                                              one_per_box=False)
                # Prune abundant options
                #n_choices = min(relevant_locs.shape[0], 64)
                #relevant_locs = relevant_locs[np.random.choice(relevant_locs.shape[0], n_choices, replace=False)]
                #relevant_locs = relevant_locs[:n_choices]

                # add triplets (rect_idx, location, rotation) to elements
                elements.extend([(rect_idx, loc, rotate) for loc in relevant_locs])

        # shuffle to not influence the algorithm by the generation order of placement
        np.random.shuffle(elements)
        print(f"[GREEDY] Generated {len(elements)} elements.")
        return elements

    def filter_elements(self, sol, elements, e):
        return list(filter(lambda x: x[0] != e[0], elements))

    def filter_elements_wiht_numpy(self, sol, elements, e):
        elements = elements[np.where(elements[:, 0] != e[0])] # list(filter(lambda x: x[0] != e[0], elements))

        e_rect_idx, e_pos, e_rotated = e
        e_w, e_h = self.sizes[e_rect_idx]
        if e_rotated:
            e_w, e_h = e_h, e_w

        e_left, e_right = e_pos[0], e_pos[0] + e_w
        e_top, e_bottom = e_pos[1], e_pos[1] + e_h

        element_sizes = self.sizes[elements[:, 0].astype(int)]
        element_rotated = elements[:, 2].astype(bool)
        element_sizes[element_rotated] = element_sizes[element_rotated][:, ::-1]

        element_w, element_h = element_sizes[:, 0], element_sizes[:, 1]

        element_pos = np.vstack(elements[:, 1])
        element_x, element_y = element_pos[:, 0], element_pos[:, 1]

        element_left, element_right = element_x, element_x + element_w
        element_top, element_bottom = element_y, element_y + element_h

        overlapping_with_e = np.logical_and(element_left < element_right, e_right > element_left)
        overlapping_with_e = np.logical_and(overlapping_with_e, e_top > element_bottom)
        overlapping_with_e = np.logical_and(overlapping_with_e, e_bottom < element_top)

        elements = elements[~overlapping_with_e]

        return elements

    def is_independent(self, sol, e):
        rect_idx, target_pos, rotated = e

        if sol.is_put[rect_idx]:
            return False

        new_sol = sol.copy()
        try:
            new_sol.put_rect(rect_idx, target_pos, rotated)
        except:
            return False

        return rects_correctly_placed(new_sol)

    def is_feasible(self, sol: RectanglePackingSolutionGreedy):
        rect_id_set = set(list(sol.rect_order))
        return len(rect_id_set) == self.num_rects \
               and np.all(sol.rect_order >= 0) \
               and np.all(sol.rect_order < self.num_rects)

    def get_empty_solution(self):
        solution = RectanglePackingSolutionGreedy(self)
        solution.reset()
        return solution

    # Deprecated
    def get_expansion(self, solution: RectanglePackingSolutionGreedy):
        return list(itertools.chain(*list(self.get_next_expansions(solution))))

    # Deprecated
    def get_next_expansions(self, solution: RectanglePackingSolutionGreedy):
        """Returns expansion (partial solutions obtained by appending an element) of the given (partial) solution."""
        ordered_by_occupancy = solution.box_occupancies.argsort()[::-1]

        # ---- Determine a good rect selection order ----
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


class RectanglePackingProblemGreedySmallestPositionStrategy(RectanglePackingProblemGreedyStrategy):
    def __init__(self, *args, **kwargs):
        super(RectanglePackingProblemGreedySmallestPositionStrategy, self).__init__(*args, **kwargs)

    def costs(self, e):
        _, target_pos, _ = e
        return sum(target_pos)


class RectanglePackingProblemGreedyLargestAreaStrategy(RectanglePackingProblemGreedyStrategy):
    def __init__(self, *args, **kwargs):
        super(RectanglePackingProblemGreedyLargestAreaStrategy, self).__init__(*args, **kwargs)

    def costs(self, e):
        rect_idx, _, _ = e
        return - np.prod(self.sizes[rect_idx])


def occupancy_heuristic(sol: RectanglePackingSolution):
    box_occupancies = sol.box_occupancies
    box_capacity = sol.problem.box_length ** 2
    cost = 1 + 0.9 * (box_occupancies[box_occupancies > 0] / box_capacity - 1) ** 3
    return np.sum(cost)


class RectanglePackingProblemGreedyLargestAreaSmallestPositionStrategy(RectanglePackingProblemGreedyStrategy):
    def __init__(self, *args, **kwargs):
        super(RectanglePackingProblemGreedyLargestAreaSmallestPositionStrategy, self).__init__(*args, **kwargs)

    def costs(self, e):
        rect_idx, target_pos, _ = e
        return - np.prod(self.sizes[rect_idx]) + sum(target_pos)


class RectanglePackingProblemGreedyUniformStrategy(RectanglePackingProblemGreedyStrategy):
    def __init__(self, *args, **kwargs):
        super(RectanglePackingProblemGreedyUniformStrategy, self).__init__(*args, **kwargs)

    def costs(self, e):
        return 0


def occupancy_heuristic(sol: RectanglePackingSolution):
    box_occupancies = sol.box_occupancies
    box_capacity = sol.problem.box_length ** 2
    cost = 1 + 0.9 * (box_occupancies[box_occupancies > 0] / box_capacity - 1) ** 3
    return np.sum(cost)


def rects_correctly_placed(sol: RectanglePackingSolution):
    return rects_respect_boxes(sol) and rects_are_disjoint(sol)


def rects_respect_boxes(sol: RectanglePackingSolution):
    """Checks if each rect lies inside a single box.
    Requires the solution to be built already."""
    sizes = sol.get_rect_sizes()
    locations_rel = sol.locations % sol.problem.box_length
    ends_rel = locations_rel + sizes
    return not np.any(ends_rel > sol.problem.box_length)


def rects_are_disjoint(sol: RectanglePackingSolution):
    """Checks if no rects intersect.
    Requires the solution to be built already."""
    return not np.any(sol.boxes_grid > 1)
