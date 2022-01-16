from abc import ABC

from problems.independence import IndependenceProblem
from problems.neighborhood import NeighborhoodProblem, OptProblem
from problems.rectangle_packing.solution import (
    RectanglePackingSolutionGeometryBased,
    RectanglePackingSolutionRuleBased,
    RectanglePackingSolutionGreedy,
    RectanglePackingSolution
)

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

        self.__heuristic = self.__box_occupancy_heuristic

    def objective_function(self, sol: RectanglePackingSolution):
        """Returns the number of boxes occupied in the current sol. Function symbol f.
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

    def is_relaxation_active(self):
        return False

    def is_relaxation_enabled(self):
        return False

    def toggle_relaxation(self, value):
        pass

    def get_instance_params(self):
        return (self.sizes,)

    def set_instance_params(self, sizes):
        self.sizes = sizes
        self.areas = self.sizes[:, 0] * self.sizes[:, 1]
        oversize = self.box_length // 2
        self.top_dogs = np.all(self.sizes > oversize, axis=1)  # "Platzhirsche"

        # Compute a lower bound for the minimum
        # "top dog" := a rectangle that requires an own box (no two top dogs fit together into one box)
        num_top_dogs = np.sum(self.top_dogs)
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
        widths = np.random.randint(w_min, w_max + 1, size=num_rects)
        heights = np.random.randint(h_min, h_max + 1, size=num_rects)
        sizes = np.stack([widths, heights], axis=1)

        self.set_instance_params(sizes)

    def is_optimal(self, sol: RectanglePackingSolution):
        """If True is returned, the solution is optimal
        (otherwise no assertion can be made)."""

        if not sol.is_complete():
            return False

        return self.objective_function(sol) <= self.minimum_lower_bound

    def place(self, rect_size, boxes_grid, selected_box_ids, box_coords, one_per_box=True):
        regions_to_place = np.lib.stride_tricks.sliding_window_view(boxes_grid[selected_box_ids],
                                                                    rect_size, axis=(1, 2))
        b, x, y = np.where(~np.any(regions_to_place, axis=(3, 4)))

        if len(b) == 0:
            return [], []

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

    def get_random_rect_selection_order(self):
        rect_ids = list(range(self.num_rects))
        np.random.shuffle(rect_ids)
        return rect_ids

    def get_rect_selection_order(self, box_occupancies, box2rects, occupancy_threshold=0.9,
                                 keep_top_dogs=False):
        """Returns a 'good' rectangle selection order."""

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

    def set_heuristic(self, heuristic_name):
        if heuristic_name == 'rectangle_count_heuristic':
            self.__heuristic = self.__rect_cnt_heuristic
        elif heuristic_name == 'box_occupancy_heuristic':
            self.__heuristic = self.__box_occupancy_heuristic
        elif heuristic_name == 'small_box_position_heuristic':
            self.__heuristic = self.__small_box_position_heuristic
        else:
            raise NotImplementedError

    def heuristic(self, sol: RectanglePackingSolution):
        return self.__heuristic(sol)

    def __rect_cnt_heuristic(self, sol: RectanglePackingSolution):
        """Depends on rectangle count per box."""
        if sol.move_pending:
            box_rect_cnts = sol.box_rect_cnts.copy()

            rect_idx, target_pos, rotated = sol.pending_move_params
            source_box_idx = sol.get_box_idx_by_rect_id(rect_idx)
            target_box_idx = sol.get_box_idx_by_pos(target_pos)

            box_rect_cnts[source_box_idx] -= 1
            box_rect_cnts[target_box_idx] += 1
        else:
            box_rect_cnts = sol.box_rect_cnts

        cost = 1 - 0.5 ** box_rect_cnts
        return np.sum(cost)

    def __box_occupancy_heuristic(self, sol: RectanglePackingSolution):
        """Penalizes comparably low occupied boxes more."""
        if sol.move_pending:
            box_occupancies = sol.box_occupancies.copy()

            rect_idx, target_pos, rotated = sol.pending_move_params
            source_box_idx = sol.get_box_idx_by_rect_id(rect_idx)
            target_box_idx = sol.get_box_idx_by_pos(target_pos)

            box_occupancies[source_box_idx] -= sol.problem.areas[rect_idx]
            box_occupancies[target_box_idx] += sol.problem.areas[rect_idx]
        else:
            box_occupancies = sol.box_occupancies

        box_capacity = self.box_length ** 2
        cost = 1 + 0.9 * (box_occupancies[box_occupancies > 0] / box_capacity - 1) ** 3
        return np.sum(cost)

    def __small_box_position_heuristic(self, sol: RectanglePackingSolution):
        # pos_sum = sol.locations.sum()
        # box_pos_sum = (sol.locations // self.box_length).sum()

        x, y = (sol.locations // self.box_length).T

        if sol.move_pending:
            rect_idx, target_pos, _ = sol.pending_move_params
            target_x, target_y = target_pos // self.box_length

            x[rect_idx] = target_x
            y[rect_idx] = target_y

        box_ids = x + self.box_length * y
        cost = np.sum(box_ids)
        return cost


class RectanglePackingProblemGeometryBased(RectanglePackingProblem, NeighborhoodProblem):
    def __init__(self, *args, **kwargs):
        super(RectanglePackingProblemGeometryBased, self).__init__(*args, **kwargs)
        self.allowed_overlap = 0.0
        self.penalty_factor = 0.0

        self.relaxation_enabled = False

    def update_relaxation(self, step):
        if not self.relaxation_enabled:
            return

        self.penalty_factor = 1e-3 * step ** 2
        self.allowed_overlap = np.exp(- 1e-1 * step)
        self.allowed_overlap = round(self.allowed_overlap, 2)

    def reset_relaxation(self):
        self.penalty_factor = 0.0
        if self.is_relaxation_enabled():
            self.allowed_overlap = 1.0
        else:
            self.allowed_overlap = 0.0

    def is_relaxation_active(self):
        return self.relaxation_enabled and self.allowed_overlap > 0

    def is_relaxation_enabled(self):
        return self.relaxation_enabled

    def toggle_relaxation(self, value: bool = None):
        if value is not None:
            self.relaxation_enabled = None
        else:
            self.relaxation_enabled = not self.relaxation_enabled

    def penalty(self, sol):
        if sol.move_pending:
            rect_idx, target_pos, rotated = sol.pending_move_params
            orig_pos = sol.locations[rect_idx]
            sol.apply_pending_move()

            boxes_grid = sol.boxes_grid.copy()

            sol.move_rect(rect_idx, orig_pos, rotated)
            sol.apply_pending_move()
            sol.move_rect(rect_idx, target_pos, rotated)
        else:
            boxes_grid = sol.boxes_grid

        penalty = np.sum(boxes_grid[boxes_grid > 1] - 1)
        return self.penalty_factor * penalty

    def heuristic(self, sol: RectanglePackingSolutionGeometryBased):
        h = super().heuristic(sol)

        if self.allowed_overlap == 0 or not self.relaxation_enabled:
            return h

        p = self.penalty(sol)

        return h + p

    def place(self, rect_size, boxes_grid, selected_box_ids, rectangle_fields=None, box2rects=None, box_coords=None,
              one_per_box=True, allowed_overlap=0.0):

        if allowed_overlap == 0:
            #  handle the simple case (can also be handles with the 'else' case but probably takes more time)
            regions_to_place = np.lib.stride_tricks.sliding_window_view(boxes_grid[selected_box_ids], rect_size,
                                                                        axis=(1, 2))
            b, x, y = np.where(~np.any(regions_to_place, axis=(3, 4)))

        else:
            # -------- quick filter for rectangles in the selected boxes -------
            # contained_rectangles = []
            # for box_id in selected_box_ids:
            #    contained_rectangles.extend(box2rects[box_id])
            # contained_rectangles = np.array(contained_rectangles)

            # (b, rects, L, L) -> (b, rects, x, y, w, h)
            regions_to_place_per_rect = np.lib.stride_tricks.sliding_window_view(rectangle_fields[selected_box_ids],
                                                                                 rect_size, axis=(2, 3))
            # regions_to_place_per_rect = np.lib.stride_tricks.sliding_window_view(rectangle_fields[selected_box_ids][:,contained_rectangles], rect_size, axis=(2, 3))

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
            # r2_areas = self.areas[contained_rectangles[r]]

            # focus on the pairwise maxima and compute ratios of overlaps
            max_areas = np.maximum(r1_area, r2_areas)
            overlap_ratios = np.divide(r_overlaps.reshape(1, -1), max_areas)

            # is the overlap not exceeding the allowed overlap?
            valid = overlap_ratios <= allowed_overlap
            # reshape to the convenient (b, x, y, r) shape
            valid = valid.reshape(r_overlaps.shape)

            # all (b, x, y) triplets are valid for which all r values are True (no rectangle invalidates this placement)
            b, x, y = np.where(np.all(valid, axis=3))

            # m = overlap_ratios.reshape(r_overlaps.shape)[b, x, y].max()
            # print("\nPLACE:", overlap_ratios.reshape(r_overlaps.shape)[b, x, y].max(), self.allowed_overlap)

        if len(b) == 0:
            return [], []

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

    def is_feasible(self, sol: RectanglePackingSolutionGeometryBased):
        if sol.move_pending:
            # Remember current configuration and apply pending move
            rect_idx, target_pos, rotated = sol.pending_move_params
            orig_pos = sol.locations[rect_idx]
            sol.apply_pending_move()

            feasible = rects_correctly_placed(sol)

            # Re-construct original sol
            sol.move_rect(rect_idx, orig_pos, rotated)
            sol.apply_pending_move()
            sol.move_rect(rect_idx, target_pos, rotated)

            return feasible
        else:
            return rects_correctly_placed(sol)

    def get_arbitrary_solution(self):
        """Returns a solution where each rectangle is placed into an own box (not rotated)."""
        if self.relaxation_enabled:
            locations = np.zeros((self.num_rects, 2), dtype=np.int)
        else:
            num_cols = int(np.ceil(np.sqrt(self.num_rects)))
            x_locations = np.arange(self.num_rects) % num_cols
            y_locations = np.arange(self.num_rects) // num_cols
            locations = np.stack([x_locations, y_locations], axis=1) * self.box_length
        rotations = np.zeros(self.num_rects, dtype=np.bool)
        sol = RectanglePackingSolutionGeometryBased(self)
        sol.set_solution(locations, rotations)
        return sol

    def get_neighborhood(self, sol: RectanglePackingSolutionGeometryBased):
        return list(itertools.chain(*list(self.get_next_neighbors(sol))))

    def get_next_neighbors(self, sol: RectanglePackingSolutionGeometryBased):
        """Returns all valid placing coordinates for all rectangles."""
        ordered_by_occupancy = sol.box_occupancies.argsort()[::-1]

        # ---- Preprocessing: Determine a good rect selection order ----
        if self.allowed_overlap > 0:
            rect_ids = self.get_random_rect_selection_order()
        else:
            rect_ids = self.get_rect_selection_order(sol.box_occupancies, sol.box2rects,
                                                     occupancy_threshold=0.9,
                                                     keep_top_dogs=False)

        # ---- Check placements using sliding window approach ----
        for rect_idx in rect_ids:
            # Select the n most promising boxes
            box_capacity = self.box_length ** 2
            max_occupancy = box_capacity - self.areas[rect_idx] * (1 - self.allowed_overlap)
            promising_boxes = sol.box_occupancies <= max_occupancy  # drop boxes which are too full
            sorted_box_ids = ordered_by_occupancy
            selected_box_ids = sorted_box_ids[promising_boxes[ordered_by_occupancy]]
            selected_box_ids = selected_box_ids[:MAX_CONSIDERED_BOXES]  # take at most a certain number of boxes

            sols = []

            for rotate in [False, True]:
                # Identify all locations which are allowed for placement
                relevant_locs, _ = self.place(
                    rect_size=self.sizes[rect_idx] if not rotate else self.sizes[rect_idx][::-1],
                    boxes_grid=sol.boxes_grid,
                    selected_box_ids=selected_box_ids,
                    rectangle_fields=sol.rectangle_fields,
                    box2rects=sol.box2rects,
                    box_coords=sol.box_coords,
                    allowed_overlap=self.allowed_overlap)

                # Prune abundant options
                relevant_locs = relevant_locs[:MAX_SELECTED_PLACINGS]

                # Generate new sols
                for loc in relevant_locs:
                    new_sol = sol.copy()
                    new_sol.move_rect(rect_idx, loc, rotate)
                    # assert self.is_feasible(new_sol)
                    sols += [new_sol]

            # print("generating %d neighbors took %.3f s" % (len(sols), time.time() - t))
            yield sols


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
                target_order_pos = int((self.num_rects - 1) * (1 - (rect_area - min_area) /
                                                               (max_area - min_area)))
                # target_order_pos = 0
            new_sol = sol.copy()
            new_sol.move_rect_to_order_pos(rect_idx, target_order_pos)
            yield [new_sol]

    def get_arbitrary_solution(self):
        sol = RectanglePackingSolutionRuleBased(self)
        sol.reset()
        sol.rect_order = np.arange(self.num_rects)
        return sol

    def objective_function(self, sol: RectanglePackingSolutionRuleBased):
        if not sol.all_rects_put():
            self.put_all_rects(sol)

        return super().objective_function(sol)

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
        return super().heuristic(sol)

    def is_feasible(self, sol: RectanglePackingSolutionRuleBased):
        rect_id_set = set(list(sol.rect_order))
        return len(rect_id_set) == self.num_rects \
               and np.all(sol.rect_order >= 0) \
               and np.all(sol.rect_order < self.num_rects)


class RectanglePackingProblemGreedy(RectanglePackingProblem, IndependenceProblem):
    def __init__(self, *args, **kwargs):
        super(RectanglePackingProblemGreedy, self).__init__(*args, **kwargs)

        self.__costs = self.__smallest_position_costs
        self.strategy_name = 'smallest_position_costs_strategy'

    def __smallest_position_costs(self, e):
        _, target_pos, _ = e
        return sum(target_pos)

    def __largest_area_costs(self, e):
        rect_idx, _, _ = e
        return - np.prod(self.sizes[rect_idx])

    def __smallest_position_plus_largest_area_costs(self, e):
        rect_idx, target_pos, _ = e
        return - np.prod(self.sizes[rect_idx]) + sum(target_pos)

    def __lowest_box_id_costs(self, e):
        _, target_pos, _ = e
        x, y = target_pos // self.box_length
        box_id = x + self.box_length * y
        return box_id

    def __uniform_costs(self, e):
        return 0

    def set_strategy(self, strategy_name):
        if strategy_name == 'smallest_position_costs':
            self.__costs = self.__smallest_position_costs
        elif strategy_name == 'largest_area_costs':
            self.__costs = self.__largest_area_costs
        elif strategy_name == 'smallest_position_plus_largest_area_costs':
            self.__costs = self.__smallest_position_plus_largest_area_costs
        elif strategy_name == 'uniform_costs':
            self.__costs = self.__uniform_costs
        elif strategy_name == 'lowest_box_id_costs':
            self.__costs = self.__lowest_box_id_costs
        else:
            raise NotImplementedError

        self.strategy_name = strategy_name

    def costs(self, elements):
        return self.__costs(elements)

    def get_elements(self, sol):
        """Returns a list of elements"""

        elements = []
        for rect_idx in range(self.num_rects):

            for rotate in [False, True]:
                # Identify all locations which are allowed for placement
                relevant_locs, _ = self.place(
                    rect_size=self.sizes[rect_idx] if not rotate else self.sizes[rect_idx][::-1],
                    boxes_grid=sol.boxes_grid,
                    selected_box_ids=np.arange(self.num_rects, dtype=int),
                    box_coords=sol.box_coords,
                    one_per_box=False)
                # Prune abundant options
                # n_choices = min(relevant_locs.shape[0], 64)
                # relevant_locs = relevant_locs[np.random.choice(relevant_locs.shape[0], n_choices, replace=False)]
                # relevant_locs = relevant_locs[:n_choices]

                # add triplets (rect_idx, location, rotation) to elements
                elements.extend([(rect_idx, loc, rotate) for loc in relevant_locs])

        # shuffle to not influence the algorithm by the generation order of placement
        np.random.shuffle(elements)
        return elements

    def filter_elements(self, sol, elements, e):
        return list(filter(lambda x: x[0] != e[0], elements))

    def is_independent(self, sol, element):
        rect_idx, target_pos, rotated = element

        if sol.is_put[rect_idx]:
            return False

        new_sol = sol.copy()
        try:
            new_sol.put_rect(rect_idx, target_pos, rotated)
        except:
            return False

        return rects_correctly_placed(new_sol)

    def get_empty_solution(self):
        sol = RectanglePackingSolutionGreedy(self)
        sol.reset()
        return sol

    def is_feasible(self, sol):
        raise NotImplementedError


class RectanglePackingProblemGreedyFast(RectanglePackingProblem):
    def __init__(self, *args, **kwargs):
        super(RectanglePackingProblemGreedyFast, self).__init__(*args, **kwargs)

        self.__costs = self.__largest_rect_costs
        self.strategy_name = 'largest_rectangle_first'

    def get_elements(self):
        return list(range(self.num_rects))

    def set_strategy(self, strategy_name):
        if strategy_name == 'largest_rectangle_first':
            self.__costs = self.__largest_rect_costs
        elif strategy_name == 'smallest_rectangle_first':
            self.__costs = self.__smallest_rect_costs
        elif strategy_name == 'uniform_rectangle':
            self.__costs = self.__uniform_rect_costs
        else:
            raise NotImplementedError

        self.strategy_name = strategy_name

    def costs(self, rect_ids):
        return self.__costs(rect_ids)

    def __largest_rect_costs(self, rect_idx):
        return - self.areas[rect_idx]

    def __smallest_rect_costs(self, rect_idx):
        return self.areas[rect_idx]

    def __uniform_rect_costs(self, rect_idx):
        return 0

    def get_empty_solution(self):
        sol = RectanglePackingSolutionGreedy(self)
        sol.reset()
        return sol

    def get_expansion(self, sol: RectanglePackingSolutionGreedy):
        """Returns expansion (partial sols obtained by appending an element) of the given (partial) sol."""
        ordered_by_occupancy = sol.box_occupancies.argsort()[::-1]
        ordered_by_idx = ordered_by_occupancy.sort()

        box_capacity = self.box_length ** 2

        # ---- Apply Greedy Sorting Strategy ----
        rect_idxs = sol.get_remaining_elements()
        next_rect_idx = sorted(rect_idxs, key=self.costs)[0]

        # Select the n most promising boxes
        max_occupancy = box_capacity - self.areas[next_rect_idx]
        promising_boxes = sol.box_occupancies <= max_occupancy  # drop boxes which are too full

        sorted_box_ids = ordered_by_occupancy
        selected_box_ids = sorted_box_ids[promising_boxes[ordered_by_occupancy]]

        for rotate in [False, True]:

            # Identify all locations which are allowed for placement
            relevant_locs, _ = self.place(
                rect_size=self.sizes[next_rect_idx] if not rotate else self.sizes[next_rect_idx][::-1],
                boxes_grid=sol.boxes_grid,
                selected_box_ids=selected_box_ids,
                box_coords=sol.box_coords)

            if len(relevant_locs) > 0:
                loc = relevant_locs[0]

                new_sol = sol.copy()
                new_sol.put_rect(next_rect_idx, loc, rotate)
                # assert self.is_feasible(new_sol)

                return new_sol

    def is_feasible(self, sol):
        raise NotImplementedError


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
