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

    def is_feasible(self, sol: RectanglePackingSolutionGeometryBased):
        if sol.move_pending:
            # Remember current configuration and apply pending move
            rect_idx, target_pos, rotated = sol.pending_move_params
            orig_pos, orig_rotated = sol.locations[rect_idx], sol.rotations[rect_idx]
            sol.apply_pending_move()

            feasible = rects_correctly_placed(sol)

            # Re-construct original solution
            sol.move_rect(rect_idx, orig_pos, orig_rotated)
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
        return self.__box_occupancy_heuristic(sol) # + self.penalty(sol)

    def penalty(self, sol: RectanglePackingSolutionOverlap):
       return np.sum(sol.boxes_grid[sol.boxes_grid > 1] - 1) * self.penalty_factor

    def __box_occupancy_heuristic(self, sol: RectanglePackingSolutionOverlap):
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

    def is_feasible(self, sol: RectanglePackingSolutionOverlap):
        if sol.move_pending:
            # Remember current configuration and apply pending move
            rect_idx, target_pos, rotated = sol.pending_move_params
            orig_pos, orig_rotated = sol.locations[rect_idx], sol.rotations[rect_idx]
            sol.apply_pending_move()

            feasible = rects_correctly_placed(sol)

            # Re-construct original solution
            sol.move_rect(rect_idx, orig_pos, orig_rotated)
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
        sol = RectanglePackingSolutionOverlap(self)
        sol.set_solution(locations, rotations)
        return sol

    def get_neighborhood(self, solution: RectanglePackingSolutionOverlap):
        return list(itertools.chain(*list(self.get_next_neighbors(solution))))

    def get_next_neighbors(self, solution: RectanglePackingSolutionOverlap):
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

                """
                Note: Why do overlaps appear although this placement method is commented out
                # Identify all locations which are allowed for placement
                relevant_locs, _ = self.__place_with_overlap(rect_size=self.sizes[rect_idx] if not rotate else self.sizes[rect_idx][::-1],
                                              boxes_grid=solution.boxes_grid,
                                              selected_box_ids=selected_box_ids,
                                              box_coords=solution.box_coords.
                                              box2rects=sol.box2rects, locations=sol.locations, rotated=sol.rotations)
                """

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

    def __rect_overlap_tolerable(self, sol: RectanglePackingSolutionOverlap):
        # Identify boxes with overlap
        overlap_box_ids = list(set(list(np.where(sol.boxes_grid > 1)[0])))

        sizes = sol.get_rect_sizes()

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
                    #if overlap_area > 0:
                    #    print("overlap_area:", overlap_area)

                    total_area = max(sol.problem.areas[rect_1], sol.problem.areas[rect_2])
                    overlap = overlap_area / total_area
                    if overlap > self.allowed_overlap:
                        return False

        return True

    def __place_with_overlap(self, rect_size, boxes_grid, selected_box_ids, box_coords, box2rects, locations, rotated):
        regions_to_place = np.lib.stride_tricks.sliding_window_view(boxes_grid[selected_box_ids],
                                                                    rect_size, axis=(1, 2))

        if self.allowed_overlap == 0:
            b, x, y = np.where(~np.any(regions_to_place, axis=(3, 4)))
        else:
            assert self.allowed_overlap > 0 and self.allowed_overlap <= 1

            # Note: An overlap of the placed rectangle R1 and an already existing rectangle R2
            # is "valid" iff |overlap(R1, R2)| / max(area(R1), area(R2)) <= OVERLAP

            # Get the area of the rectangle that is to be placed
            r1_area = np.prod(rect_size)
            # Precompute the rectangle areas
            rect_areas = self.get_rect_areas()

            # Compute the number of potential overlaps for each of the candidate regions
            num_overlaps_per_region = np.sum(regions_to_place > 0, axis=(3, 4))
            # Determine the regions that have a valid overlap regarding R1
            ratio = np.divide(num_overlaps_per_region, r1_area)
            overlap_valid_for_r1 = np.less_equal(ratio, self.allowed_overlap) # shape: (b, x, y)

            # Get the fields in each region that already hold a rectangle
            b, x, y, _, _ = np.where(regions_to_place > 0)

            # Determine the field coordinates of the region (left, top)
            b_ids = selected_box_ids[b]
            region_locs = box_coords[b_ids] * self.box_length + np.stack([x, y], axis=1)

            overlap_valid_for_r2 = np.ones_like(overlap_valid_for_r1, dtype=bool)

            # Go over all of the regions and check if there is already a rectangle
            # that is covered by it and that is too small to allow an overlap
            for box_idx, box_id, region_loc, region_x, region_y in zip(b, b_ids, region_locs, x, y):

                # If this region is already valid for R1 then it does not have to be valid for R2
                if overlap_valid_for_r1[box_idx, region_x, region_y]:
                    continue

                # Compute (left, top) and (right, bottom) coordinates of region
                l_region, t_region = region_loc
                r_region = l_region + rect_size[0]
                b_region = t_region + rect_size[1]

                # Go over all the contained rectangles of the box
                for rect_id in box2rects[box_id]:

                    # If the area of R2 is not larger than the one of R1
                    # then R2 has nothing to say
                    r2_area = self.areas[rect_id]
                    if r2_area <= r1_area:
                        continue

                    # Compute (left, top) and (right, bottom) coordinates of the rectangle
                    rect_x, rect_y = locations[rect_id]
                    rect_w, rect_h = self.sizes[rect_id]
                    if rotated[rect_id]:
                        rect_w, rect_h = rect_h, rect_w

                    l_rect, t_rect = rect_x, rect_y
                    r_rect, b_rect = l_rect + rect_w, t_rect + rect_h

                    # Compute overlap area of region and the respective rectangle
                    x_overlap = max(r_rect - l_region, 0) - max(l_rect - l_region, 0) - max(r_rect - r_region, 0) + max(l_rect - r_region, 0)
                    y_overlap = max(b_rect - t_region, 0) - max(t_rect - t_region, 0) - max(b_rect - b_region, 0) + max(t_rect - b_region, 0)

                    overlap_area = x_overlap * y_overlap

                    # Invalidate the region if it violates the allowed overlap for R2
                    if overlap_area / r2_area > self.allowed_overlap:
                        overlap_valid_for_r2[box_idx, region_x, region_y] = False

            # Region is valid if it valid for R1 or for R2
            b, x, y = np.where(np.logical_or(overlap_valid_for_r1, overlap_valid_for_r2))

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
