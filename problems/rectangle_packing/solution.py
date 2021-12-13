from problems.optimization import Solution
import copy
import numpy as np


class RectanglePackingSolution(Solution):
    def __init__(self, problem):
        super(RectanglePackingSolution, self).__init__()

        self.problem = problem

        self.locations = None
        self.rotations = None


class RectanglePackingSolutionGeometryBased(RectanglePackingSolution):
    def __init__(self, problem):
        super(RectanglePackingSolutionGeometryBased, self).__init__(problem)

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

        new_solution = RectanglePackingSolutionGeometryBased(self.problem)

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
