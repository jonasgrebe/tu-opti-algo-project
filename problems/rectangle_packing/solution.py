from problems.optimization import Solution
import copy
import numpy as np


class RectanglePackingSolution(Solution):
    def __init__(self, problem):
        super(RectanglePackingSolution, self).__init__()

        self.problem = problem

        self.locations = None
        self.rotations = None
        self.is_put = None

        self.box_ids = None
        self.box_coords = None

        self.box_occupancies = None
        self.box_rect_cnts = None
        self.box2rects = None

        self.boxes_grid = None

    def put_rect(self, rect_idx, target_pos, rotated, update_ids=False):
        box_idx = self.get_box_idx_by_pos(target_pos, update_ids)

        self.box_occupancies[box_idx] += self.problem.areas[rect_idx]

        self.box_rect_cnts[box_idx] += 1

        self.box2rects[box_idx].append(rect_idx)

        x, y = target_pos % self.problem.box_length
        w, h = self.problem.sizes[rect_idx]
        if rotated:
            w, h = h, w
        self.boxes_grid[box_idx, x:x + w, y:y + h] += 1

        self.locations[rect_idx] = target_pos
        self.rotations[rect_idx] = rotated
        self.is_put[rect_idx] = True

    def remove_rect(self, rect_idx):
        box_idx = self.get_box_idx_by_rect_id(rect_idx)

        if box_idx is None:
            return RuntimeError

        self.box_occupancies[box_idx] -= self.problem.areas[rect_idx]

        self.box_rect_cnts[box_idx] -= 1

        self.box2rects[box_idx].remove(rect_idx)

        x, y = self.locations[rect_idx] % self.problem.box_length
        w, h = self.problem.sizes[rect_idx]
        if self.rotations[rect_idx]:
            w, h = h, w
        self.boxes_grid[box_idx, x:x + w, y:y + h] -= 1

        self.is_put[rect_idx] = False

    def get_box_idx_by_rect_id(self, rect_idx):
        box = tuple(self.locations[rect_idx] // self.problem.box_length)
        return self.get_box_idx(box)

    def get_box_idx_by_pos(self, pos, update_ids=False):
        box = tuple(pos // self.problem.box_length)
        box_idx = self.get_box_idx(box)

        # If box doesn't exist, give it an ID from an empty box
        if box_idx is None:
            box_idx = np.where(self.box_rect_cnts == 0)[0][0]

            if update_ids:
                box_to_replace = tuple(self.box_coords[box_idx])
                self.box_ids.pop(box_to_replace)
                self.box_ids[box] = box_idx
                self.box_coords[box_idx] = box

        return box_idx

    def get_box_idx(self, box):
        if box in self.box_ids.keys():
            return self.box_ids[box]
        else:
            return None


class RectanglePackingSolutionGeometryBased(RectanglePackingSolution):
    def __init__(self, problem):
        super(RectanglePackingSolutionGeometryBased, self).__init__(problem)

        self.standalone = True  # If False, attributes of this class require deepcopy before any modification
        self.move_pending = False
        self.pending_move_params = None

    def set_solution(self, locations, rotations):
        self.locations = locations.copy()
        self.rotations = rotations.copy()
        self.is_put = np.ones(self.problem.num_rects, dtype=np.bool)

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

        self.remove_rect(rect_idx)
        self.put_rect(rect_idx, target_pos, rotated, update_ids=True)

        self.move_pending = False
        self.pending_move_params = None

    def copy(self):
        if self.move_pending:
            self.apply_pending_move()

        new_solution = RectanglePackingSolutionGeometryBased(self.problem)

        new_solution.locations = self.locations
        new_solution.rotations = self.rotations
        new_solution.is_put = self.is_put
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
            self.is_put = self.is_put.copy()
            self.box_ids = self.box_ids.copy()
            self.box_coords = self.box_coords.copy()
            self.box_occupancies = self.box_occupancies.copy()
            self.box_rect_cnts = self.box_rect_cnts.copy()
            self.box2rects = copy.deepcopy(self.box2rects)
            self.boxes_grid = self.boxes_grid.copy()
            self.standalone = True


class RectanglePackingSolutionRuleBased(RectanglePackingSolution):
    def __init__(self, problem):
        super(RectanglePackingSolutionRuleBased, self).__init__(problem)

        self.rect_order = None
        self.placed = False

    def reset(self):
        self.locations = np.zeros((self.problem.num_rects, 2), dtype=np.int)
        self.rotations = np.zeros(self.problem.num_rects, dtype=np.bool)
        self.is_put = np.zeros(self.problem.num_rects, dtype=np.bool)

        self.box_coords = np.zeros((self.problem.num_rects, 2), dtype=np.int)
        num_cols = int(np.ceil(np.sqrt(self.problem.num_rects)))
        self.box_coords[:, 0] = np.arange(self.problem.num_rects) % num_cols
        self.box_coords[:, 1] = np.arange(self.problem.num_rects) // num_cols
        self.box_ids = {tuple(box): idx for idx, box in enumerate(self.box_coords)}

        self.box_occupancies = np.zeros(self.problem.num_rects, dtype=np.int)
        self.box_rect_cnts = np.zeros(self.problem.num_rects, dtype=np.int)
        self.box2rects = {idx: [] for idx in range(self.problem.num_rects)}

        self.boxes_grid = np.zeros((self.problem.num_rects,
                                    self.problem.box_length,
                                    self.problem.box_length), dtype=np.int)

    def set_rect_selection_order(self, rect_selection_order):
        self.rect_order = rect_selection_order

    def move_rect_to_order_pos(self, rect_idx, target_order_pos):
        source_order_pos = np.where(self.rect_order == rect_idx)[0][0]
        if source_order_pos > target_order_pos:
            self.rect_order[target_order_pos + 1:source_order_pos + 1] = \
                self.rect_order[target_order_pos:source_order_pos]
        elif source_order_pos < target_order_pos:
            self.rect_order[source_order_pos:target_order_pos] = \
                self.rect_order[source_order_pos + 1:target_order_pos + 1]
        self.rect_order[target_order_pos] = rect_idx
        self.placed = False
        self.is_put[min(rect_idx, target_order_pos):] = False

    def copy(self):
        duplicate = RectanglePackingSolutionRuleBased(self.problem)

        duplicate.rect_order = self.rect_order.copy()

        duplicate.locations = self.locations.copy()
        duplicate.rotations = self.rotations.copy()
        duplicate.is_put = self.is_put.copy()
        duplicate.box_ids = self.box_ids.copy()
        duplicate.box_coords = self.box_coords.copy()
        duplicate.box_occupancies = self.box_occupancies.copy()
        duplicate.box_rect_cnts = self.box_rect_cnts.copy()
        duplicate.box2rects = copy.deepcopy(self.box2rects)
        duplicate.boxes_grid = self.boxes_grid.copy()

        return duplicate
