from problems.optimization import Solution
import copy
import numpy as np

class RectanglePackingSolution(Solution):
    def __init__(self, problem):
        super(RectanglePackingSolution, self).__init__()

        self.move_pending = False
        self.problem = problem

        self.locations = None
        self.rotations = None
        self.is_put = None
        self.last_put_rect = None

        self.box_ids = None
        self.box_coords = None

        self.box_occupancies = None
        self.box_rect_cnts = None
        self.box2rects = None

        self.boxes_grid = None


    def reset(self, locations=None, rotations=None):
        if locations is not None:
            self.locations = locations
        else:
            self.locations = np.zeros((self.problem.num_rects, 2), dtype=np.int)

        if rotations is not None:
            self.rotations = rotations
        else:
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

    def build(self, locations, rotations):
        """Builds all the box information (such as self.boxes_grid) from rect
        locations and rect rotations."""
        self.reset(locations, rotations)
        for rect_idx in range(self.problem.num_rects):
            self.put_rect(rect_idx, self.locations[rect_idx], self.rotations[rect_idx], update_ids=True)

    def put_rect(self, rect_idx, target_pos, rotated, update_ids=False):
        """Puts the specified rect to the given target position, rotated accordingly.
        Ignores whether this put action creates an infeasible solution."""

        assert not self.is_put[rect_idx]

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
        self.last_put_rect = rect_idx

    def remove_rect(self, rect_idx):
        assert self.is_put[rect_idx]

        box_idx = self.get_box_idx_by_rect_id(rect_idx)

        self.box_occupancies[box_idx] -= self.problem.areas[rect_idx]

        self.box_rect_cnts[box_idx] -= 1

        self.box2rects[box_idx].remove(rect_idx)

        x, y = self.locations[rect_idx] % self.problem.box_length
        w, h = self.problem.sizes[rect_idx]
        if self.rotations[rect_idx]:
            w, h = h, w
        self.boxes_grid[box_idx, x:x + w, y:y + h] -= 1

        self.is_put[rect_idx] = False

    def remove_rects(self, rect_ids):
        for rect_idx in rect_ids:
            self.remove_rect(rect_idx)
        self.is_put[rect_ids] = False

    def move_rect(self, rect_idx, target_pos, rotated):
        self.remove_rect(rect_idx)
        self.put_rect(rect_idx, target_pos, rotated, update_ids=True)

    def get_rect_sizes(self):
        """Returns the rect sizes, taking rotations into account."""
        _, rotations = self.locations, self.rotations

        sizes = self.problem.sizes.copy()
        sizes[rotations, 0] = self.problem.sizes[rotations, 1]
        sizes[rotations, 1] = self.problem.sizes[rotations, 0]

        return sizes

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

    def get_occupied_box_ids(self) -> np.array:
        return np.where(self.box_rect_cnts > 0)[0]

    def get_empty_box_ids(self) -> np.array:
        return np.where(self.box_rect_cnts == 0)[0]

    def all_rects_put(self):
        return np.all(self.is_put)

    def copy(self, true=True):
        duplicate = type(self)(self.problem)
        if true:
            true_copy(self, duplicate)
        else:
            ref_copy(self, duplicate)
        return duplicate

    def is_complete(self):
        return np.all(self.is_put)

def true_copy(from_sol: RectanglePackingSolution, to_sol: RectanglePackingSolution):
    to_sol.locations = from_sol.locations.copy()
    to_sol.rotations = from_sol.rotations.copy()
    to_sol.is_put = from_sol.is_put.copy()
    to_sol.box_ids = from_sol.box_ids.copy()
    to_sol.box_coords = from_sol.box_coords.copy()
    to_sol.box_occupancies = from_sol.box_occupancies.copy()
    to_sol.box_rect_cnts = from_sol.box_rect_cnts.copy()
    to_sol.box2rects = copy.deepcopy(from_sol.box2rects)
    to_sol.boxes_grid = from_sol.boxes_grid.copy()
    to_sol.last_put_rect = from_sol.last_put_rect


def ref_copy(from_sol: RectanglePackingSolution, to_sol: RectanglePackingSolution):
    to_sol.locations = from_sol.locations
    to_sol.rotations = from_sol.rotations
    to_sol.is_put = from_sol.is_put
    to_sol.box_ids = from_sol.box_ids
    to_sol.box_coords = from_sol.box_coords
    to_sol.box_occupancies = from_sol.box_occupancies
    to_sol.box_rect_cnts = from_sol.box_rect_cnts
    to_sol.box2rects = from_sol.box2rects
    to_sol.boxes_grid = from_sol.boxes_grid
    to_sol.last_put_rect = from_sol.last_put_rect


class RectanglePackingSolutionGeometryBased(RectanglePackingSolution):
    def __init__(self, problem):
        super(RectanglePackingSolutionGeometryBased, self).__init__(problem)

        self.standalone = True  # If False, attributes of this class require deepcopy before any modification
        self.pending_move_params = None
        self.rectangle_fields = None

    def reset(self, locations=None, rotations=None):
        super().reset(locations, rotations)
        self.rectangle_fields = np.zeros((self.problem.num_rects, self.problem.num_rects, self.problem.box_length, self.problem.box_length), dtype=bool)


    def put_rect(self, rect_idx, target_pos, rotated, update_ids=False):
        """Puts the specified rect to the given target position, rotated accordingly.
        Ignores whether this put action creates an infeasible solution."""

        assert not self.is_put[rect_idx]

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

        self.rectangle_fields[box_idx, rect_idx, :, :] = 0
        self.rectangle_fields[box_idx, rect_idx, x:x + w, y:y + h] = 1

    def remove_rect(self, rect_idx):
        assert self.is_put[rect_idx]

        box_idx = self.get_box_idx_by_rect_id(rect_idx)

        self.box_occupancies[box_idx] -= self.problem.areas[rect_idx]

        self.box_rect_cnts[box_idx] -= 1

        self.box2rects[box_idx].remove(rect_idx)

        x, y = self.locations[rect_idx] % self.problem.box_length
        w, h = self.problem.sizes[rect_idx]
        if self.rotations[rect_idx]:
            w, h = h, w
        self.boxes_grid[box_idx, x:x + w, y:y + h] -= 1

        self.is_put[rect_idx] = False

        self.rectangle_fields[box_idx, rect_idx, :, :] = 0

    def set_solution(self, locations, rotations):
        self.build(locations.copy(), rotations.copy())

    def move_rect(self, rect_idx, target_pos, rotated):
        """Assumes that this action leads to a feasible solution."""
        if self.move_pending:
            raise ValueError("Cannot add another pending move if there is already one.")
        self.move_pending = True
        self.pending_move_params = rect_idx, target_pos, rotated

    def apply_pending_move(self):
        if not self.move_pending:
            return

        rect_idx, target_pos, rotated = self.pending_move_params
        super().move_rect(rect_idx, target_pos, rotated)

        self.move_pending = False
        self.pending_move_params = None

    def copy(self, true=False):
        if self.move_pending:
            self.apply_pending_move()

        new_solution = super().copy(true)
        new_solution.standalone = true

        if true:
            new_solution.rectangle_fields = self.rectangle_fields.copy()
        else:
            new_solution.rectangle_fields = self.rectangle_fields

        return new_solution

    def make_standalone(self):
        if not self.standalone:
            true_copy(self, self)
            self.standalone = True


class RectanglePackingSolutionRuleBased(RectanglePackingSolution):
    def __init__(self, problem):
        super(RectanglePackingSolutionRuleBased, self).__init__(problem)
        self.rect_order = None

    def set_rect_selection_order(self, rect_selection_order):
        self.rect_order = rect_selection_order

    def move_rect_to_order_pos(self, rect_idx, target_order_pos):
        # Update rect order
        source_order_pos = np.where(self.rect_order == rect_idx)[0][0]
        if source_order_pos > target_order_pos:
            self.rect_order[target_order_pos + 1:source_order_pos + 1] = \
                self.rect_order[target_order_pos:source_order_pos]
        elif source_order_pos < target_order_pos:
            self.rect_order[source_order_pos:target_order_pos] = \
                self.rect_order[source_order_pos + 1:target_order_pos + 1]
        self.rect_order[target_order_pos] = rect_idx

        # Remove touched rects from currently built solution
        moved_rect_ids = self.rect_order[min(source_order_pos, target_order_pos):]
        self.remove_rects(moved_rect_ids)

    def copy(self, true=True):
        duplicate = super().copy(True)
        duplicate.rect_order = self.rect_order.copy()
        return duplicate



class RectanglePackingSolutionGreedy(RectanglePackingSolution):
    def __init__(self, problem):
        super(RectanglePackingSolutionGreedy, self).__init__(problem)

    def get_remaining_elements(self):
        return np.where(~self.is_put)[0]

    def add_element(self, e):
        rect_idx, target_pos, rotated = e
        self.put_rect(rect_idx, target_pos, rotated)
