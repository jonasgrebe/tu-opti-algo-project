import time
import pygame
import pygame_menu
import threading
import numpy as np
import json
import copy

from algos import local_search, greedy_search
from gui import BaseGUI
from problems.rectangle_packing.problem import RectanglePackingProblemGeometryBased, RectanglePackingSolutionGeometryBased


ZOOM_STEP_FACTOR = 1.1
ANIM_SPEED = 0.1  # sec


class RectanglePackingGUI(BaseGUI):
    def __init__(self):
        super().__init__()

        # Problem constants
        self.problem_config = dict(box_length=8, num_rects=32, w_min=1, w_max=8, h_min=1, h_max=8)
        self.problem = None
        self.init_sol = None

        self.current_sol = None
        self.rect_dims = None

        # GUI constants
        self.running = True
        self.dragging_camera = False
        self.old_mouse_pos = None
        self.selected_rect_idx = None
        self.selection_rotated = False

        with open("gui/config.json") as json_data_file:
            self.config = json.load(json_data_file)

        self.render_thread = threading.Thread(target=self.__run)
        self.render_thread.start()

        self.search = local_search  # the search algorithm routine
        self.is_searching = False
        self.is_paused = False
        self.search_thread = None
        self.search_start_time = None

    @property
    def colors(self):
        return self.config['colors']

    @property
    def field_size(self):
        return np.round(self.config['field_size'] * self.zoom_level)

    def __init_gui(self):
        pygame.init()
        pygame.display.set_caption("Rectangle Packing")

        pygame.font.init()
        self.font = pygame.font.SysFont(self.config['font'], self.config['font_size'])

        self.screen = pygame.display.set_mode((self.config['window_width'], self.config['window_height']),
                                              pygame.RESIZABLE)

        self.area_width = self.screen.get_width()
        self.area_height = self.screen.get_height()
        self.scr_marg_left = 0
        self.scr_marg_top = 0

        self.cam_pos = np.array([0, 0])
        self.zoom_level = 1.0

        self.frame_times = np.zeros(2000, dtype=np.float)

    def stop_search(self):
        # not private because search algorithm shall invoke it as well
        self.search_stop_time = time.time()
        self.is_searching = False

        btn_search = self.menu.get_widget('run_search')
        btn_search.set_title('Run Search')

        btn_configure = self.menu.get_widget('configure_problem')
        btn_configure.readonly = False


    def __setup_menu(self):

        def button_onmouseover(w) -> None:
            w.set_background_color((100, 100, 100))

        def button_onmouseleave(w) -> None:
            w.set_background_color((0, 0, 0))

        theme = pygame_menu.Theme(
            background_color=pygame_menu.themes.TRANSPARENT_COLOR,
            title=False,
            widget_font=pygame_menu.font.FONT_FIRACODE,
            widget_font_color=(255, 255, 255),
            widget_margin=(0, 15),
            widget_alignment=pygame_menu.locals.ALIGN_RIGHT,
            widget_selection_effect=pygame_menu.widgets.NoneSelection()
        )
        self.menu = pygame_menu.Menu(
            width=self.screen.get_width(),
            height=self.screen.get_height(),
            mouse_motion_selection=True,
            theme=theme,
            columns=2,
            rows=10,
            title='',
        )

        self.mini_menu = pygame_menu.Menu(
            width=self.screen.get_width(),
            height=self.screen.get_height(),
            mouse_motion_selection=True,
            theme=theme,
            title='Problem Configuration',
        )

        def rangeslider_box_length_onchange(s, *args) -> None:
            rangeslider_box_length = self.mini_menu.get_widget('rangeslider_box_length')
            box_length = int(rangeslider_box_length.get_value())
            self.__update_problem_config({'box_length': box_length})

        rangeslider_box_length = self.mini_menu.add.range_slider(
            'L',
            rangeslider_id='rangeslider_box_length',
            default=self.problem_config['box_length'],
            range_values=(3, 32),
            increment=1,
            onchange=rangeslider_box_length_onchange,
            font_size=20,
            shadow_width=10,
            background_color=(0, 0, 0)
        )

        def rangeslider_num_rects_onchange(s, *args) -> None:
            rangeslider_num_rects = self.mini_menu.get_widget('rangeslider_num_rects')
            num_rects = int(rangeslider_num_rects.get_value())
            self.__update_problem_config({'num_rects': num_rects})

        rangeslider_num_rects = self.mini_menu.add.range_slider(
            '# rects',
            rangeslider_id='rangeslider_num_rects',
            default=self.problem_config['num_rects'],
            range_values=(1, 1000),
            increment=1,
            onchange=rangeslider_num_rects_onchange,
            font_size=20,
            shadow_width=10,
            background_color=(0, 0, 0)
        )

        self.mini_menu.add.vertical_margin(40)

        def rangeslider_w_min_onchange(s, *args) -> None:
            rangeslider_w_min = self.mini_menu.get_widget('rangeslider_w_min')
            w_min = int(rangeslider_w_min.get_value())
            self.__update_problem_config({'w_min': w_min})

        rangeslider_w_min = self.mini_menu.add.range_slider(
            'w_min',
            rangeslider_id='rangeslider_w_min',
            default=self.problem_config['w_min'],
            range_values=(1, self.problem_config['w_max']),
            increment=1,
            onchange=rangeslider_w_min_onchange,
            font_size=20,
            shadow_width=10,
            background_color=(0, 0, 0)
        )

        def rangeslider_w_max_onchange(s, *args) -> None:
            rangeslider_w_max = self.mini_menu.get_widget('rangeslider_w_max')
            w_max = int(rangeslider_w_max.get_value())
            self.__update_problem_config({'w_max': w_max})

        rangeslider_w_max = self.mini_menu.add.range_slider(
            'w_max',
            rangeslider_id='rangeslider_w_max',
            default=self.problem_config['w_max'],
            range_values=(self.problem_config['w_min'] + 1, self.problem_config['box_length']),
            increment=1,
            onchange=rangeslider_w_max_onchange,
            font_size=20,
            shadow_width=10,
            background_color=(0, 0, 0)
        )

        def rangeslider_h_min_onchange(s, *args) -> None:
            rangeslider_h_min = self.mini_menu.get_widget('rangeslider_h_min')
            h_min = int(rangeslider_h_min.get_value())
            self.__update_problem_config({'h_min': h_min})

        self.mini_menu.add.vertical_margin(20)

        rangeslider_h_min = self.mini_menu.add.range_slider(
            'h_min',
            rangeslider_id='rangeslider_h_min',
            default=self.problem_config['h_min'],
            range_values=(1, self.problem_config['h_max']),
            increment=1,
            onchange=rangeslider_h_min_onchange,
            font_size=20,
            shadow_width=10,
            background_color=(0, 0, 0)
        )

        def rangeslider_h_max_onchange(s, *args) -> None:
            rangeslider_w_max = self.mini_menu.get_widget('rangeslider_h_max')
            h_max = int(rangeslider_h_max.get_value())
            self.__update_problem_config({'h_max': h_max})

        rangeslider_h_max = self.mini_menu.add.range_slider(
            'h_max',
            rangeslider_id='rangeslider_h_max',
            default=self.problem_config['h_max'],
            range_values=(self.problem_config['h_min'] + 1, self.problem_config['box_length']),
            increment=1,
            onchange=rangeslider_h_max_onchange,
            font_size=20,
            shadow_width=10,
            background_color=(0, 0, 0)
        )

        self.mini_menu.add.vertical_margin(40)

        btn_close_mini_menu = self.mini_menu.add.button(
            'Save and Return',
            pygame_menu.events.BACK,
            button_id='close_mini_menu',
            font_size=20,
            shadow_width=10,
            background_color=(0, 0, 0)
        )
        btn_close_mini_menu.set_onmouseover(lambda: button_onmouseover(btn_close_mini_menu))
        btn_close_mini_menu.set_onmouseleave(lambda: button_onmouseleave(btn_close_mini_menu))

        btn_configure = self.menu.add.button(
            'Configure Problem',
            self.mini_menu,
            button_id='configure_problem',
            font_size=20,
            shadow_width=10,
            background_color=(0, 0, 0)
        )
        btn_configure.set_onmouseover(lambda: button_onmouseover(btn_configure))
        btn_configure.set_onmouseleave(lambda: button_onmouseleave(btn_configure))

        def generate_instance():
            self.stop_search()  # IMPORTANT!
            self.__setup_new_problem()

        btn_generate = self.menu.add.button(
            'Generate Instance',
            generate_instance,
            button_id='generate_instance',
            font_size=20,
            shadow_width=10,
            background_color=(0, 0, 0)
        )
        btn_generate.set_onmouseover(lambda: button_onmouseover(btn_generate))
        btn_generate.set_onmouseleave(lambda: button_onmouseleave(btn_generate))

        self.menu.add.vertical_margin(100)

        def dropselect_algorithm_onchange(s, *args) -> None:
            self.stop_search()

            btn_search = self.menu.get_widget('run_search')
            algorithm = args[0]
            self.search = algorithm

            # btn_search.readonly = False
            btn_search.is_selectable = True
            btn_search.set_cursor(pygame_menu.locals.CURSOR_HAND)

        dropselect_algorithm = self.menu.add.dropselect(
            title='Algorithm',
            items=[
                ('Local Search', local_search),
                ('Greedy Search', greedy_search)
            ],
            dropselect_id='algorithm',
            font_size=20,
            onchange=dropselect_algorithm_onchange,
            padding=10,
            default=0,
            placeholder='Select one',
            selection_box_height=5,
            selection_box_inflate=(0, 10),
            selection_box_margin=5,
            selection_box_text_margin=10,
            selection_box_width=200,
            selection_option_font_size=20,
            background_color=(0, 0, 0),
        )
        dropselect_algorithm.set_onmouseover(lambda: button_onmouseover(dropselect_algorithm))
        dropselect_algorithm.set_onmouseleave(lambda: button_onmouseleave(dropselect_algorithm))

        dropselect_heuristic = self.menu.add.dropselect(
            title='Heuristic',
            items=[
                ('None', None),
                ('Geometric', None),
                ('Something Else', None)
            ],
            dropselect_id='heuristic',
            font_size=20,
            onchange=None,
            padding=10,
            default=0,
            placeholder='Select one',
            selection_box_height=5,
            selection_box_inflate=(0, 10),
            selection_box_margin=5,
            selection_box_text_margin=10,
            selection_box_width=200,
            selection_option_font_size=20,
            background_color=(0, 0, 0),
        )
        dropselect_heuristic.set_onmouseover(lambda: button_onmouseover(dropselect_heuristic))
        dropselect_heuristic.set_onmouseleave(lambda: button_onmouseleave(dropselect_heuristic))

        def run_search():
            btn_search = self.menu.get_widget('run_search')
            if not self.is_searching:
                self.__start_search()
            else:
                self.is_paused = True
                self.stop_search()

        btn_search = self.menu.add.button(
            'Run Search',
            run_search,
            button_id='run_search',
            font_size=20,
            shadow_width=10,
            background_color=(0, 0, 0)
        )

        # btn_search.readonly = True
        btn_search.set_onmouseover(lambda: button_onmouseover(btn_search))
        btn_search.set_onmouseleave(lambda: button_onmouseleave(btn_search))

        def reset_search():
            #btn_reset = self.menu.get_widget('reset_search')

            if self.is_searching:
                self.stop_search()
                self.search_thread.join()

            self.search_start_time = None
            self.search_stop_time = None

            self.set_current_solution(copy.deepcopy(self.init_sol))

        btn_reset = self.menu.add.button(
            'Reset Search',
            reset_search,
            button_id='reset_search',
            font_size=20,
            shadow_width=10,
            background_color=(0, 0, 0)
        )
        btn_reset.set_onmouseover(lambda: button_onmouseover(btn_reset))
        btn_reset.set_onmouseleave(lambda: button_onmouseleave(btn_reset))

        btn_exit = self.menu.add.button(
            'Exit',
            pygame_menu.events.EXIT,
            button_id='exit',
            font_size=20,
            shadow_width=10,
            background_color=(0, 0, 0),

        )
        btn_exit.set_onmouseover(lambda: button_onmouseover(btn_exit))
        btn_exit.set_onmouseleave(lambda: button_onmouseleave(btn_exit))

    def __start_search(self):
        self.is_searching = True
        self.search_thread = threading.Thread(target=self.search, args=(self.get_current_solution(),
                                                                        self.problem, self))
        self.search_thread.start()

        if self.is_paused:
            self.is_paused = False
        else:
            self.search_start_time = time.time()

        btn_search = self.menu.get_widget('run_search')
        btn_search.set_title('Pause Search')

        btn_configure = self.menu.get_widget('configure_problem')
        btn_configure.readonly = True

    def __setup_new_problem(self):
        self.problem = RectanglePackingProblemGeometryBased(**self.problem_config)

        sol = self.problem.get_arbitrary_solution()
        self.init_sol = copy.deepcopy(sol)
        self.set_current_solution(sol)

    def __update_problem_config(self, update_dict: dict):
        problem_config = self.problem_config.copy()
        problem_config.update(update_dict)

        L = problem_config['box_length']
        N = problem_config['num_rects']

        # ensure that w_min, w_max, h_min, h_max <= L
        problem_config['w_min'] = min(problem_config['w_min'], L)
        problem_config['w_max'] = min(problem_config['w_max'], L)
        problem_config['h_min'] = min(problem_config['h_min'], L)
        problem_config['h_max'] = min(problem_config['h_max'], L)

        # ensure that w_min <= w_max
        problem_config['w_min'] = min(problem_config['w_max'], problem_config['w_min'])
        problem_config['h_min'] = min(problem_config['h_max'], problem_config['h_min'])

        # get all the widgets and update their values
        rangeslider_box_length = self.mini_menu.get_widget('rangeslider_box_length')
        rangeslider_num_rects = self.mini_menu.get_widget('rangeslider_num_rects')
        rangeslider_w_min = self.mini_menu.get_widget('rangeslider_w_min')
        rangeslider_w_max = self.mini_menu.get_widget('rangeslider_w_max')
        rangeslider_h_min = self.mini_menu.get_widget('rangeslider_h_min')
        rangeslider_h_max = self.mini_menu.get_widget('rangeslider_h_max')

        rangeslider_box_length.set_value(L)
        rangeslider_num_rects.set_value(N)

        rangeslider_w_min.set_value(problem_config['w_min'])
        rangeslider_w_max.set_value(problem_config['w_max'])
        rangeslider_h_min.set_value(problem_config['h_min'])
        rangeslider_h_max.set_value(problem_config['h_max'])

        rangeslider_w_min._range_values = (1, L)
        rangeslider_w_max._range_values = (1, L)
        rangeslider_h_min._range_values = (1, L)
        rangeslider_h_max._range_values = (1, L)

        for k, v in update_dict.items():
            if self.problem_config[k] != v:
                self.problem_config = problem_config
                self.__setup_new_problem()
                return

    def resize_window(self, w, h):
        pygame.display.set_mode((w, h), pygame.RESIZABLE)
        self.area_width = self.screen.get_width()
        self.area_height = self.screen.get_height()

        self.menu.resize(w, h, position=(1, 1, False))
        self.mini_menu.resize(w, h, position=(1, 1, False))

    def set_current_solution(self, solution: RectanglePackingSolutionGeometryBased):
        self.current_sol = solution
        self.rect_dims = self.get_rect_dimensions()

    def get_current_solution(self):
        return self.current_sol

    def set_and_animate_solution(self, solution: RectanglePackingSolutionGeometryBased):
        # Identify modified rect
        current_sol_matrix = np.zeros((self.problem.num_rects, 3))
        new_sol_matrix = np.zeros((self.problem.num_rects, 3))
        current_sol_matrix[:, 0:2], current_sol_matrix[:, 2] = self.current_sol.locations, self.current_sol.rotations
        new_sol_matrix[:, 0:2], new_sol_matrix[:, 2] = solution.locations, solution.rotations
        differences = np.any(current_sol_matrix != new_sol_matrix, axis=1)

        if not np.any(differences):
            changed_rect_idx = solution.pending_move_params[0]
        else:
            changed_rect_idx = np.argmax(differences)

        # Select the rect to change
        self.selected_rect_idx = changed_rect_idx
        time.sleep(ANIM_SPEED)

        # Apply new solution
        solution.apply_pending_move()
        self.set_current_solution(solution)
        time.sleep(ANIM_SPEED)

        # Unselect the changed rect
        self.selected_rect_idx = None

    def __run(self):
        self.__init_gui()
        self.__setup_menu()
        self.__setup_new_problem()

        frame = 0

        while self.running:
            t = time.time()
            self.__handle_user_input()
            t_unser_input = time.time() - t
            self.__render()
            t_total = time.time()
            t_render = t_total - t_unser_input

            # print("\rTime shares: Input handling %.1f - rendering %.1f" %
            #       (t_unser_input / t_total, t_render / t_total), end="")

            self.frame_times[frame % 2000] = time.time()
            frame += 1

        pygame.quit()

    def __handle_user_input(self):
        mouse_pos = np.asarray(pygame.mouse.get_pos())
        x, y = self.mouse_pos_to_field_coords(mouse_pos)

        rect_idx = self.get_rect_idx_at(x, y)

        if rect_idx is not None:
            pygame.mouse.set_cursor(*pygame.cursors.broken_x)
        else:
            pygame.mouse.set_cursor(*pygame.cursors.arrow)

        events = pygame.event.get()
        if self.menu.is_enabled():
            self.menu.update(events)

        for event in events:

            # Did the user click the window close button?
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 2:  # center mousebutton
                    self.dragging_camera = True
                    self.old_mouse_pos = mouse_pos

                elif event.button == 1:  # left mousebutton
                    if self.selected_rect_idx is None:
                        self.selected_rect_idx = rect_idx
                        self.selection_rotated = False
                    else:
                        new_solution = self.current_sol.copy()
                        new_solution.make_standalone()
                        rotated = self.selection_rotated != self.current_sol.rotations[self.selected_rect_idx]
                        new_solution.move_rect(self.selected_rect_idx, np.array([x, y]), rotated)
                        new_solution.apply_pending_move()
                        if self.problem.is_feasible(new_solution):
                            self.set_current_solution(new_solution)
                            self.selected_rect_idx = None

                elif event.button == 3:  # right mousebutton
                    if self.selected_rect_idx is not None:
                        self.selection_rotated = not self.selection_rotated

                elif event.button == 4:  # mousewheel up
                    self.zoom(zoom_in=True)

                elif event.button == 5:  # mousewheel down
                    self.zoom(zoom_in=False)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:  # center mousebutton
                    self.dragging_camera = False

            elif self.dragging_camera and event.type == pygame.MOUSEMOTION:
                shift_delta = mouse_pos - self.old_mouse_pos
                self.cam_pos += shift_delta
                self.old_mouse_pos = mouse_pos

            elif event.type == pygame.VIDEORESIZE:  # Resize pygame display area on window resize
                self.resize_window(event.w, event.h)

    def zoom(self, zoom_in: bool):
        # Fetch info
        window_size = np.asarray(pygame.display.get_surface().get_size())
        mouse_pos = np.asarray(pygame.mouse.get_pos())
        rel_mouse_pos = mouse_pos / window_size - 0.5
        cam_center_pos = self.cam_pos - 0.5 * window_size

        # Determine zoom level
        zoom_step_factor = ZOOM_STEP_FACTOR if zoom_in else 1 / ZOOM_STEP_FACTOR
        true_zoom_step_factor = np.round(self.config['field_size'] * self.zoom_level * zoom_step_factor) / \
                                np.round(self.config['field_size'] * self.zoom_level)
        target_zoom_level = self.zoom_level * true_zoom_step_factor
        if target_zoom_level > 3 or target_zoom_level < 0.25:
            return

        # Adjust camera position
        cam_zoom_shift = rel_mouse_pos * (1 - true_zoom_step_factor) * window_size
        target_cam_center_pos = np.round(cam_center_pos * true_zoom_step_factor + cam_zoom_shift).astype(np.int)
        target_cam_pos = target_cam_center_pos + 0.5 * window_size

        # Set target values
        self.cam_pos = target_cam_pos
        self.zoom_level = target_zoom_level

    def __render(self):
        times = np.zeros(7)
        t = time.time()

        # Screen area
        pygame.draw.rect(self.screen, self.colors['grid_bg'],
                         [self.scr_marg_left, self.scr_marg_top,
                          self.area_width, self.area_height])

        # Get visible grid boundary
        top_left = -self.cam_pos // self.field_size + 1
        bottom_right = (-self.cam_pos + np.asarray([self.area_width, self.area_height])) // self.field_size + 1

        top_left = top_left.astype(np.int32)
        bottom_right = bottom_right.astype(np.int32)

        times[0] = time.time() - t
        t = time.time()

        self.highlight_non_empty_boxes(top_left, bottom_right)

        times[1] = time.time() - t
        t = time.time()

        self.draw_fine_grid(top_left, bottom_right)

        if self.problem is not None:
            self.draw_box_grid(top_left, bottom_right)

            times[2] = time.time() - t
            t = time.time()

            if self.current_sol is not None:
                self.draw_rects(top_left, bottom_right)

            times[3] = time.time() - t
            t = time.time()

        self.draw_hover_shape()

        times[4] = time.time() - t
        t = time.time()

        if self.problem is not None and self.current_sol is not None:
            self.draw_text_info()

        times[5] = time.time() - t
        t = time.time()

        if self.menu.is_enabled():
            self.menu.draw(self.screen)

        times[6] = time.time() - t
        t_total = np.sum(times)
        shares = times / t_total

        print("\rTime shares: preparations %.3f - box highlight %.3f - grid lines %.3f - "
              "rects %.3f - hover %.3f - text %.3f - menu %.3f" %
              (shares[0], shares[1], shares[2], shares[3], shares[4], shares[5], shares[6]), end="")

        # Update the screen
        pygame.display.flip()

    def highlight_non_empty_boxes(self, view_top_left, view_bottom_right):
        l = self.problem.box_length
        # occupied_boxes = self.current_sol.box_coords[self.current_sol.box_occupancies > 0]
        visible_boxes = self.get_visible_boxes(view_top_left, view_bottom_right)
        for (x, y) in visible_boxes:
            self.draw_rect(x * l, y * l, l, l, color=self.colors['non_empty_boxes'])

    def get_visible_boxes(self, view_top_left, view_bottom_right):
        l = self.problem.box_length
        occupied_boxes = self.current_sol.box_coords[self.current_sol.box_occupancies > 0]
        left, top = view_top_left
        right, bottom = view_bottom_right
        xs, ys = occupied_boxes.T * l
        visible_boxes = ((left <= xs + l - 1) & (xs < right)) & \
                        ((top <= ys + l - 1) & (ys < bottom))
        return occupied_boxes[visible_boxes]

    def draw_rects(self, view_top_left, view_bottom_right):
        # Identify visible rects
        top_lefts = self.rect_dims[:, [0, 1]]
        top_rights = top_lefts.copy()
        top_rights[:, 0] += self.rect_dims[:, 2]
        bottom_lefts = top_lefts.copy()
        bottom_lefts[:, 1] += self.rect_dims[:, 3]
        bottom_rights = top_lefts + self.rect_dims[:, [2, 3]]

        inside_view = np.zeros(self.problem.num_rects, dtype=np.bool)
        for corners in [top_lefts, top_rights, bottom_lefts, bottom_rights]:
            inside_view |= np.all(view_top_left <= corners, axis=1) & \
                           np.all(corners < view_bottom_right, axis=1)

        visible_rect_ids = np.where(inside_view)[0]

        for rect_idx in visible_rect_ids:
            x, y, w, h = self.rect_dims[rect_idx]
            color = self.colors['rectangles_search'] if self.is_searching else self.colors['rectangles']
            color = self.colors['active_rectangle'] if rect_idx == self.selected_rect_idx else color
            self.draw_rect(x, y, w, h, color=color)

    def draw_rect(self, x, y, w, h, color, surface=None):
        if surface is None:
            surface = self.screen
        x_p, y_p, w_p, h_p = self.coords2pixels(x, y, w, h)
        pygame.draw.rect(surface, color, [x_p, y_p, w_p, h_p])

    def coords2pixels(self, x, y, w=None, h=None):
        x_p = self.cam_pos[0] + self.scr_marg_left + x * self.field_size
        y_p = self.cam_pos[1] + self.scr_marg_top + y * self.field_size
        if w is None or h is None:
            return x_p, y_p
        else:
            w_p = w * self.field_size - self.config['line_width']
            h_p = h * self.field_size - self.config['line_width']
            return x_p, y_p, w_p, h_p

    def draw_fine_grid(self, top_left, bottom_right):
        xs = np.arange(top_left[0], bottom_right[0])
        ys = np.arange(top_left[1], bottom_right[1])
        self.draw_grid(xs, ys, self.colors['grid_lines'])

    def draw_box_grid(self, top_left, bottom_right):
        xs = np.arange(top_left[0], bottom_right[0])
        ys = np.arange(top_left[1], bottom_right[1])
        xs = xs[xs % self.problem.box_length == 0]
        ys = ys[ys % self.problem.box_length == 0]
        self.draw_grid(xs, ys, self.colors['box_boundary_lines'])

    def draw_grid(self, xs, ys, color):
        xs_p, ys_p = self.coords2pixels(xs, ys)
        xs_p -= self.config['line_width']
        ys_p -= self.config['line_width']

        # Draw vertical lines
        one_side = self.scr_marg_top - self.config['line_width']
        other_side = self.area_height + self.config['line_width']
        self.draw_lines(xs_p, True, one_side, other_side, color)

        # Draw horizontal lines
        one_side = self.scr_marg_left - self.config['line_width']
        other_side = self.area_width + self.config['line_width']
        self.draw_lines(ys_p, False, one_side, other_side, color)

    def draw_lines(self, positions, vertical, one_side, other_side, color):
        positions_ax1 = np.repeat(positions, 2)
        positions_ax2 = np.tile([one_side, other_side, other_side, one_side], len(positions) // 2 + 1)[:len(positions_ax1)]
        if not vertical:
            positions_ax1, positions_ax2 = positions_ax2, positions_ax1
        point_list = np.stack([positions_ax1, positions_ax2], axis=1)
        pygame.draw.lines(self.screen, color, False, point_list, self.config['line_width'])

    def draw_hover_shape(self):
        mouse_pos = np.asarray(pygame.mouse.get_pos())
        x, y = self.mouse_pos_to_field_coords(mouse_pos)
        rect_under_mouse = self.get_rect_idx_at(x, y)

        if self.selected_rect_idx is not None:
            w, h = self.rect_dims[self.selected_rect_idx, 2:4]
            if self.selection_rotated:
                w, h = h, w
        elif rect_under_mouse is not None:
            x, y, w, h = self.rect_dims[rect_under_mouse]
        else:
            w, h = 1, 1

        x_p, y_p, w_p, h_p = self.coords2pixels(x, y, w, h)

        hover_surface = pygame.Surface((w_p, h_p))
        hover_surface.set_alpha(60)
        hover_surface.set_colorkey((0, 0, 0))
        pygame.draw.rect(hover_surface, self.colors['hover'], [0, 0, w_p, h_p])
        self.screen.blit(hover_surface, (x_p, y_p))

    def draw_text_info(self):
        # Display current solution value
        value = self.problem.objective_function(self.current_sol)
        text_surface = self.font.render('Objective Value: %d' % value, True, self.colors['font'])
        self.screen.blit(text_surface, (32, 32))

        # Display current heuristic value
        heuristic = self.problem.heuristic(self.current_sol)
        text_surface = self.font.render('Heuristic Value: %.2f' % heuristic, True, self.colors['font'])
        self.screen.blit(text_surface, (32, 70))

        if self.search_start_time is not None:
            if self.is_searching:
                elapsed = time.time() - self.search_start_time
            else:
                elapsed = self.search_stop_time - self.search_start_time
        else:
            elapsed = 0

        minutes = elapsed // 60
        seconds = int(elapsed)
        milliseconds = int(elapsed*1000) % 1000
        text_surface = self.font.render('Elapsed Time: %d:%02d:%03d min' % (minutes, seconds, milliseconds),
                                       True, self.colors['font'])
        self.screen.blit(text_surface, ((self.screen.get_width() - self.font.size('Elapsed Time:')[0]) // 2, 32))

        # Display if current solution is optimal
        if self.problem.is_optimal(self.current_sol):
            text_surface = self.font.render('This solution is optimal.', True, self.colors['font'])
            self.screen.blit(text_surface, (32, 108))

        # FPS
        t = time.time()
        during_last_sec = (t - self.frame_times) < 1
        fps = np.sum(during_last_sec)
        text_surface = self.font.render('FPS: %d' % int(fps), True, self.colors['font'])
        self.screen.blit(text_surface, (32, 146))

    def mouse_pos_to_field_coords(self, mouse_pos):
        field_y = mouse_pos[1] - self.scr_marg_top - self.cam_pos[1]
        field_x = mouse_pos[0] - self.scr_marg_left - self.cam_pos[0]
        y_coord = int(field_y // self.field_size)
        x_coord = int(field_x // self.field_size)
        return x_coord, y_coord

    def get_rect_idx_at(self, x, y):
        # if self.current_sol is None:
        #     return None
        # np.where(self.current_sol)

        if self.rect_dims is None:
            return None

        xs, ys, ws, hs = self.rect_dims.T

        result = np.where((xs <= x) &
                          (x < xs + ws) &
                          (ys <= y) &
                          (y < ys + hs))[0]

        if not result:
            return None
        else:
            return result[0]

        # for rect_idx, (rx, ry, rw, rh) in enumerate(self.rect_dims):
        #     if rx <= x < rx + rw and ry <= y < ry + rh:
        #         return rect_idx
        # return None

    def get_rect_dimensions(self):
        """Returns an array containing all dimension information (x, y, w, h) for each rectangle."""
        dims = np.zeros((self.problem.num_rects, 4), dtype=np.int32)

        locations = self.current_sol.locations
        rotations = self.current_sol.rotations
        sizes = self.problem.sizes

        dims[:, 0:2] = locations
        dims[:, 2:4] = sizes

        # Swap x and y for rotated rects
        dims[rotations, 2] = sizes[rotations, 1]
        dims[rotations, 3] = sizes[rotations, 0]

        return dims
