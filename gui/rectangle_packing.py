import time
import pygame
import pygame_menu
import threading
import numpy as np
import json
import copy

from algos import local_search, greedy_search
from gui import BaseGUI
from problems.examples.rectangle_packing import RectanglePackingProblem, RectanglePackingSolution


ZOOM_STEP_FACTOR = 1.1
ANIM_SPEED = 0.1  # sec


class RectanglePackingGUI(BaseGUI):
    def __init__(self):
        super().__init__()

        # Problem constants
        self.problem_config = dict(box_length=8, num_rects=32, w_min=1, w_max=8, h_min=1, h_max=8)
        self.problem = None
        self.problem_copy = None
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

    def stop_search(self):
        # not private because search algorithm shall invoke it as well
        self.is_searching = False
        self.search_stop_time = time.time()

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

        def run_search():
            btn_search = self.menu.get_widget('run_search')
            if not self.is_searching:
                self.__start_search()
            else:
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
            btn_reset = self.menu.get_widget('reset_search')
            if self.is_searching:
                self.stop_search()
            self.search_start_time = None

            self.problem = self.problem_copy
            self.set_current_solution(self.init_sol)

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

        self.search_start_time = time.time()

        btn_search = self.menu.get_widget('run_search')
        btn_search.set_title('Pause Search')

        btn_configure = self.menu.get_widget('configure_problem')
        btn_configure.readonly = True

    def __setup_new_problem(self):
        self.problem = RectanglePackingProblem(**self.problem_config)
        self.problem_copy = copy.deepcopy(self.problem)

        self.init_sol = self.problem.get_arbitrary_solution()
        self.set_current_solution(self.init_sol)

    def __update_problem_config(self, update_dict: dict):
        problem_config = self.problem_config.copy()
        problem_config.update(update_dict)

        print(update_dict)

        problem_config['w_min'] = min(problem_config['w_max'],
                                      min(problem_config['w_min'], problem_config['box_length']))
        problem_config['w_max'] = max(problem_config['w_min'],
                                      min(problem_config['w_max'], problem_config['box_length']))
        problem_config['h_min'] = min(problem_config['h_max'],
                                      min(problem_config['h_min'], problem_config['box_length']))
        problem_config['h_max'] = max(problem_config['h_min'],
                                      min(problem_config['h_max'], problem_config['box_length']))

        rangeslider_box_length = self.mini_menu.get_widget('rangeslider_box_length')
        rangeslider_num_rects = self.mini_menu.get_widget('rangeslider_num_rects')
        rangeslider_w_min = self.mini_menu.get_widget('rangeslider_w_min')
        rangeslider_w_max = self.mini_menu.get_widget('rangeslider_w_max')
        rangeslider_h_min = self.mini_menu.get_widget('rangeslider_h_min')
        rangeslider_h_max = self.mini_menu.get_widget('rangeslider_h_max')

        rangeslider_box_length.set_value(problem_config['box_length'])
        rangeslider_num_rects.set_value(problem_config['num_rects'])

        rangeslider_w_min.set_value(problem_config['w_min'])
        rangeslider_w_max.set_value(problem_config['w_max'])
        rangeslider_h_min.set_value(problem_config['h_min'])
        rangeslider_h_max.set_value(problem_config['h_max'])

        rangeslider_w_min._range_values = (1, problem_config['w_max'] - 1)
        rangeslider_w_max._range_values = (problem_config['w_min'] + 1, problem_config['box_length'])
        rangeslider_h_min._range_values = (1, problem_config['h_max'] - 1)
        rangeslider_h_max._range_values = (problem_config['h_min'] + 1, problem_config['box_length'])

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

    def set_current_solution(self, solution: RectanglePackingSolution):
        self.current_sol = solution
        self.rect_dims = self.get_rect_dimensions()

    def get_current_solution(self):
        return self.current_sol

    def set_and_animate_solution(self, solution: RectanglePackingSolution):
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

        while self.running:
            self.__handle_user_input()
            self.__render()

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
        # Grid area
        pygame.draw.rect(self.screen, self.colors['grid_bg'],
                         [self.scr_marg_left - self.config['line_width'],
                          self.scr_marg_top - self.config['line_width'],
                          self.area_width, self.area_height])

        # Get visible grid boundary
        top_left = -self.cam_pos // self.field_size + 1
        bottom_right = (-self.cam_pos + np.asarray([self.area_width, self.area_height])) // self.field_size + 1

        top_left = top_left.astype(np.int32)
        bottom_right = bottom_right.astype(np.int32)

        # Highlight non-empty boxes
        l = self.problem.box_length
        occupied_boxes = self.current_sol.box_coords[self.current_sol.box_occupancies > 0]
        for (x, y) in occupied_boxes:
            self.draw_rect(x * l, y * l, l, l, color=self.colors['non_empty_boxes'])

        # Draw grid lines
        for y in range(top_left[1], bottom_right[1]):
            self.draw_h_line(y, self.colors['grid_lines'])
        for x in range(top_left[0], bottom_right[0]):
            self.draw_v_line(x, self.colors['grid_lines'])

        if self.problem is not None:
            # Draw box boundary grid lines
            for y in range(top_left[1], bottom_right[1]):
                if y % self.problem.box_length == 0:
                    self.draw_h_line(y, self.colors['box_boundary_lines'])
            for x in range(top_left[0], bottom_right[0]):
                if x % self.problem.box_length == 0:
                    self.draw_v_line(x, self.colors['box_boundary_lines'])

            if self.current_sol is not None:
                # Draw rectangles from current solution
                for rect_idx, (x, y, w, h) in enumerate(self.rect_dims):
                    color = self.colors['rectangles_search'] if self.is_searching else self.colors['rectangles']
                    color = self.colors['active_rectangle'] if rect_idx == self.selected_rect_idx else color

                    self.draw_rect(x, y, w, h, color=color)

        self.draw_hover_shape()

        if self.problem is not None and self.current_sol is not None:
            self.draw_text_info()

        if self.menu.is_enabled():
            self.menu.draw(self.screen)

        # Update the screen
        pygame.display.flip()

    def draw_rect(self, x, y, w, h, color, surface=None):
        if surface is None:
            surface = self.screen
        pygame.draw.rect(surface, color,
                         [self.cam_pos[0] + self.scr_marg_left + x * self.field_size,
                          self.cam_pos[1] + self.scr_marg_top + y * self.field_size,
                          w * self.field_size - self.config['line_width'],
                          h * self.field_size - self.config['line_width']])

    def draw_h_line(self, y, color):
        start = [self.scr_marg_left,
                 self.cam_pos[1] + self.scr_marg_top + y * self.field_size - self.config['line_width']]
        end = [self.area_width,
               self.cam_pos[1] + self.scr_marg_top + y * self.field_size - self.config['line_width']]
        pygame.draw.line(self.screen, color, start, end, self.config['line_width'])

    def draw_v_line(self, x, color):
        start = [self.cam_pos[0] + self.scr_marg_left + x * self.field_size - self.config['line_width'],
                 self.scr_marg_top]
        end = [self.cam_pos[0] + self.scr_marg_left + x * self.field_size - self.config['line_width'],
               self.area_height]
        pygame.draw.line(self.screen, color, start, end, self.config['line_width'])

    def draw_hover_shape(self):
        hover_surface = pygame.Surface((self.screen.get_width(), self.screen.get_height()))
        hover_surface.set_alpha(60)
        hover_surface.set_colorkey((0, 0, 0))

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

        self.draw_rect(x, y, w, h, self.colors['hover'], surface=hover_surface)
        self.screen.blit(hover_surface, (0, 0))

    def draw_text_info(self):
        # Display current solution value
        value = self.problem.objective_function(self.current_sol)
        textsurface = self.font.render('Objective Value: %d' % value, True, self.colors['font'])
        self.screen.blit(textsurface, (32, 32))

        # Display current heuristic value
        heuristic = self.problem.heuristic(self.current_sol)
        textsurface = self.font.render('Heuristic Value: %.2f' % heuristic, True, self.colors['font'])
        self.screen.blit(textsurface, (32, 70))

        if self.search_start_time is not None:
            if self.is_searching:
                elapsed = time.time() - self.search_start_time
            else:
                elapsed = self.search_stop_time - self.search_start_time
        else:
            elapsed = 0
        textsurface = self.font.render('Elapsed Time: %.4f s' % elapsed, True, self.colors['font'])
        self.screen.blit(textsurface, ((self.screen.get_width() - self.font.size('Elapsed Time:')[0]) // 2, 32))

        # Display if current solution is optimal
        if self.problem.is_optimal(self.current_sol):
            textsurface = self.font.render('This solution is optimal.', True, self.colors['font'])
            self.screen.blit(textsurface, (32, 108))

    def mouse_pos_to_field_coords(self, mouse_pos):
        field_y = mouse_pos[1] - self.scr_marg_top - self.cam_pos[1]
        field_x = mouse_pos[0] - self.scr_marg_left - self.cam_pos[0]
        y_coord = int(field_y // self.field_size)
        x_coord = int(field_x // self.field_size)
        return x_coord, y_coord

    def get_rect_idx_at(self, x, y):
        if self.rect_dims is None:
            return None
        for rect_idx, (rx, ry, rw, rh) in enumerate(self.rect_dims):
            if rx <= x < rx + rw and ry <= y < ry + rh:
                return rect_idx
        return None

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
