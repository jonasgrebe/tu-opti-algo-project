import time

import pygame
import pygame_menu
import threading
import numpy as np
import json

from algos import local_search

class RectanglePackingGUI:
    def __init__(self):
        # Problem constants
        self.problem = None
        self.current_sol = None
        self.rect_dims = None
        self.search = local_search  # the search algorithm routine

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

        self.is_searching = False
        self.search_thread = None


    @property
    def colors(self):
        return self.config['colors']

    @property
    def field_size(self):
        return np.round(self.config['field_size'] * self.zoom)

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
        self.zoom = 1.0

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
            widget_selection_effect=pygame_menu.widgets.NoneSelection()
        )
        self.menu = pygame_menu.Menu(
            width=240,
            height=self.screen.get_height(),
            mouse_motion_selection=True,
            theme=theme,
            title='',
        )

        def generate_instance():
            self.is_searching = False # IMPORTANT!
            self.problem.generate()
            init_sol = self.problem.get_arbitrary_solution()
            self.set_current_solution(init_sol)

        btn_generate = self.menu.add.button(
            'Generate Instance',
            generate_instance,
            button_id='generate_instance',
            font_size=20,
            shadow_width=10,
            align=pygame_menu.locals.ALIGN_RIGHT,
            background_color=(0, 0, 0)
        )
        btn_generate.translate(-50, -200)
        btn_generate.set_onmouseover(lambda: button_onmouseover(btn_generate))
        btn_generate.set_onmouseleave(lambda: button_onmouseleave(btn_generate))

        def run_search():
            if self.is_searching:
                return
            self.is_searching = True # IMPORTANT!
            self.search_thread = threading.Thread(target=self.search, args=(self.problem, self))
            self.search_thread.start()

        btn_search = self.menu.add.button(
            'Run Search',
            run_search,
            button_id='run_search',
            font_size=20,
            shadow_width=10,
            align=pygame_menu.locals.ALIGN_RIGHT,
            background_color=(0, 0, 0)
        )
        btn_search.translate(-50, -200)
        btn_search.set_onmouseover(lambda: button_onmouseover(btn_search))
        btn_search.set_onmouseleave(lambda: button_onmouseleave(btn_search))

        self.menu.center_content()


    def resize_window(self, w, h):
        pygame.display.set_mode((w, h), pygame.RESIZABLE)
        self.area_width = self.screen.get_width()
        self.area_height = self.screen.get_height()

        self.menu.resize(w, h, position=(1, 1, False))

    def set_current_solution(self, solution):
        self.current_sol = solution
        self.rect_dims = self.get_rect_dimensions()

    def set_and_animate_solution(self, solution):
        # Identify modified rect
        current_sol_matrix = np.zeros((self.problem.num_rects, 3))
        new_sol_matrix = np.zeros((self.problem.num_rects, 3))
        current_sol_matrix[:, 0:2], current_sol_matrix[:, 2] = self.current_sol
        new_sol_matrix[:, 0:2], new_sol_matrix[:, 2] = solution
        differences = np.any(current_sol_matrix != new_sol_matrix, axis=1)
        changed_rect_idx = np.argmax(differences)

        # Select the rect to change
        self.selected_rect_idx = changed_rect_idx
        time.sleep(1)

        # Apply new solution
        self.set_current_solution(solution)
        time.sleep(1)

        # Unselect the changed rect
        self.selected_rect_idx = None

    def __run(self):
        self.__init_gui()
        self.__setup_menu()

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
                    locations, rotations = self.current_sol
                    if self.selected_rect_idx is None:
                        self.selected_rect_idx = rect_idx
                        self.selection_rotated = False
                    else:
                        locations_new = locations.copy()
                        locations_new[self.selected_rect_idx] = [x, y]
                        rotations_new = rotations.copy()
                        if self.selection_rotated:
                            rotations_new[self.selected_rect_idx] = ~ rotations_new[self.selected_rect_idx]
                        new_solution = (locations_new, rotations_new)
                        if self.problem.is_feasible(new_solution):
                            self.set_current_solution(new_solution)
                            self.selected_rect_idx = None

                elif event.button == 3:  # right mousebutton
                    if self.selected_rect_idx is not None:
                        self.selection_rotated = not self.selection_rotated

                elif event.button == 4:  # mousewheel up
                    self.zoom = min(self.zoom * 1.1, 3)
                    print(self.zoom)

                elif event.button == 5:  # mousewheel down
                    self.zoom = max(self.zoom / 1.1, 0.25)
                    print(self.zoom)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:  # center mousebutton
                    self.dragging_camera = False

            elif self.dragging_camera and event.type == pygame.MOUSEMOTION:
                shift_delta = mouse_pos - self.old_mouse_pos
                self.cam_pos += shift_delta
                self.old_mouse_pos = mouse_pos

            elif event.type == pygame.VIDEORESIZE:  # Resize pygame display area on window resize
                self.resize_window(event.w, event.h)


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
        occupied_boxes = self.problem.get_occupied_boxes(self.current_sol)
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
                    color = self.colors['active_rectangle'] if rect_idx == self.selected_rect_idx else \
                        self.colors['rectangles']
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
        textsurface = self.font.render(f'Objective Value: {self.problem.f(self.current_sol)}', True,
                                       self.colors['font'])
        self.screen.blit(textsurface, (32, 32))

        # Display if current solution is optimal
        if self.problem.is_optimal(self.current_sol):
            textsurface = self.font.render(f'This solution is optimal.', True,
                                           self.colors['font'])
            self.screen.blit(textsurface, (32, 70))

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

        locations, rotations = self.current_sol
        sizes = self.problem.sizes

        dims[:, 0:2] = locations
        dims[:, 2:4] = sizes

        # Swap x and y for rotated rects
        dims[rotations, 2] = sizes[rotations, 1]
        dims[rotations, 3] = sizes[rotations, 0]

        return dims
