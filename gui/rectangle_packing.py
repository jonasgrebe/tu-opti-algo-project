import pygame
import threading
import numpy as np
import json


class RectanglePackingGUI:
    def __init__(self):
        self.problem = None
        self.current_sol = None
        self.rect_dims = None
        self.search = None  # the search algorithm routine
        self.running = True

        with open("gui/config.json") as json_data_file:
            self.config = json.load(json_data_file)

        self.render_thread = threading.Thread(target=self.__render)
        self.render_thread.start()

    @property
    def colors(self):
        return self.config['colors']

    @property
    def field_size(self):
        return self.config['field_size'] * self.zoom

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

    def resize_window(self, w, h):
        pygame.display.set_mode((w, h), pygame.RESIZABLE)
        self.area_width = self.screen.get_width()
        self.area_height = self.screen.get_height()

    def set_current_solution(self, solution):
        self.current_sol = solution
        self.rect_dims = self.get_rect_dimensions()

    def __render(self):
        self.__init_gui()

        dragging = False
        old_mouse_pos = None
        moving_rect_idx = None

        while self.running:
            mouse_pos = np.asarray(pygame.mouse.get_pos())
            x, y = self.mouse_pos_to_field_coords(mouse_pos)
            rect_idx = self.get_rect_idx_at(x, y)
            if rect_idx is not None:
                pygame.mouse.set_cursor(*pygame.cursors.broken_x)
            else:
                pygame.mouse.set_cursor(*pygame.cursors.arrow)

            # Handle user input
            for event in pygame.event.get():
                # Did the user click the window close button?
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 2:  # center mousebutton
                        dragging = True
                        old_mouse_pos = mouse_pos

                    elif event.button == 1:  # left mousebutton
                        if moving_rect_idx is None:
                            moving_rect_idx = rect_idx
                        else:
                            locations, rotations = self.current_sol
                            locations_new = locations.copy()
                            locations_new[moving_rect_idx] = [x, y]
                            new_solution = (locations_new, rotations)
                            if self.problem.is_feasible(new_solution):
                                self.set_current_solution(new_solution)
                                moving_rect_idx = None

                    elif event.button == 4:  # mousewheel up
                        self.zoom *= 1.1

                    elif event.button == 5:  # mousewheel down
                        self.zoom /= 1.1

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 2:  # center mousebutton
                        dragging = False

                elif dragging and event.type == pygame.MOUSEMOTION:
                    shift_delta = mouse_pos - old_mouse_pos
                    self.cam_pos += shift_delta
                    old_mouse_pos = mouse_pos

                elif event.type == pygame.VIDEORESIZE:  # Resize pygame display area on window resize
                    self.resize_window(event.w, event.h)

            # self.screen.fill(self.colors['bg_color'])

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

            # highlight non-empty boxes
            for x in range(top_left[0] - self.problem.box_length, bottom_right[0] + self.problem.box_length):
                if x % self.problem.box_length == 0:
                    for y in range(top_left[1] - self.problem.box_length, bottom_right[1] + self.problem.box_length):
                        if y % self.problem.box_length == 0:
                            if not self.is_empty_block(x, y):
                                l = self.problem.box_length
                                self.draw_rect(x, y, l, l, color=self.colors['non_empty_boxes'])

            # Draw grid lines
            for y in range(top_left[1], bottom_right[1]):
                self.draw_h_line(y, self.colors['grid_lines'])
            for x in range(top_left[0], bottom_right[0]):
                self.draw_v_line(x, self.colors['grid_lines'])

            if self.problem is not None:
                # Draw box boundaries
                for y in range(top_left[1], bottom_right[1]):
                    if y % self.problem.box_length == 0:
                        self.draw_h_line(y, self.colors['box_boundary_lines'])
                for x in range(top_left[0], bottom_right[0]):
                    if x % self.problem.box_length == 0:
                        self.draw_v_line(x, self.colors['box_boundary_lines'])

                if self.current_sol is not None:
                    # Draw rectangles from current solution
                    for x, y, w, h in self.rect_dims:
                        self.draw_rect(x, y, w, h, color=self.colors['rectangles'])

            self.draw_hover_shape(mouse_pos, moving_rect_idx)

            # display current solution value
            if self.current_sol is not None:
                textsurface = self.font.render(f'Objective Value: {self.problem.f(self.current_sol)}', True,
                                               self.colors['font'])
                self.screen.blit(textsurface, (32, 32))

            # Update the screen
            pygame.display.flip()
            # time.sleep(0.016)

        pygame.quit()

    def is_empty_block(self, x, y):
        if self.current_sol is not None:
            locations, rotations = self.current_sol
            locations_temp = locations.copy()
            locations_temp[rotations, 0] = locations[rotations, 1]
            locations_temp[rotations, 1] = locations[rotations, 0]

            for rect_idx in range(self.problem.num_rects):
                rx, ry = locations_temp[rect_idx]
                rw, rh = self.problem.rectangles[rect_idx]

                if x <= rx and rx < (x + self.problem.box_length) and y <= ry and ry < (y + self.problem.box_length):
                    return False
        return True

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

    def draw_hover_shape(self, mouse_pos, rect_idx):
        x, y = self.mouse_pos_to_field_coords(mouse_pos)

        hover_surface = pygame.Surface((self.screen.get_width(), self.screen.get_height()))
        hover_surface.set_alpha(60)
        hover_surface.set_colorkey((0, 0, 0))

        if rect_idx is not None:
            w, h = self.rect_dims[rect_idx, 2:4]
        else:
            w, h = 1, 1

        self.draw_rect(x, y, w, h, self.colors['hover'], surface=hover_surface)
        self.screen.blit(hover_surface, (0, 0))

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
        dims[:, 0:2] = locations
        dims[:, 2:4] = self.problem.rectangles

        # Swap x and y for rotated rects
        dims[rotations, 0] = locations[rotations, 1]
        dims[rotations, 1] = locations[rotations, 0]

        return dims
