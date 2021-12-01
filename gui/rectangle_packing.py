import pygame
import threading
import numpy as np
import json



class RectanglePackingGUI:
    def __init__(self):
        self.problem = None
        self.current_sol = None
        self.search = None  # the search algorithm routine
        self.running = True

        with open("gui/config.json") as json_data_file:
            self.config = json.load(json_data_file)

        self.render_thread = threading.Thread(target=self.__render, daemon=True)
        self.render_thread.start()


    @property
    def colors(self):
        return self.config['colors']


    def __init_gui(self):
        pygame.init()
        pygame.display.set_caption("Rectangle Packing")

        pygame.font.init()
        self.font = pygame.font.SysFont(self.config['font'], self.config['font_size'])

        self.screen = pygame.display.set_mode((self.config['window_width'], self.config['window_height']))

        self.area_width = self.screen.get_width()  # FIELD_SZ * GRID_WIDTH + LINE_WIDTH
        self.area_height = self.screen.get_height()  # FIELD_SZ * GRID_HEIGHT + LINE_WIDTH
        self.scr_marg_left = 0  # (self.screen.get_width() - self.area_width) // 2
        self.scr_marg_top = 0  # self.screen.get_height() - self.area_height - self.scr_marg_left

        self.cam_pos = np.array([0, 0])

    def set_current_solution(self, solution):
        self.current_sol = solution

    def __render(self):
        self.__init_gui()

        mouse_btn_hold = False
        old_mouse_pos = None

        while self.running:
            # Handle user input
            for event in pygame.event.get():
                # Did the user click the window close button?
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # left mousebutton
                        mouse_btn_hold = True
                        old_mouse_pos = np.asarray(pygame.mouse.get_pos())

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # left mousebutton
                        mouse_btn_hold = False

                elif mouse_btn_hold and event.type == pygame.MOUSEMOTION:
                    new_mouse_pos = np.asarray(pygame.mouse.get_pos())
                    shift_delta = new_mouse_pos - old_mouse_pos
                    self.cam_pos += shift_delta
                    old_mouse_pos = new_mouse_pos

            # self.screen.fill(self.colors['bg_color'])

            # Grid area
            pygame.draw.rect(self.screen, self.colors['grid_bg'],
                             [self.scr_marg_left - self.config['line_width'], self.scr_marg_top - self.config['line_width'],
                              self.area_width, self.area_height])

            # Empty fields
            # for y in range(GRID_HEIGHT):
            #     for x in range(GRID_WIDTH):
            #         self.draw_block(x, y)

            # Get visible grid boundary
            top_left = -self.cam_pos // self.config['field_size'] + 1
            bottom_right = (-self.cam_pos + np.asarray([self.area_width, self.area_height])) // self.config['field_size'] + 1

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
                    locations, rotations = self.current_sol
                    locations_temp = locations.copy()
                    locations_temp[rotations, 0] = locations[rotations, 1]
                    locations_temp[rotations, 1] = locations[rotations, 0]
                    for rect_idx in range(self.problem.num_rects):
                        x, y = locations_temp[rect_idx]
                        w, h = self.problem.rectangles[rect_idx]
                        self.draw_rect(x, y, w, h)

                    # highlight non-empty boxes
                    for x in range(top_left[0], bottom_right[0]):
                        if x % self.problem.box_length == 0:
                            for y in range(top_left[1], bottom_right[1]):
                                if y % self.problem.box_length == 0:
                                    if not self.is_empty_block(x, y):
                                        self.draw_box(x, y)

            # display current solution value
            textsurface = self.font.render(f'Objective Value: {self.problem.f(self.current_sol)}', False, self.colors['font'])
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


    def draw_block(self, x, y):
        pygame.draw.rect(self.screen, self.colors['grid_bg'],
                         [self.cam_pos[0] + self.scr_marg_left + self.config['line_width'] + x * self.config['field_size'],
                          self.cam_pos[1] + self.scr_marg_top + self.config['line_width'] + y * self.config['field_size'],
                          self.config['field_size'] - self.config['line_width'] * 2, self.config['field_size'] - self.config['line_width'] * 2])

    def draw_box(self, x, y):
        pygame.draw.rect(self.screen, self.colors['non_empty_boxes'],
                         [self.cam_pos[0] + self.scr_marg_left + self.config['line_width'] + x * self.config['field_size'],
                          self.cam_pos[1] + self.scr_marg_top + self.config['line_width']+ y * self.config['field_size'],
                          self.problem.box_length * self.config['field_size'] - self.config['line_width'] * 2, self.problem.box_length * self.config['field_size'] - self.config['line_width'] * 2], 1)


    def draw_rect(self, x, y, w, h):
        pygame.draw.rect(self.screen, self.colors['rectangles'],
                         [self.cam_pos[0] + self.scr_marg_left + self.config['line_width'] + x * self.config['field_size'],
                          self.cam_pos[1] + self.scr_marg_top + self.config['line_width'] + y * self.config['field_size'],
                          w * self.config['field_size'] - self.config['line_width'] * 2, h * self.config['field_size'] - self.config['line_width'] * 2])

    def draw_h_line(self, y, color):
        start = [self.scr_marg_left, self.cam_pos[1] + self.scr_marg_top + y * self.config['field_size'] - self.config['line_width']]
        end = [self.area_width, self.cam_pos[1] + self.scr_marg_top + y * self.config['field_size'] - self.config['line_width']]
        pygame.draw.line(self.screen, color, start, end, self.config['line_width'])

    def draw_v_line(self, x, color):
        start = [self.cam_pos[0] + self.scr_marg_left + x * self.config['field_size'] - self.config['line_width'], self.scr_marg_top]
        end = [self.cam_pos[0] + self.scr_marg_left + x * self.config['field_size'] - self.config['line_width'], self.area_height]
        pygame.draw.line(self.screen, color, start, end, self.config['line_width'])
