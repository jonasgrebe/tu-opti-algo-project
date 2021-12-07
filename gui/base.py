import pygame
import pygame_menu
import threading
import numpy as np
import json

class BaseGUI:

    def __init__(self):
        pass

    def __init_gui(self):
        pass

    def resize_window(self, w, h):
        pass

    def set_current_solution(self, solution):
        pass

    def __render(self):
        pass

    def __handle_user_input(self, mouse_pos, x, y):
        pass