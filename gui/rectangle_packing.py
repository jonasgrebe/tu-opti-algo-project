import time
import pygame
import pygame_menu
import threading
import numpy as np
import json
import copy

from algos import local_search, greedy_search
from gui import BaseGUI

from problems.rectangle_packing.problem import (
    RectanglePackingSolution,
    RectanglePackingSolutionGeometryBased,
    RectanglePackingSolutionRuleBased,
    RectanglePackingSolutionGreedy,

    RectanglePackingProblem,
    RectanglePackingProblemGeometryBased,
    RectanglePackingProblemRuleBased,
    RectanglePackingProblemGreedy
    )

ZOOM_STEP_FACTOR = 1.1

class RectanglePackingGUI(BaseGUI):
    def __init__(self):
        super().__init__()

        # Problem constants
        self.problem_config = dict(box_length=8, num_rects=32, w_min=1, w_max=8, h_min=1, h_max=8)
        self.problem = None
        self.problem_type_name = 'rectangle_packing_geometry_based'
        self.problem_types = {
            'rectangle_packing_geometry_based': RectanglePackingProblemGeometryBased,
            'rectangle_packing_rule_based': RectanglePackingProblemRuleBased,
            'rectangle_packing_greedy': RectanglePackingProblemGreedy,
        }
        self.init_sol = None

        self.current_sol = None
        self.rect_dims = None

        # GUI constants
        self.running = True
        self.dragging_camera = False
        self.old_mouse_pos = None
        self.selected_rect_idx = None
        self.highlighted_rects = None
        self.selection_rotated = False

        with open("gui/config.json") as json_data_file:
            self.config = json.load(json_data_file)

        self.render_thread = threading.Thread(target=self.__run)
        self.render_thread.start()

        self.search_algorithm_name = 'local_search'  # ['local_search', 'greedy_search']
        self.search = local_search  # the search algorithm routine
        self.search_algorithms = {
            'local_search': local_search,
            'greedy_search': greedy_search
        }

        self.is_searching = False
        self.is_paused = False
        self.search_thread = None
        self.search_start_time = None
        self.anim_sleep = 2
        self.search_info = {}

        self.animation_on = True

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

        btn_search = self.main_menu.get_widget('run_search')
        btn_search.set_title('Run Search')

        btn_configure_problem = self.main_menu.get_widget('configure_problem')
        btn_configure_problem.readonly = False

        btn_configure_algo = self.main_menu.get_widget('configure_algo')
        btn_configure_algo.readonly = False

        # self.__toogle_animation_on(True)

    def __setup_menu(self):

        def button_onmouseover(w) -> None:
            w.set_background_color(self.colors['button_bg_hover'])

        def button_onmouseleave(w) -> None:
            w.set_background_color(self.colors['button_bg'])

        def dropselect_onmouseover(w) -> None:
            w._selection_box_bgcolor = self.colors['button_bg_hover']
            w._force_render()

        def dropselect_onmouseleave(w) -> None:
            w._selection_box_bgcolor = self.colors['button_bg']
            w._force_render()

        theme = pygame_menu.Theme(
            background_color=pygame_menu.themes.TRANSPARENT_COLOR,
            title=False,
            widget_font=pygame_menu.font.FONT_FIRACODE,
            widget_font_size=20,
            widget_font_color=self.colors['font'],
            widget_alignment=pygame_menu.locals.ALIGN_LEFT,
            widget_selection_effect=pygame_menu.widgets.NoneSelection(),
            widget_background_color=self.colors['button_bg']
        )
        self.main_menu = pygame_menu.Menu(
            width=self.screen.get_width(),
            height=self.screen.get_height(),
            mouse_motion_selection=True,
            theme=theme,
            title='',
        )

        # Problem Config Menu
        self.problem_config_menu = pygame_menu.Menu(
            width=self.screen.get_width(),
            height=self.screen.get_height(),
            mouse_motion_selection=True,
            theme=theme,
            title='Problem Configuration'
        )

        self.problem_config_frame = self.problem_config_menu.add.frame_v(width=320, height=500,
                                                                         padding=(15, 15),
                                                                         background_color=self.colors["menu_bg"],
                                                                         align=pygame_menu.locals.ALIGN_RIGHT)
        self.problem_config_frame._relax = True

        def rangeslider_box_length_onchange(s, *args) -> None:
            rangeslider_box_length = self.problem_config_menu.get_widget('rangeslider_box_length')
            box_length = int(rangeslider_box_length.get_value())
            self.__update_problem_config({'box_length': box_length})

        rangeslider_box_length = self.problem_config_menu.add.range_slider(
            'L',
            rangeslider_id='rangeslider_box_length',
            default=self.problem_config['box_length'],
            range_values=(3, 32),
            increment=1,
            onchange=rangeslider_box_length_onchange,
            shadow_width=10
        )
        self.problem_config_frame.pack(rangeslider_box_length)

        def rangeslider_num_rects_onchange(s, *args) -> None:
            rangeslider_num_rects = self.problem_config_menu.get_widget('rangeslider_num_rects')
            num_rects = int(rangeslider_num_rects.get_value())
            self.__update_problem_config({'num_rects': num_rects})

        rangeslider_num_rects = self.problem_config_menu.add.range_slider(
            '# rects',
            rangeslider_id='rangeslider_num_rects',
            default=self.problem_config['num_rects'],
            range_values=(1, 1000),
            increment=1,
            onchange=rangeslider_num_rects_onchange,
            shadow_width=10
        )
        self.problem_config_frame.pack(rangeslider_num_rects, margin=(0, 15))

        def rangeslider_w_min_onchange(s, *args) -> None:
            rangeslider_w_min = self.problem_config_menu.get_widget('rangeslider_w_min')
            w_min = int(rangeslider_w_min.get_value())
            self.__update_problem_config({'w_min': w_min})

        rangeslider_w_min = self.problem_config_menu.add.range_slider(
            'w_min',
            rangeslider_id='rangeslider_w_min',
            default=self.problem_config['w_min'],
            range_values=(1, self.problem_config['w_max']),
            increment=1,
            onchange=rangeslider_w_min_onchange,
            shadow_width=10
        )
        self.problem_config_frame.pack(rangeslider_w_min, margin=(0, 35))

        def rangeslider_w_max_onchange(s, *args) -> None:
            rangeslider_w_max = self.problem_config_menu.get_widget('rangeslider_w_max')
            w_max = int(rangeslider_w_max.get_value())
            self.__update_problem_config({'w_max': w_max})

        rangeslider_w_max = self.problem_config_menu.add.range_slider(
            'w_max',
            rangeslider_id='rangeslider_w_max',
            default=self.problem_config['w_max'],
            range_values=(self.problem_config['w_min'] + 1, self.problem_config['box_length']),
            increment=1,
            onchange=rangeslider_w_max_onchange,
            shadow_width=10
        )
        self.problem_config_frame.pack(rangeslider_w_max, margin=(0, 15))

        def rangeslider_h_min_onchange(s, *args) -> None:
            rangeslider_h_min = self.problem_config_menu.get_widget('rangeslider_h_min')
            h_min = int(rangeslider_h_min.get_value())
            self.__update_problem_config({'h_min': h_min})

        rangeslider_h_min = self.problem_config_menu.add.range_slider(
            'h_min',
            rangeslider_id='rangeslider_h_min',
            default=self.problem_config['h_min'],
            range_values=(1, self.problem_config['h_max']),
            increment=1,
            onchange=rangeslider_h_min_onchange,
            shadow_width=10
        )
        self.problem_config_frame.pack(rangeslider_h_min, margin=(0, 35))

        def rangeslider_h_max_onchange(s, *args) -> None:
            rangeslider_h_max = self.problem_config_menu.get_widget('rangeslider_h_max')
            h_max = int(rangeslider_h_max.get_value())
            self.__update_problem_config({'h_max': h_max})

        rangeslider_h_max = self.problem_config_menu.add.range_slider(
            'h_max',
            rangeslider_id='rangeslider_h_max',
            default=self.problem_config['h_max'],
            range_values=(self.problem_config['h_min'] + 1, self.problem_config['box_length']),
            increment=1,
            onchange=rangeslider_h_max_onchange,
            shadow_width=10
        )
        self.problem_config_frame.pack(rangeslider_h_max, margin=(0, 15))

        btn_close_problem_config_menu = self.problem_config_menu.add.button(
            title='Apply',
            action=pygame_menu.events.BACK,
            button_id='close_config_menu',
            shadow_width=10
        )
        btn_close_problem_config_menu.set_onmouseover(lambda: button_onmouseover(btn_close_problem_config_menu))
        btn_close_problem_config_menu.set_onmouseleave(lambda: button_onmouseleave(btn_close_problem_config_menu))
        self.problem_config_frame.pack(btn_close_problem_config_menu, margin=(0, 35))


        # Algorithm Config Menu
        self.algo_config_menu = pygame_menu.Menu(
            width=self.screen.get_width(),
            height=self.screen.get_height(),
            mouse_motion_selection=True,
            theme=theme,
            title='Algorithm Configuration'
        )

        self.algo_config_frame = self.algo_config_menu.add.frame_v(width=350, height=400,
                                                                        padding=(15, 15),
                                                                         background_color=self.colors["menu_bg"],
                                                                         align=pygame_menu.locals.ALIGN_RIGHT)
        self.algo_config_menu._relax = True


        label = self.algo_config_menu.add.label("Neighborhood", label_id="neighborhood_label",
                                                      background_color=pygame_menu.themes.TRANSPARENT_COLOR)

        self.algo_config_frame.pack(label, margin=(0, 15))

        def dropselect_neighborhood_onchange(s, *args) -> None:
            self.stop_search()
            self.problem_type_name = args[0]
            self.__setup_new_problem()

            #rangeslider_overlap = self.algo_config_menu.get_widget('rangeslider_overlap')
            #rangeslider_penalty = self.algo_config_menu.get_widget('rangeslider_penalty')
            btn_relaxation = self.algo_config_menu.get_widget('toggle_relaxation')

            #if self.problem_type_name == 'rectangle_packing_geometry_based':
            #    rangeslider_overlap.show()
            #    #rangeslider_penalty.show()
            #else:
            #    rangeslider_overlap.hide()
            #    #rangeslider_penalty.hide()

            if self.problem_type_name == 'rectangle_packing_geometry_based':
                btn_relaxation.show()
            else:
                btn_relaxation.hide()

        dropselect_neighborhood = self.algo_config_menu.add.dropselect(
            title='',
            items=[
                ('Geometry-based', 'rectangle_packing_geometry_based'),
                ('Rule-based', 'rectangle_packing_rule_based'),
            ],
            dropselect_id='neighborhood',
            onchange=dropselect_neighborhood_onchange,
            default=0,
            padding=0,
            placeholder='Select',
            selection_box_height=5,
            selection_box_width=250,
            selection_box_inflate=(5, 15),
            selection_box_margin=0,
            selection_box_border_color=self.colors['button_bg'],
            selection_option_font_color=self.colors['font'],
            selection_box_bgcolor=self.colors['button_bg'],
            shadow_width=10
        )
        dropselect_neighborhood.set_onmouseover(lambda: dropselect_onmouseover(dropselect_neighborhood))
        dropselect_neighborhood.set_onmouseleave(lambda: dropselect_onmouseleave(dropselect_neighborhood))
        self.algo_config_frame.pack(dropselect_neighborhood, margin=(15, 0))

        def toggle_relaxation():
            if self.is_searching:
                return

            btn_relaxation = self.algo_config_menu.get_widget('toggle_relaxation')
            self.problem.toggle_relaxation()
            if self.problem.is_relaxation_enabled():
                btn_relaxation.set_title("Disable Relaxation")
            else:
                btn_relaxation.set_title("Enable Relaxation")


        btn_relaxation = self.algo_config_menu.add.button(
            'Enable Relaxation',
            toggle_relaxation,
            button_id='toggle_relaxation',
            shadow_width=10
        )
        btn_relaxation.set_onmouseover(lambda: button_onmouseover(btn_relaxation))
        btn_relaxation.set_onmouseleave(lambda: button_onmouseleave(btn_relaxation))
        self.algo_config_frame.pack(btn_relaxation, margin=(0, 15))

        """
        def rangeslider_overlap_onchange(s, *args) -> None:

            rangeslider_overlap = self.algo_config_menu.get_widget('rangeslider_overlap')
            self.problem.allowed_overlap = rangeslider_overlap.get_value()

        rangeslider_overlap = self.algo_config_menu.add.range_slider(
            'Overlap',
            rangeslider_id='rangeslider_overlap',
            default=1.0,
            range_values=(0, 1),
            increment=0.01,
            onchange=rangeslider_overlap_onchange,
            shadow_width=10
        )
        self.algo_config_frame.pack(rangeslider_overlap, margin=(0, 15))


        def rangeslider_penalty_onchange(s, *args) -> None:
            assert isinstance(self.problem, RectanglePackingProblemGeometryBased)

            rangeslider_penalty = self.main_menu.get_widget('rangeslider_penalty')
            self.problem.penalty_factor = rangeslider_penalty.get_value()

        rangeslider_penalty = self.main_menu.add.range_slider(
            'Penalty',
            rangeslider_id='rangeslider_penalty',
            default=0.0,
            range_values=(0, 1000),
            increment=0.1,
            onchange=rangeslider_penalty_onchange,
            shadow_width=10
        )
        rangeslider_penalty.hide()
        self.main_frame.pack(rangeslider_penalty, margin=(0, 15))
        """

        label = self.algo_config_menu.add.label("Strategy",
                                         label_id="selection_strategy_label",
                                         background_color=pygame_menu.themes.TRANSPARENT_COLOR)
        label.hide()
        self.algo_config_frame.pack(label, margin=(0, 15))

        def dropselect_selection_strategy_onchange(s, *args) -> None:
            self.stop_search()

            assert isinstance(self.problem, RectanglePackingProblemGreedy)

            dropselect_selection_strategy = self.algo_config_menu.get_widget('selection_strategy')
            self.problem.set_cost_strategy(args[0])

        dropselect_selection_strategy = self.algo_config_menu.add.dropselect(
            title='',
            items=[
                ('Position', 'smallest_position_costs_strategy'),
                ('Largest Area', 'largest_area_costs_strategy'),
                ('Position + Largest Area', 'smallest_position_plus_largest_area_costs_strategy'),
                ('Uniform', 'uniform_costs_strategy'),
                ('Lowest Box ID', 'lowest_box_id_costs_strategy')
            ],
            dropselect_id='selection_strategy',
            onchange=dropselect_selection_strategy_onchange,
            default=0,
            padding=0,
            placeholder='Select',
            selection_box_height=5,
            selection_box_width=250,
            selection_box_inflate=(5, 15),
            selection_box_margin=0,
            selection_box_border_color=self.colors['button_bg'],
            selection_option_font_color=self.colors['font'],
            selection_box_bgcolor=self.colors['button_bg'],
            shadow_width=10
        )
        dropselect_selection_strategy.set_onmouseover(lambda: dropselect_onmouseover(dropselect_selection_strategy))
        dropselect_selection_strategy.set_onmouseleave(lambda: dropselect_onmouseleave(dropselect_selection_strategy))
        dropselect_selection_strategy.hide()
        self.algo_config_frame.pack(dropselect_selection_strategy, margin=(15, 0))


        label = self.algo_config_menu.add.label("Heuristic", label_id="heuristic_label",
                                                      background_color=pygame_menu.themes.TRANSPARENT_COLOR)
        self.algo_config_frame.pack(label, margin=(0, 15))

        def dropselect_heuristic_onchange(s, *args) -> None:
            self.is_paused = True
            self.stop_search()
            self.problem.set_heuristic(args[0])

        dropselect_heuristic = self.algo_config_menu.add.dropselect(
            title='',
            items=[
                ('Rectangle Count', 'rectangle_count_heuristic'),
                ('Box Occupancy', 'box_occupancy_heuristic'),
                ('Small Box Position', 'small_box_position_heuristic')
            ],
            dropselect_id='heuristic',
            onchange=dropselect_heuristic_onchange,
            default=1,
            padding=0,
            placeholder='Select',
            selection_box_height=5,
            selection_box_width=250,
            selection_box_inflate=(5, 15),
            selection_box_margin=0,
            selection_box_border_color=self.colors['button_bg'],
            selection_option_font_color=self.colors['font'],
            selection_box_bgcolor=self.colors['button_bg'],
            shadow_width=10
        )
        dropselect_heuristic.set_onmouseover(lambda: dropselect_onmouseover(dropselect_heuristic))
        dropselect_heuristic.set_onmouseleave(lambda: dropselect_onmouseleave(dropselect_heuristic))
        self.algo_config_frame.pack(dropselect_heuristic, margin=(15, 0))


        btn_close_algo_config_menu = self.algo_config_menu.add.button(
            title='Apply',
            action=pygame_menu.events.BACK,
            button_id='close_algo_config_menu',
            shadow_width=10
        )
        btn_close_algo_config_menu.set_onmouseover(lambda: button_onmouseover(btn_close_algo_config_menu))
        btn_close_algo_config_menu.set_onmouseleave(lambda: button_onmouseleave(btn_close_algo_config_menu))
        self.algo_config_frame.pack(btn_close_algo_config_menu, margin=(0, 35))

        # Main Menu:

        self.main_frame = self.main_menu.add.frame_v(width=270, height=575,
                                                     padding=(15, 15),
                                                     background_color=self.colors["menu_bg"],
                                                     align=pygame_menu.locals.ALIGN_RIGHT)
        self.main_frame._relax = True

        btn_configure = self.main_menu.add.button(
            title='Configure Problem',
            action=self.problem_config_menu,
            button_id='configure_problem',
            shadow_width=10
        )
        btn_configure.set_onmouseover(lambda: button_onmouseover(btn_configure))
        btn_configure.set_onmouseleave(lambda: button_onmouseleave(btn_configure))
        self.main_frame.pack(btn_configure, margin=(0, 15))

        def generate_instance():
            self.stop_search()  # IMPORTANT!
            self.__setup_new_problem(new_instance=True)

        btn_generate = self.main_menu.add.button(
            'Generate Instance',
            generate_instance,
            button_id='generate_instance',
            shadow_width=10
        )
        btn_generate.set_onmouseover(lambda: button_onmouseover(btn_generate))
        btn_generate.set_onmouseleave(lambda: button_onmouseleave(btn_generate))
        self.main_frame.pack(btn_generate, margin=(0, 15))

        self.main_frame.pack(self.main_menu.add.label("Algorithm",
                                                      background_color=pygame_menu.themes.TRANSPARENT_COLOR),
                             margin=(0, 40))

        def dropselect_algorithm_onchange(s, *args) -> None:
            self.stop_search()

            btn_search = self.main_menu.get_widget('run_search')
            self.search_algorithm_name = args[0]
            self.search = self.search_algorithms[self.search_algorithm_name]


            dropselect_neighborhood = self.algo_config_menu.get_widget('neighborhood')
            dropselect_neighborhood = self.algo_config_menu.get_widget('neighborhood')
            dropselect_neighborhood_label = self.algo_config_menu.get_widget('neighborhood_label')
            dropselect_selection_strategy = self.algo_config_menu.get_widget('selection_strategy')
            dropselect_selection_strategy_label = self.algo_config_menu.get_widget('selection_strategy_label')

            btn_relaxation = self.algo_config_menu.get_widget('toggle_relaxation')

            #rangeslider_overlap = self.algo_config_menu.get_widget('rangeslider_overlap')

            if self.search_algorithm_name == 'local_search':
                self.problem_type_name = dropselect_neighborhood.get_value()[0][1]

                dropselect_neighborhood.show()
                dropselect_neighborhood_label.show()
                dropselect_selection_strategy.hide()
                dropselect_selection_strategy_label.hide()

                if not isinstance(self.problem, RectanglePackingSolutionRuleBased):
                    btn_relaxation.show()

            elif self.search_algorithm_name == 'greedy_search':
                self.problem_type_name = 'rectangle_packing_greedy'

                dropselect_neighborhood.hide()
                dropselect_neighborhood_label.hide()
                dropselect_selection_strategy.show()
                dropselect_selection_strategy_label.show()
                btn_relaxation.hide()

            self.__setup_new_problem()

            if self.search_algorithm_name == 'greedy_search':
                cost_strategy_name = dropselect_selection_strategy.get_value()[0][1]
                self.problem.set_cost_strategy(cost_strategy_name)

            heuristic_name = dropselect_heuristic.get_value()[0][1]
            self.problem.set_heuristic(heuristic_name)

            # btn_search.readonly = False
            btn_search.is_selectable = True
            btn_search.set_cursor(pygame_menu.locals.CURSOR_HAND)

        dropselect_algorithm = self.main_menu.add.dropselect(
            title='',
            items=[
                ('Local Search', 'local_search'),
                ('Greedy Search', 'greedy_search'),
            ],
            dropselect_id='algorithm',
            onchange=dropselect_algorithm_onchange,
            default=0,
            padding=0,
            placeholder='Select',
            selection_box_height=5,
            selection_box_width=220,
            selection_box_inflate=(5, 15),
            selection_box_margin=0,
            selection_box_border_color=self.colors['button_bg'],
            selection_option_font_color=self.colors['font'],
            selection_box_bgcolor=self.colors['button_bg'],
            shadow_width=10
        )
        dropselect_algorithm.set_onmouseover(lambda: dropselect_onmouseover(dropselect_algorithm))
        dropselect_algorithm.set_onmouseleave(lambda: dropselect_onmouseleave(dropselect_algorithm))
        self.main_frame.pack(dropselect_algorithm, margin=(15, 0))

        btn_configure = self.main_menu.add.button(
            title='Configure Algorithm',
            action=self.algo_config_menu,
            button_id='configure_algo',
            shadow_width=10
        )
        btn_configure.set_onmouseover(lambda: button_onmouseover(btn_configure))
        btn_configure.set_onmouseleave(lambda: button_onmouseleave(btn_configure))
        self.main_frame.pack(btn_configure, margin=(0, 15))

        def run_search():
            btn_search = self.main_menu.get_widget('run_search')
            if not self.is_searching:
                self.__start_search()
            else:
                self.is_paused = True
                self.stop_search()

        btn_search = self.main_menu.add.button(
            'Run Search',
            run_search,
            button_id='run_search',
            shadow_width=10
        )

        # btn_search.readonly = True
        btn_search.set_onmouseover(lambda: button_onmouseover(btn_search))
        btn_search.set_onmouseleave(lambda: button_onmouseleave(btn_search))
        self.main_frame.pack(btn_search, margin=(0, 40))

        def rangeslider_anim_speed_onchange(s, *args) -> None:
            rangeslider_animation_speed = self.main_menu.get_widget('rangeslider_animation_speed')
            MAX_SLEEP_IN_SEC = 4
            value = int(rangeslider_animation_speed.get_value())
            self.anim_sleep = MAX_SLEEP_IN_SEC * 1 / (value + 0.01)
            rangeslider_animation_speed.set_value(value)


        rangeslider_anim_speed = self.main_menu.add.range_slider(
            'Speed',
            rangeslider_id='rangeslider_animation_speed',
            default=self.anim_sleep,
            range_values=(1, 100),
            increment=1,
            onchange=rangeslider_anim_speed_onchange,
            shadow_width=10
        )
        self.main_frame.pack(rangeslider_anim_speed, margin=(0, 15))

        def reset_search():
            # btn_reset = self.menu.get_widget('reset_search')

            if self.is_searching:
                self.stop_search()
                self.search_thread.join()

            self.search_start_time = None
            self.search_stop_time = None

            self.set_current_solution(copy.deepcopy(self.init_sol))

        btn_animation = self.main_menu.add.button(
            'Disable Animation',
            self.__toogle_animation_on,
            button_id='btn_animation',
            shadow_width=10
        )
        btn_animation.set_onmouseover(lambda: button_onmouseover(btn_animation))
        btn_animation.set_onmouseleave(lambda: button_onmouseleave(btn_animation))
        self.main_frame.pack(btn_animation, margin=(0, 15))

        btn_reset = self.main_menu.add.button(
            'Reset Search',
            reset_search,
            button_id='reset_search',
            shadow_width=10
        )
        btn_reset.set_onmouseover(lambda: button_onmouseover(btn_reset))
        btn_reset.set_onmouseleave(lambda: button_onmouseleave(btn_reset))
        self.main_frame.pack(btn_reset, margin=(0, 15))

        btn_exit = self.main_menu.add.button(
            'Exit',
            pygame_menu.events.EXIT,
            button_id='exit',
            shadow_width=10
        )
        btn_exit.set_onmouseover(lambda: button_onmouseover(btn_exit))
        btn_exit.set_onmouseleave(lambda: button_onmouseleave(btn_exit))
        self.main_frame.pack(btn_exit, margin=(0, 40))

    def __toogle_animation_on(self, value=None):
        btn_animation = self.main_menu.get_widget('btn_animation')
        rangeslider_anim_speed = self.main_menu.get_widget('rangeslider_animation_speed')

        if value == None:
            self.animation_on = not self.animation_on
        else:
            self.animation_on = value

        if self.animation_on:
            btn_animation.set_title("Disable Animation")
            rangeslider_anim_speed.show()
        else:
            btn_animation.set_title("Enable Animation")
            rangeslider_anim_speed.hide()


    def __mouse_over_menu(self):
        x, y, w, h = self.__get_menu_bounds()
        mouse_x, mouse_y = pygame.mouse.get_pos()
        return x <= mouse_x < x + w and y <= mouse_y < y + h

    def __get_menu_bounds(self):
        current_menu = self.main_menu.get_current()
        displayed_frame = self.main_frame if current_menu == self.main_menu else self.problem_config_frame
        x, y = displayed_frame.get_position(apply_padding=True)
        w = displayed_frame.get_width()
        h = displayed_frame.get_height()
        return x, y, w, h

    def __render_rectangle_preview(self):
        if self.problem_type_name not in ('rectangle_packing_greedy', 'rectangle_packing_rule_based'):
            return

        bg_color = [0, 0, 0]

        margin_top = self.field_size
        margin_bot = self.field_size
        margin_left = self.field_size
        margin_vertical = self.field_size

        reference_size = self.problem.h_max
        preview_height = reference_size * self.field_size + margin_top + margin_bot
        y_offset = self.screen.get_height() - preview_height

        pygame.draw.rect(self.screen, bg_color, [0, y_offset, self.screen.get_width(), preview_height])

        x = margin_left
        y = y_offset + margin_top

        rect_order = range(self.problem.num_rects) if self.problem_type_name == 'rectangle_packing_greedy' else self.current_sol.rect_order

        for rect_idx in rect_order:
            w, h = self.problem.sizes[rect_idx]

            if self.current_sol.is_put[rect_idx] and self.problem_type_name == 'rectangle_packing_greedy':
                continue

            if x + w * self.field_size + margin_vertical >= self.screen.get_width():
                break

            if self.highlighted_rects[rect_idx]:
                color = self.colors['highlighted_rectangle']
            else:
                color = self.colors['rectangles']

            pygame.draw.rect(self.screen, color,
                             [x, y + (reference_size - h) * self.field_size, w * self.field_size, h * self.field_size])
            x += w * self.field_size + margin_vertical

        if self.problem_type_name == "rectangle_packing_greedy":
            if 'num_remaining_elements' in self.search_info:
                num_remaining_elements = self.search_info['num_remaining_elements']
            else:
                num_remaining_elements = 0

            text = f"Remaining Elements: {num_remaining_elements}"
            width = self.font.size(text)[0]

            pygame.draw.rect(self.screen, bg_color, [0, y_offset - 40, 30 + width, 40])
            text_surface = self.font.render(text, True, self.colors['font'])
            self.screen.blit(text_surface, (15, y_offset - 30))

    def update_search_info(self, update_dict):
        self.search_info.update(update_dict)

    def __start_search(self):
        self.is_searching = True
        self.search_thread = threading.Thread(target=self.search, args=(self.problem, self))
        self.search_thread.start()

        if self.is_paused:
            self.is_paused = False
        else:
            self.search_start_time = time.time()
            self.search_info = {}

        btn_search = self.main_menu.get_widget('run_search')
        btn_search.set_title('Pause Search')

        btn_configure_problem = self.main_menu.get_widget('configure_problem')
        btn_configure_problem.readonly = True

        btn_configure_algo = self.main_menu.get_widget('configure_algo')
        btn_configure_algo.readonly = True

    def __setup_new_problem(self, new_instance=False):

        if not new_instance:
            instance_params = self.problem.get_instance_params()

        if self.problem is not None:
            relaxation_enabled = self.problem.is_relaxation_enabled()
        else:
            relaxation_enabled = False

        self.problem = self.problem_types[self.problem_type_name](**self.problem_config)

        if not new_instance:
            self.problem.set_instance_params(*instance_params)

        if relaxation_enabled:
            self.problem.toggle_relaxation()

        if isinstance(self.problem, RectanglePackingProblemGreedy):
            sol = self.problem.get_empty_solution()
        else:
            sol = self.problem.get_arbitrary_solution()

        self.init_sol = copy.deepcopy(sol)
        self.set_current_solution(sol)
        self.highlighted_rects = np.zeros(self.problem.num_rects, dtype=np.bool)

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
        rangeslider_box_length = self.problem_config_menu.get_widget('rangeslider_box_length')
        rangeslider_num_rects = self.problem_config_menu.get_widget('rangeslider_num_rects')
        rangeslider_w_min = self.problem_config_menu.get_widget('rangeslider_w_min')
        rangeslider_w_max = self.problem_config_menu.get_widget('rangeslider_w_max')
        rangeslider_h_min = self.problem_config_menu.get_widget('rangeslider_h_min')
        rangeslider_h_max = self.problem_config_menu.get_widget('rangeslider_h_max')

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
                self.__setup_new_problem(new_instance=True)
                return

    def resize_window(self, w, h):
        pygame.display.set_mode((w, h), pygame.RESIZABLE)
        self.area_width = self.screen.get_width()
        self.area_height = self.screen.get_height()

        self.main_menu.resize(w, h, position=(1, 1, False))
        self.problem_config_menu.resize(w, h, position=(1, 1, False))
        self.algo_config_menu.resize(w, h, position=(1, 1, False))

    def set_current_solution(self, solution: RectanglePackingSolution):
        if isinstance(solution, RectanglePackingSolutionGeometryBased):
            solution.apply_pending_move()

        self.current_sol = solution
        self.rect_dims = self.get_rect_dimensions()

    def get_current_solution(self):
        return self.current_sol

    def get_rect_info(self, rect_idx):
        """rect_idx can also be a list of indices."""
        x, y = self.current_sol.locations[rect_idx].T
        w, h = self.problem.sizes[rect_idx].T
        rotations = self.current_sol.rotations[rect_idx]
        if np.isscalar(rect_idx):
            if rotations:
                w, h = h, w
        else:
            w[rotations], h[rotations] = h[rotations], w[rotations]
        return x, y, w, h

    def set_and_animate_solution(self, sol: RectanglePackingSolution):

        # Identify modified rect and highlight it
        if isinstance(sol, RectanglePackingSolutionGeometryBased) and sol.move_pending:
            changed_rect_idx = sol.pending_move_params[0]
            self.highlighted_rects[changed_rect_idx] = True
        elif isinstance(sol, RectanglePackingSolutionRuleBased):
            diff = (sol.rect_order != self.current_sol.rect_order)
            self.highlighted_rects[sol.moved_rect_ids] = True
        elif isinstance(sol, RectanglePackingSolutionGreedy):
            self.highlighted_rects[sol.last_put_rect] = True

        if self.animation_on:
            time.sleep(self.anim_sleep / 2)

        # Apply new solution
        self.set_current_solution(sol)

        if self.animation_on:
            time.sleep(self.anim_sleep / 2)

        # Unhighlight the changed rects
        self.highlighted_rects[:] = 0

    def __run(self):
        self.__init_gui()
        self.__setup_menu()
        self.__setup_new_problem(new_instance=True)

        frame = 0
        while self.running:
            self.__handle_user_input()
            self.__render()

            self.frame_times[frame % 2000] = time.time()
            frame += 1

        pygame.quit()

    def __handle_user_input(self):
        mouse_pos = np.asarray(pygame.mouse.get_pos())
        x, y = self.mouse_pos_to_field_coords(mouse_pos)

        rect_idx = self.get_rect_idx_at(x, y)

        if rect_idx is not None and not self.__mouse_over_menu():
            pygame.mouse.set_cursor(*pygame.cursors.broken_x)
        else:
            pygame.mouse.set_cursor(*pygame.cursors.arrow)

        events = pygame.event.get()
        if self.main_menu.is_enabled():
            self.main_menu.update(events)

        for event in events:
            # Did the user click the window close button?
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.__mouse_over_menu():
                    continue

                if event.button == 2:  # center mousebutton
                    self.dragging_camera = True
                    self.old_mouse_pos = mouse_pos

                elif event.button == 1 and isinstance(self.problem, RectanglePackingProblemGeometryBased):  # left mousebutton
                    if self.selected_rect_idx is None:
                        self.selected_rect_idx = rect_idx
                        self.selection_rotated = False
                    else:
                        new_solution = self.current_sol.copy(True)
                        rotated = self.selection_rotated != new_solution.rotations[self.selected_rect_idx]
                        new_solution.move_rect(self.selected_rect_idx, np.array([x, y]), rotated)
                        if self.problem.is_feasible(new_solution):
                            if isinstance(new_solution, RectanglePackingSolutionGeometryBased):
                                new_solution.apply_pending_move()
                            self.set_current_solution(new_solution)
                            self.selected_rect_idx = None

                elif event.button == 3 and isinstance(self.problem, RectanglePackingProblemGeometryBased):  # right mousebutton
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
        # Screen area
        pygame.draw.rect(self.screen, self.colors['grid_bg'],
                         [self.scr_marg_left, self.scr_marg_top,
                          self.area_width, self.area_height])

        # Get visible grid boundary
        top_left = -self.cam_pos // self.field_size + 1
        top_left = top_left.astype(np.int32)
        bottom_right = (-self.cam_pos + np.asarray([self.area_width, self.area_height])) // self.field_size + 1
        bottom_right = bottom_right.astype(np.int32)

        if self.animation_on or not self.is_searching:
            self.highlight_non_empty_boxes(top_left, bottom_right)
            self.draw_fine_grid(top_left, bottom_right)

        if self.problem is not None:
            self.draw_box_grid(top_left, bottom_right)

            if self.current_sol is not None and (self.animation_on or not self.is_searching):
                self.draw_rects(top_left, bottom_right)

        if self.animation_on or not self.is_searching:
            self.highlight_overlapping_fields()
        self.draw_hover_shape()

        if self.problem is not None and self.current_sol is not None:
            self.draw_text_info()

        if self.main_menu.is_enabled():
            self.main_menu.draw(self.screen)

        if self.animation_on or not self.is_searching:
            self.__render_rectangle_preview()

        # Update the screen
        pygame.display.flip()

    def highlight_non_empty_boxes(self, view_top_left, view_bottom_right):
        l = self.problem.box_length
        # occupied_boxes = self.current_sol.box_coords[self.current_sol.box_occupancies > 0]
        visible_boxes = self.get_visible_boxes(view_top_left, view_bottom_right)
        for (x, y) in visible_boxes:
            self.draw_rect(x * l, y * l, l, l, color=self.colors['non_empty_boxes'])


    def highlight_overlapping_fields(self):
        if self.current_sol.boxes_grid.max() <= 1:
            return

        fs = self.field_size
        l = self.problem.box_length

        b, x, y = np.where(self.current_sol.boxes_grid > 1)

        for (bf, xf, yf) in zip(b, x, y):

            bfx, bfy = self.current_sol.box_coords[bf]
            overlaps =  self.current_sol.boxes_grid[bf, xf, yf] - 1

            min_color = self.colors['min_overlap'].copy()
            max_color = self.colors['max_overlap']

            overlaps = min(overlaps, 7)
            color = [int(min + (overlaps / 7) * (max - min)) for min, max in zip(min_color, max_color)]

            self.draw_rect(bfx * l + xf, bfy * l + yf, 1, 1, color=color)


    def get_visible_boxes(self, view_top_left, view_bottom_right):
        l = self.problem.box_length
        occupied_boxes = self.current_sol.box_coords[self.current_sol.box_occupancies > 0]
        left, top = view_top_left
        right, bottom = view_bottom_right
        xs, ys = occupied_boxes.T * l
        visible_boxes = ((left <= xs + l) & (xs < right)) & \
                        ((top <= ys + l) & (ys < bottom))
        return occupied_boxes[visible_boxes]

    def draw_rects(self, view_top_left, view_bottom_right):
        # Identify visible rects
        x, y, w, h = self.get_rect_info(range(self.problem.num_rects))

        view_left, view_top = view_top_left
        view_right, view_bottom = view_bottom_right

        inside_view = (view_left <= x + w) & (x < view_right) & \
                      (view_top <= y + h) & (y < view_bottom)

        visible_rect_ids = np.where(inside_view)[0]
        visible_rect_ids = visible_rect_ids[self.current_sol.is_put[visible_rect_ids]]

        for rect_idx in visible_rect_ids:
            x, y, w, h = self.get_rect_info(rect_idx)
            color = self.colors['rectangles_search'] if self.is_searching else self.colors['rectangles']
            if rect_idx == self.selected_rect_idx:
                color = self.colors['active_rectangle']
            elif self.highlighted_rects[rect_idx]:
                color = self.colors['highlighted_rectangle']
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
        positions_ax2 = np.tile([one_side, other_side, other_side, one_side], len(positions) // 2 + 1)[
                        :len(positions_ax1)]
        if not vertical:
            positions_ax1, positions_ax2 = positions_ax2, positions_ax1
        point_list = np.stack([positions_ax1, positions_ax2], axis=1)
        pygame.draw.lines(self.screen, color, False, point_list, self.config['line_width'])

    def draw_hover_shape(self):
        if not self.__mouse_over_menu():
            mouse_pos = np.asarray(pygame.mouse.get_pos())
            x, y = self.mouse_pos_to_field_coords(mouse_pos)
            rect_under_mouse = self.get_rect_idx_at(x, y)

            if self.selected_rect_idx is not None:
                _, _, w, h = self.get_rect_info(self.selected_rect_idx)
                if self.selection_rotated:
                    w, h = h, w
            elif rect_under_mouse is not None:
                x, y, w, h = self.get_rect_info(rect_under_mouse)
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

        text_surface = self.font.render(f'Heuristic Value: %.2f' % heuristic, True, self.colors['font'])
        self.screen.blit(text_surface, (32, 70))

        if isinstance(self.problem, RectanglePackingProblemGeometryBased):

            if self.problem.is_relaxation_enabled():
                penalty = self.problem.penalty(self.current_sol)

                text_surface = self.font.render(f'Penalty Value: %.2f' % penalty, True, self.colors['font'])
                self.screen.blit(text_surface, (32, 108))
                text_surface = self.font.render(f'Allowed Overlap: %.3f' % self.problem.allowed_overlap, True, self.colors['font'])
                self.screen.blit(text_surface, (32, 250))
                text_surface = self.font.render(f'Penalty Factor: %.3f' % self.problem.penalty_factor, True, self.colors['font'])
                self.screen.blit(text_surface, (32, 288))


        if self.search_start_time is not None:
            if self.is_searching:
                elapsed = time.time() - self.search_start_time
            else:
                elapsed = self.search_stop_time - self.search_start_time
        else:
            elapsed = 0

        minutes = elapsed // 60
        seconds = int(elapsed) % 60
        milliseconds = int(elapsed * 1000) % 1000
        text_surface = self.font.render('Elapsed Time: %d:%02d:%03d min' % (minutes, seconds, milliseconds),
                                        True, self.colors['font'])
        self.screen.blit(text_surface, ((self.screen.get_width() - self.font.size('Elapsed Time:')[0]) // 2, 32))

        # Display if current solution is optimal
        if self.problem.is_optimal(self.current_sol):
            text_surface = self.font.render('This solution is optimal.', True, self.colors['font'])
            self.screen.blit(text_surface, (32, 204))

        # FPS
        t = time.time()
        during_last_sec = (t - self.frame_times) < 1
        fps = np.sum(during_last_sec)
        text = 'FPS: %d' % int(fps)
        text_surface = self.font.render(text, True, self.colors['font'])
        self.screen.blit(text_surface, (self.screen.get_width() -  self.font.size(text)[0] - 32, 32))

    def mouse_pos_to_field_coords(self, mouse_pos):
        field_y = mouse_pos[1] - self.scr_marg_top - self.cam_pos[1]
        field_x = mouse_pos[0] - self.scr_marg_left - self.cam_pos[0]
        y_coord = int(field_y // self.field_size)
        x_coord = int(field_x // self.field_size)
        return x_coord, y_coord

    def get_rect_idx_at(self, x, y):
        if self.current_sol is None:
            return None

        xs, ys, ws, hs = self.get_rect_info(range(self.problem.num_rects))

        result = np.where((xs <= x) &
                          (x < xs + ws) &
                          (ys <= y) &
                          (y < ys + hs))[0]

        if len(result) == 0:
            return None
        else:
            return result[0]

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
