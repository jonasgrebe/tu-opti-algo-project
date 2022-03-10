# Interactive GUI for the Rectangle Packing Problem
![GUI screenshot](https://github.com/jonasgrebe/tu-opti-algo-project/blob/main/docs/automatic_solve.gif)

## Rectangle Packing Problem (RPP)
The _Rectangle Packing Problem (RPP)_ describes the issue of putting n (almost arbitrary) rectangles into N squares (boxes) such that N is minimized.

## Features
### Solving the RPP automatically
![Automatic RPP solving](https://github.com/jonasgrebe/tu-opti-algo-project/blob/main/docs/automatic_solve.gif)
![Automatic RPP solving](https://github.com/jonasgrebe/tu-opti-algo-project/blob/main/docs/automatic_solve_2.gif)
Automatic solving using one of two different and efficiently implemented algorithmic approaches (local search and greedy search) with multiple tuning possibilities

### Solving the RPP by hand
![Manual RPP solving](https://github.com/jonasgrebe/tu-opti-algo-project/blob/main/docs/manual_solve.gif)

### Defining an instance of the RPP
By setting the hyperparameters such as number and sizes of rectangles, box size, etc.
![Problem configuration](https://github.com/jonasgrebe/tu-opti-algo-project/blob/main/docs/configuration.gif)


## How to use
Just execute `run_gui.py`.

## Implementation
This GUI is implemented using PyGame (for the grid world) and PyGame Menu (for the UI).
