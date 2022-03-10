# Interactive GUI for the Rectangle Packing Problem
![alt text](https://github.com/jonasgrebe/tu-opti-algo-project/docs/screenshot.png "GUI screenshot")

## Rectangle Packing Problem (RPP)
The _Rectangle Packing Problem (RPP)_ describes the issue of putting n (almost arbitrary) rectangles into N squares (boxes) such that N is minimized.

## Features
### Solving the RPP automatically
![alt text](https://github.com/jonasgrebe/tu-opti-algo-project/docs/automatic_solve.gif "Automatic RPP solving")
![alt text](https://github.com/jonasgrebe/tu-opti-algo-project/docs/automatic_solve_2.gif "Automatic RPP solving")
Automatic solving using one of two different and efficiently implemented algorithmic approaches (local search and greedy search) with multiple tuning possibilities

### Solving the RPP by hand
![alt text](https://github.com/jonasgrebe/tu-opti-algo-project/docs/manual_solve.gif "Manual RPP solving")

### Defining an instance of the RPP
By setting the hyperparameters such as number and sizes of rectangles, box size, etc.
![alt text](https://github.com/jonasgrebe/tu-opti-algo-project/docs/configuration.gif "Problem configuration")


## How to use
Just execute `run_gui.py`.

## Implementation
This GUI is implemented using PyGame (for the grid world) and PyGame Menu (for the UI).
