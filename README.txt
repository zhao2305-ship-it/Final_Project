Final Project – Water Pinball

This project is a small interactive 3D scene with an animated water surface and a pinball-style mini game.

The water surface is animated using several wave components. You can see light reflections on the surface, and the image placed under the water is distorted by refraction and caustics, so the water motion is clearly visible.

A ball can be launched into the scene. The ball moves at a constant speed and bounces off the boundary walls and the cube in the center. Whenever the ball hits a wall or the cube, it creates a strong ripple on the water surface. The goal is to get the ball into the exit area within 15 seconds. If the time runs out, the game is lost.

You can rotate the camera by dragging the mouse to see the scene from different angles. The mouse wheel is used to zoom in and out. Before launching the ball, the left and right arrow keys are used to adjust the launch angle. Press Enter to launch the ball. Press R to reset the game. Keys 1, 2, and 3 switch between different wave strengths. Press ESC to quit.

Files needed:
- simple_water.cpp (main source file)
- Hachimi.bmp (ground texture under the water)

Both files need to be in the same folder.

Build (Windows, MSYS2 UCRT64):
cd /d/Final_project  
g++ simple_water.cpp -o water.exe -lmingw32 -lSDL2main -lSDL2 -lglew32 -lopengl32

Run:
./water.exe

Note:
If the ground texture does not show up, make sure the filename is exactly "Hachimi.bmp" and it is in the same directory as the executable.
