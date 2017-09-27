# idea2birds
Simulation of the collective motion of a flock of birds for the course idea to result at university of copenhagen in winter semester 2017/2018

## Usage:
```
usage: [python3] birds.py [-h] [--angle ANGLE] [--max_velocity MAX_VELOCITY]
                [--min_velocity MIN_VELOCITY]
                [--max_acceleration MAX_ACCELERATION] [--width WIDTH]
                [--height HEIGHT] [--n N] [--random RANDOM]
                [--alignment_radius ALIGNMENT_RADIUS]
                [--cohesion_radius COHESION_RADIUS]
                [--separation_radius SEPARATION_RADIUS]
                [--alignment ALIGNMENT] [--cohesion COHESION]
                [--separation SEPARATION] [--export] [--frames FRAMES]
                [--vfile VFILE] [--fps FPS] [--speed SPEED] [--scale SCALE]
                [-s S]

optional arguments:
  -h, --help            show this help message and exit
  --angle ANGLE, -a ANGLE
                        Boid field of Vision, default=60.0
  --max_velocity MAX_VELOCITY
                        Maximum velocity for a boid, default=1.0
  --min_velocity MIN_VELOCITY
                        Minimum velocity for a boid, default=0.5
  --max_acceleration MAX_ACCELERATION
                        Maximum acceleration per effect
  --width WIDTH         Range for x-coordinate, default=640
  --height HEIGHT       Range for y-coordinate, default=480
  --n N, -n N           Number of boids, default=500
  --random RANDOM, -r RANDOM
                        Scale factor of normal distribution of random turning
                        angles, default=0.1
  --alignment_radius ALIGNMENT_RADIUS
                        Radius for boids considered in alignment, default=50
  --cohesion_radius COHESION_RADIUS
                        Radius for boids considered in cohesion, default=50
  --separation_radius SEPARATION_RADIUS
                        Radius for boids considered in separation, default=25
  --alignment ALIGNMENT
                        Weight of alignment in acceleration sum, default=1
  --cohesion COHESION   Weight of cohesion in acceleration sum, default=1
  --separation SEPARATION
                        Weight of separation in acceleration sum, default=1.5
  --export, -e          Export video file and exit
  --frames FRAMES       Number of frames for export to video file,
                        default=1000
  --vfile VFILE         Out-File for video export, default=birds.mp4
  --fps FPS             Set Video Framerate for export, default=40
  --speed SPEED         Modify speed: Scale max_velocity, min_velocity,
                        max_acceleration, alignment_radius, cohesion_radius,
                        separation_radius at once.
  --scale SCALE         Scale field size.
  -s S                  equals --speed S --scale S
```
