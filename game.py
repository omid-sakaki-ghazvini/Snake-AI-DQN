import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# Initialize pygame (only once)
pygame.init()
font = pygame.font.Font(None, 25)


class Direction(Enum):
    """Possible movement directions for the snake."""
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# RGB colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)

BLOCK_SIZE = 20
SPEED = 40  # Only used when render=True


class SnakeGameAI:
    """
    Snake game environment with Gym-like API for RL.

    Args:
        w (int): Window width in pixels.
        h (int): Window height in pixels.
        render (bool): If True, opens a window and renders the game.
                       If False, runs headless (much faster for training).
    """

    def __init__(self, w=640, h=480, render=True):
        self.w = w
        self.h = h
        self.render = render

        if self.render:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake AI')
        else:
            self.display = None

        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """Reset the game to its initial state and return the state vector."""
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        return self.get_state()

    def _place_food(self):
        """Place food at a random position not occupied by the snake."""
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        """
        Execute one game step.

        Args:
            action: One-hot vector [straight, right, left].

        Returns:
            reward (float), game_over (bool), score (int)
        """
        self.frame_iteration += 1

        # 1. Handle pygame events (only if window is open)
        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        # 2. Move the snake
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. Check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Handle food consumption or just move forward
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            reward = -0.1  # Small penalty per step encourages faster food finding

        # 5. Update UI and clock (only if rendering)
        if self.render:
            self._update_ui()
            self.clock.tick(SPEED)

        # 6. Return reward, game_over, score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        """Check if given point (default: head) collides with wall or body."""
        if pt is None:
            pt = self.head

        # Wall collision
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        # Self collision (skip the head itself)
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        """Render the game frame. Skipped entirely if render=False."""
        if not self.render:
            return

        self.display.fill(BLACK)

        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1,
                             pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2,
                             pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        # Draw food
        pygame.draw.rect(self.display, RED,
                         pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw score
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        """
        Update head position based on action.

        Action is a one-hot vector:
            [1, 0, 0] -> go straight
            [0, 1, 0] -> turn right
            [0, 0, 1] -> turn left
        """
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # No change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4  # Turn right
            new_dir = clock_wise[next_idx]
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4  # Turn left
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def get_state(self):
        """
        Return an 11-dimensional binary state vector:
        [danger_straight, danger_right, danger_left,
         dir_left, dir_right, dir_up, dir_down,
         food_left, food_right, food_up, food_down]
        """
        head = self.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger straight ahead
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            # Danger to the right
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),

            # Danger to the left
            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),

            # Current direction
            dir_l, dir_r, dir_u, dir_d,

            # Food location relative to head
            self.food.x < head.x,  # food is to the left
            self.food.x > head.x,  # food is to the right
            self.food.y < head.y,  # food is above
            self.food.y > head.y   # food is below
        ]

        return np.array(state, dtype=int)