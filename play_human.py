import pygame
import sys
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE


def play_human():
    """Play the game manually using arrow keys."""
    game = SnakeGameAI()

    # Slow speed for human play
    HUMAN_SPEED = 8

    print("=" * 50)
    print("بازی شروع شد!")
    print("روی پنجره کلیک کنید و با کلیدهای جهت‌دار بازی کنید.")
    print("ESC برای خروج.")
    print("=" * 50)

    game_over_pause = 0  # Frames to pause after game over

    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

                # Change direction (prevent 180-degree turns)
                if event.key == pygame.K_LEFT and game.direction != Direction.RIGHT:
                    game.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and game.direction != Direction.LEFT:
                    game.direction = Direction.RIGHT
                elif event.key == pygame.K_UP and game.direction != Direction.DOWN:
                    game.direction = Direction.UP
                elif event.key == pygame.K_DOWN and game.direction != Direction.UP:
                    game.direction = Direction.DOWN

        # If we're in game-over pause, don't move the snake
        if game_over_pause > 0:
            game_over_pause -= 1
            game.clock.tick(HUMAN_SPEED)
            continue

        # Calculate new head position
        x, y = game.head.x, game.head.y
        if game.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif game.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif game.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif game.direction == Direction.UP:
            y -= BLOCK_SIZE

        new_head = Point(x, y)

        # Check collision BEFORE moving
        if game.is_collision(new_head):
            print(f"Game Over! Score: {game.score}")
            game.reset()
            # Pause for 30 frames (~4 seconds) so player can see the reset
            game_over_pause = 30
            continue

        # Move snake
        game.head = new_head
        game.snake.insert(0, game.head)

        # Check food
        if game.head == game.food:
            game.score += 1
            game._place_food()
        else:
            game.snake.pop()

        # Render
        game._update_ui()
        game.clock.tick(HUMAN_SPEED)


if __name__ == '__main__':
    play_human()