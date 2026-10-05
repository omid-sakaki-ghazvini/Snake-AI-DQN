import pygame
import sys

pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Key Test v2 - Click here first!")

print("=" * 50)
print("پنجره باز شد. روی آن کلیک کنید.")
print("سپس کلیدهای جهت‌دار را فشار دهید.")
print("زبان کیبورد باید ENGLISH باشد!")
print("=" * 50)

# Print SDL version info
print(f"Pygame version: {pygame.version.ver}")
print(f"SDL version: {pygame.version.SDL}")

running = True
while running:
    # Get ALL events (not just filtered)
    events = pygame.event.get()
    for event in events:
        # Print every single event type
        print(f"Event type: {event.type} - {pygame.event.event_name(event.type)}")

        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            print(f"  KEYDOWN: key={event.key} name='{pygame.key.name(event.key)}' mod={event.mod} unicode='{event.unicode}'")
            if event.key == pygame.K_ESCAPE:
                running = False
        if event.type == pygame.KEYUP:
            print(f"  KEYUP: key={event.key} name='{pygame.key.name(event.key)}'")
        if event.type == pygame.MOUSEBUTTONDOWN:
            print(f"  Mouse click at {event.pos}")

    # Small delay to avoid flooding console
    pygame.time.wait(50)

pygame.quit()
sys.exit()