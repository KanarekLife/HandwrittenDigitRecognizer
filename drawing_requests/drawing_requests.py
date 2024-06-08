import pygame
from PIL import Image
import time
import random
import os
import zipfile
import shutil

# Initialize Pygame
pygame.init()
pygame.font.init()

# set up the font
myfont = pygame.font.SysFont('Arial', 24)

# Set up colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Set up the screen
screen_width = 800
screen_height = 800
screen = pygame.display.set_mode((screen_width, screen_height))
screen.fill(WHITE)
pygame.display.set_caption("Paint Application")

# Set up the canvas
canvas = pygame.Surface((screen.get_width(), screen.get_height()))
canvas.fill(WHITE)

# Set up the drawing variables
drawing = False
last_pos = None
last_draw_time = time.time()

# setup requests variables
try:
    os.mkdir("temp_draws")
except FileExistsError:
    pass

DIGIT_REPEATS = 3

draw_requests = list(range(10)) * DIGIT_REPEATS

# shuffle without two same digits next to each other
while True:
    random.shuffle(draw_requests)
    if all(a != b for a, b in zip(draw_requests, draw_requests[1:])):
        break

drawn_counts = [0] * 10
drawn_count = 0
draw_total = len(draw_requests)

# Function to clear the canvas
def clear_canvas():
    canvas.fill(WHITE)

def image_to_surface(image: Image) -> pygame.Surface:
    return pygame.image.fromstring(image.convert("RGB").tobytes(), image.size, "RGB")

def save_image(path: str):
    image = Image.frombytes("RGB", (screen_width, screen_height), pygame.image.tostring(canvas, "RGB"))
    image.point(lambda x: 255 if x > 30 else 0).save(path)

# Main game loop
running = True
anything_changed_since_last_recognition = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                drawing = True
                last_pos = event.pos
                last_draw_time = time.time()
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                drawing = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:  # Clear canvas when 'c' is pressed
                clear_canvas()
            elif event.key == pygame.K_RETURN:
                if not draw_requests:
                    running = False
                    continue
                requested = draw_requests.pop(0)
                filename = f"temp_draws/drawn_{requested}_{drawn_counts[requested]}.png"
                drawn_counts[requested] += 1
                drawn_count += 1
                save_image(filename)
                clear_canvas()
            elif event.key == pygame.K_q:
                running = False

    if drawing:
        current_pos = pygame.mouse.get_pos()
        pygame.draw.line(canvas, BLACK, last_pos, current_pos, 15)
        pygame.draw.circle(canvas, BLACK, current_pos, 7)
        anything_changed_since_last_recognition = True
        last_pos = current_pos
        last_draw_time = time.time()

    screen.blit(canvas, (0, 0))

    if not draw_requests:
        text = myfont.render("All done! Press enter to exit", False, (0, 0, 0))
        screen.blit(text, (10, 10))
    else:
        text = myfont.render(f"Draw {draw_requests[0]} and press enter ({drawn_count}/{draw_total})", False, (0, 0, 0))
        screen.blit(text, (10, 10))

    pygame.display.flip()

# zip files
with zipfile.ZipFile("drawn_digits.zip", "w") as file:
    for filename in os.listdir("temp_draws"):
        if filename.endswith(".png"):
            file.write(f"temp_draws/{filename}", filename)

# cleanup
shutil.rmtree("temp_draws")

# Quit Pygame
pygame.quit()