import pygame
from PIL import Image
from torchvision import datasets
from KNearestNeighborsRecognizer import KNearestNeighborsRecognizer
from RandomForestTreeRecognizer import RandomForestTreeRecognizer
from SVMRecognizer import SVMRecognizer
from utils import normalize_image
import time

# Initialize Pygame
pygame.init()

# Set up colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Set up the canvas
canvas_width = 800
canvas_height = 600
canvas = pygame.display.set_mode((canvas_width, canvas_height))
canvas.fill(WHITE)
pygame.display.set_caption("Paint Application")

# Set up the drawing variables
drawing = False
last_pos = None
last_draw_time = time.time()

# Set up the recognizer
training_dataset = datasets.MNIST('./data', train=True, download=True)
recognizers = [
    RandomForestTreeRecognizer(training_dataset),
    KNearestNeighborsRecognizer(training_dataset),
    SVMRecognizer(training_dataset)
]

# Function to clear the canvas
def clear_canvas():
    canvas.fill(WHITE)

# Function to recognize the letter
def recognize_letter():
    image = Image.frombytes("RGB", (canvas_width, canvas_height), pygame.image.tostring(canvas, "RGB"))
    predictions = [recognizer.recognize(image) for recognizer in recognizers]
    print("=====================================")
    print(f"Random Forest: {predictions[0]}")
    print(f"KNN: {predictions[1]}")
    print(f"SVM: {predictions[2]}")
    print("=====================================")

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
            elif event.key == pygame.K_SPACE:  # Save canvas when spacebar is pressed
                recognize_letter()
                normalize_image(Image.frombytes("RGB", (canvas_width, canvas_height), pygame.image.tostring(canvas, "RGB"))).show()

    if drawing:
        current_pos = pygame.mouse.get_pos()
        pygame.draw.line(canvas, BLACK, last_pos, current_pos, 15)
        anything_changed_since_last_recognition = True
        last_pos = current_pos
        last_draw_time = time.time()
    else:
        if time.time() - last_draw_time >= 0.5 and anything_changed_since_last_recognition:
            recognize_letter()
            anything_changed_since_last_recognition = False

    pygame.display.flip()

# Quit Pygame
pygame.quit()