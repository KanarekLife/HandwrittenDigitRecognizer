import pygame
from PIL import Image
from torchvision import datasets
from KNearestNeighborsRecognizer import KNearestNeighborsRecognizer
from RandomForestTreeRecognizer import RandomForestTreeRecognizer
from SVMRecognizer import SVMRecognizer
from utils import normalize_image, center_image, convert_to_image
import time
import numpy as np

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

# get image for each class in dataset
# for i in range(10):
#     image = training_dataset.data[training_dataset.targets == i][0]
#     image = convert_to_image(image.numpy())
#     image.show()


def display_helper_digit(digit: int):
    clear_canvas()
    digit_images = training_dataset.data[training_dataset.targets == digit]
    random_index = np.random.randint(len(digit_images))
    image = digit_images[random_index]
    image = convert_to_image(image.numpy())
    canvas.blit(pygame.image.fromstring(image.convert("RGB").tobytes(), (800, 600), "RGB"), (0, 0))

# Function to clear the canvas
def clear_canvas():
    canvas.fill(WHITE)

# Function to recognize the letter
def recognize_letter():
    image = Image.frombytes("RGB", (canvas_width, canvas_height), pygame.image.tostring(canvas, "RGB"))

    # center_image(normalize_image(image)).show()

    predictions = np.empty(len(recognizers), dtype=int)
    for i, recognizer in enumerate(recognizers):
        recognizer_predictions = np.array([recognizer.recognize(image) for _ in range(10)])
        most_common_prediction = np.argmax(np.bincount(recognizer_predictions))
        predictions[i] = most_common_prediction

    print("=====================================")
    print(f"RandomForestTreeRecognizer: {predictions[0]}")
    print(f"KNearestNeighborsRecognizer: {predictions[1]}")
    print(f"SVMRecognizer: {predictions[2]}")
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
            elif event.key in [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]:
                display_helper_digit(int(chr(event.key)))
            elif event.key == pygame.K_p:
                image = Image.frombytes("RGB", (canvas_width, canvas_height), pygame.image.tostring(canvas, "RGB"))
                center_image(normalize_image(image)).show()
            elif event.key == pygame.K_s:
                image = Image.frombytes("RGB", (canvas_width, canvas_height), pygame.image.tostring(canvas, "RGB"))
                image.point(lambda x: 255 if x > 30 else 0).save("temp.png")
            

    if drawing:
        current_pos = pygame.mouse.get_pos()
        pygame.draw.line(canvas, BLACK, last_pos, current_pos, 15)
        pygame.draw.circle(canvas, BLACK, current_pos, 7)
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