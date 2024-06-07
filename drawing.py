import pygame
from PIL import Image
from torchvision import datasets
from recognizers.all_recognizers import (RandomForestTreeRecognizer, KNearestNeighborsRecognizer,
                                     NonLinearSVMRecognizer, LinearSVMRecognizer, NeuralNetworkRecognizer)
from recognizers.Recognizer import Recognizer
from utils import normalize_image, center_image, convert_to_image
import time
import numpy as np
import threading
import datetime
import concurrent.futures

# Set up the recognizer
training_dataset = datasets.MNIST('./data', train=True, download=True)
recognizers: dict[str, Recognizer] = {
    "Random Forest Tree": RandomForestTreeRecognizer(training_dataset),
    "KNearest Neighbors": KNearestNeighborsRecognizer(training_dataset),
    "NonLinear SVM": NonLinearSVMRecognizer(training_dataset),
    "Linear SVM": LinearSVMRecognizer(training_dataset),
    "Neural Network": NeuralNetworkRecognizer(training_dataset, 'cpu', epochs=14),
}

# Initialize Pygame
pygame.init()
pygame.font.init()

# set up the font
myfont = pygame.font.SysFont('Arial', 24)

# Set up colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Set up the screen
screen_width = 700
screen_height = 700
screen = pygame.display.set_mode((screen_width, screen_height))
screen.fill(WHITE)
pygame.display.set_caption("Handwritten Digit Recognition")

# Set up the canvas
canvas = pygame.Surface((screen.get_width(), screen.get_height()))
canvas.fill(WHITE)

# Set up the drawing variables
drawing = False
last_pos = None
last_draw_time = time.time()

# Set up recognition variables
recognizing = False
predictions = {name: -1 for name in recognizers.keys()}

# Prepare labels
processed_label = myfont.render("Models' input:", True, BLACK)
recognizing_label = myfont.render("Recognizing...", True, BLACK)


def display_helper_digit(digit: int):
    clear_canvas()
    digit_images = training_dataset.data[training_dataset.targets == digit]
    random_index = np.random.randint(len(digit_images))
    image = digit_images[random_index]
    image = convert_to_image(image.numpy())
    canvas.blit(pygame.image.fromstring(image.convert("RGB").tobytes(), (800, 600), "RGB"), (0, 0))

def clear_canvas():
    canvas.fill(WHITE)

def recognize_letter():
    global recognizing
    global predictions

    image = Image.frombytes("RGB", (screen_width, screen_height), pygame.image.tostring(canvas, "RGB"))

    for name in recognizers.keys():
        predictions[name] = -1

    def recognize_parallel(name: str, recognizer: Recognizer):
        recognizer_predictions = np.array([recognizer.recognize(image) for _ in range(10)])
        most_common_prediction = np.argmax(np.bincount(recognizer_predictions))
        return name, most_common_prediction

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(recognize_parallel, name, recognizer) for name, recognizer in recognizers.items()]
        for future in concurrent.futures.as_completed(futures):
            name, prediction = future.result()
            predictions[name] = prediction

    print("=====================================")
    for name, prediction in predictions.items():
        print(f"{name}: {prediction}")
    print("=====================================")

    recognizing = False

def get_processed_image() -> Image:
    image = Image.frombytes("RGB", (screen_width, screen_height), pygame.image.tostring(canvas, "RGB"))
    return center_image(normalize_image(image))

def image_to_surface(image: Image) -> pygame.Surface:
    return pygame.image.fromstring(image.convert("RGB").tobytes(), image.size, "RGB")

# Main game loop
running = True
anything_changed_since_last_recognition = False
color = BLACK
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                drawing = True
                color = BLACK
                last_pos = event.pos
                last_draw_time = time.time()
            elif event.button == 3:
                drawing = True
                color = WHITE
                last_pos = event.pos
                last_draw_time = time.time()
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                drawing = False
            elif event.button == 3:
                drawing = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:  # Clear canvas when 'c' is pressed
                clear_canvas()
            elif event.key in [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]:
                display_helper_digit(int(chr(event.key)))
            elif event.key == pygame.K_s:
                image = Image.frombytes("RGB", (screen_width, screen_height), pygame.image.tostring(canvas, "RGB"))
                filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
                image.point(lambda x: 255 if x > 30 else 0).save(filename)

            elif event.key == pygame.K_q:
                running = False

    if drawing:
        current_pos = pygame.mouse.get_pos()
        pygame.draw.line(canvas, color, last_pos, current_pos, 15)
        pygame.draw.circle(canvas, color, current_pos, 7)
        anything_changed_since_last_recognition = True
        last_pos = current_pos
        last_draw_time = time.time()
    elif time.time() - last_draw_time >= 0.5 and anything_changed_since_last_recognition and not recognizing:
        anything_changed_since_last_recognition = False
        recognizing = True
        threading.Thread(target=recognize_letter).start()

    screen.blit(canvas, (0, 0))

    # Draw the processed image
    processed_image = image_to_surface(get_processed_image())
    processed_image = pygame.transform.scale(processed_image, (56, 56))
    screen.blit(processed_label, (screen.get_width() - processed_image.get_width() - processed_label.get_width() - 10, processed_image.get_height()/2 - processed_label.get_height()/2))
    screen.blit(processed_image, (screen.get_width() - processed_image.get_width(), 0))

    # Draw recognition results
    for i, name in enumerate(recognizers.keys()):
        prediction = predictions[name]
        if prediction == -1:
            prediction = ''
        text = myfont.render(f"{name}: {prediction}", True, BLACK)

        screen.blit(text, (10, 10 + i * 30))

    # Draw recognition status
    if recognizing:
        screen.blit(recognizing_label, (10, screen.get_height() - recognizing_label.get_height() - 10))

    pygame.display.flip()

# Quit Pygame
pygame.quit()