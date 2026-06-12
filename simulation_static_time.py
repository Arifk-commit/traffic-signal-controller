import random
import math
import time
import threading
import pygame
import sys
import os
import json
import numpy as np

# Load YOLO model
try:
    from ultralytics import YOLO
    model = YOLO('best.pt')
except:
    print("Warning: YOLO model not found")

# Default signal times
defaultRed = 150
defaultYellow = 5
defaultGreen = 20
defaultMinimum = 10
defaultMaximum = 60

signals = []
noOfSignals = 4
simTime = 300
timeElapsed = 0

currentGreen = 0
nextGreen = (currentGreen+1) % noOfSignals
currentYellow = 0

# Vehicle timing
carTime = 2
bikeTime = 1
vanTime = 2.25
busTime = 2.5
truckTime = 2.5

speeds = {'car': 2.25, 'bus': 1.8, 'truck': 1.8, 'van': 2, 'bike': 2.5}

x = {'right': [0, 0, 0], 'down': [755, 727, 697], 'left': [1400, 1400, 1400], 'up': [602, 627, 657]}
y = {'right': [348, 370, 398], 'down': [0, 0, 0], 'left': [498, 466, 436], 'up': [800, 800, 800]}

vehicles = {'right': {0: [], 1: [], 2: [], 'crossed': 0},
            'down': {0: [], 1: [], 2: [], 'crossed': 0},
            'left': {0: [], 1: [], 2: [], 'crossed': 0},
            'up': {0: [], 1: [], 2: [], 'crossed': 0}}

vehicleTypes = {0: 'car', 1: 'bus', 2: 'truck', 3: 'van', 4: 'bike'}
directionNumbers = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

signalCoods = [(530, 230), (810, 230), (810, 570), (530, 570)]
signalTimerCoods = [(530, 210), (810, 210), (810, 550), (530, 550)]
vehicleCountCoods = [(480, 210), (880, 210), (880, 550), (480, 550)]

stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
defaultStop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}
stops = {'right': [580, 580, 580], 'down': [320, 320, 320], 'left': [810, 810, 810], 'up': [545, 545, 545]}

mid = {'right': {'x': 705, 'y': 445}, 'down': {'x': 695, 'y': 450},
       'left': {'x': 695, 'y': 425}, 'up': {'x': 695, 'y': 400}}

rotationAngle = 3
gap = 15
gap2 = 15

pygame.init()
simulation = pygame.sprite.Group()

# Global variables for detected vehicles
detected_vehicles_from_file = {}
vehicles_created = False


class TrafficSignal:
    def __init__(self, red, yellow, green, minimum, maximum):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.minimum = minimum
        self.maximum = maximum
        self.signalText = "30"
        self.totalGreenTime = 0


class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction, will_turn, is_detected=False):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds.get(vehicleClass, 2)
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        self.willTurn = will_turn
        self.turned = 0
        self.rotateAngle = 0
        self.is_detected = is_detected
        
        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1
        
        # Map similar vehicle types for fallback
        vehicle_fallbacks = {
            'motorbike': 'bike',
            'bicycle': 'bike',
            'van': 'car'
        }
        
        # Load vehicle image from vehicles folder
        # Try different extensions for the vehicle class and its fallback
        image_loaded = False
        vehicle_to_load = vehicleClass
        
        # Try original vehicle class first, then fallback
        for attempt in [vehicleClass, vehicle_fallbacks.get(vehicleClass)]:
            if attempt is None:
                continue
            for ext in ['.png', '.jpg', '.jpeg']:
                path = f"images/vehicles/{attempt}{ext}"
                if os.path.exists(path):
                    try:
                        self.originalImage = pygame.image.load(path)
                        self.currentImage = pygame.image.load(path)
                        image_loaded = True
                        break
                    except:
                        continue
            if image_loaded:
                break
        
        # Fallback to colored rectangle if image not found
        if not image_loaded:
            self.originalImage = pygame.Surface((50, 30))
            self.originalImage.fill((100, 100, 100))
            self.currentImage = self.originalImage.copy()
        
        # Rotate image based on direction
        # Assuming original images face UP, rotate accordingly
        rotation_angles = {
            'right': -90,  # Rotate 90 degrees clockwise
            'down': 180,   # Rotate 180 degrees
            'left': 90,    # Rotate 90 degrees counter-clockwise
            'up': 0        # No rotation needed
        }
        
        if direction in rotation_angles and rotation_angles[direction] != 0:
            self.originalImage = pygame.transform.rotate(self.originalImage, rotation_angles[direction])
            self.currentImage = pygame.transform.rotate(self.currentImage, rotation_angles[direction])
        
        # Calculate stop position
        if direction == 'right':
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index-1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index-1].stop - vehicles[direction][lane][self.index-1].currentImage.get_rect().width - gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap
            x[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif direction == 'left':
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index-1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index-1].stop + vehicles[direction][lane][self.index-1].currentImage.get_rect().width + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap
            x[direction][lane] += temp
            stops[direction][lane] += temp
        elif direction == 'down':
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index-1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index-1].stop - vehicles[direction][lane][self.index-1].currentImage.get_rect().height - gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif direction == 'up':
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index-1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index-1].stop + vehicles[direction][lane][self.index-1].currentImage.get_rect().height + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] += temp
            stops[direction][lane] += temp
        
        simulation.add(self)

    def move(self):
        if self.direction == 'right':
            if self.crossed == 0 and self.x + self.currentImage.get_rect().width > stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if ((self.x + self.currentImage.get_rect().width <= self.stop or self.crossed == 1 or (currentGreen == 0 and currentYellow == 0)) and (self.index == 0 or self.x + self.currentImage.get_rect().width < (vehicles[self.direction][self.lane][self.index-1].x - gap2))):
                self.x += self.speed
        elif self.direction == 'down':
            if self.crossed == 0 and self.y + self.currentImage.get_rect().height > stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if ((self.y + self.currentImage.get_rect().height <= self.stop or self.crossed == 1 or (currentGreen == 1 and currentYellow == 0)) and (self.index == 0 or self.y + self.currentImage.get_rect().height < (vehicles[self.direction][self.lane][self.index-1].y - gap2))):
                self.y += self.speed
        elif self.direction == 'left':
            if self.crossed == 0 and self.x < stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if ((self.x >= self.stop or self.crossed == 1 or (currentGreen == 2 and currentYellow == 0)) and (self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2))):
                self.x -= self.speed
        elif self.direction == 'up':
            if self.crossed == 0 and self.y < stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if ((self.y >= self.stop or self.crossed == 1 or (currentGreen == 3 and currentYellow == 0)) and (self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height + gap2))):
                self.y -= self.speed


def load_detected_vehicles():
    """Load vehicles from detected_vehicles.json"""
    global detected_vehicles_from_file, vehicles_created
    
    if os.path.exists("detected_vehicles.json"):
        try:
            with open("detected_vehicles.json", "r") as f:
                detected_vehicles_from_file = json.load(f)
            print("✓ Loaded detected vehicles from app.py")
            return True
        except Exception as e:
            print(f"Error loading detections: {e}")
    return False


def create_vehicles_from_detections():
    """Create vehicles in simulation based on detections"""
    global vehicles_created
    
    if not detected_vehicles_from_file or vehicles_created:
        return
    
    for direction, detections in detected_vehicles_from_file.items():
        for idx, detection in enumerate(detections):
            vehicle_type = detection.get('class', 'car')
            lane = idx % 2 + 1  # Distribute across lanes
            direction_num = list(directionNumbers.keys())[list(directionNumbers.values()).index(direction)]
            
            vehicle = Vehicle(lane, vehicle_type, direction_num, direction, will_turn=0, is_detected=True)
            print(f"Created: {vehicle_type} in {direction} lane {lane}")
    
    vehicles_created = True


def initialize():
    ts1 = TrafficSignal(0, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts1)
    ts2 = TrafficSignal(ts1.red + ts1.yellow + ts1.green, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts4)
    repeat()


def repeat():
    global currentGreen, currentYellow, nextGreen
    while signals[currentGreen].green > 0:
        printStatus()
        updateValues()
        time.sleep(1)
    currentYellow = 1
    
    for i in range(0, 3):
        stops[directionNumbers[currentGreen]][i] = defaultStop[directionNumbers[currentGreen]]
    
    while signals[currentGreen].yellow > 0:
        printStatus()
        updateValues()
        time.sleep(1)
    currentYellow = 0
    
    signals[currentGreen].green = defaultGreen
    signals[currentGreen].yellow = defaultYellow
    signals[currentGreen].red = defaultRed
    
    currentGreen = nextGreen
    nextGreen = (currentGreen + 1) % noOfSignals
    signals[nextGreen].red = signals[currentGreen].yellow + signals[currentGreen].green
    repeat()


def printStatus():
    for i in range(0, noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                print(f" GREEN TS{i+1}-> r:{signals[i].red} y:{signals[i].yellow} g:{signals[i].green}")
            else:
                print(f"YELLOW TS{i+1}-> r:{signals[i].red} y:{signals[i].yellow} g:{signals[i].green}")
        else:
            print(f"   RED TS{i+1}-> r:{signals[i].red} y:{signals[i].yellow} g:{signals[i].green}")
    print()


def updateValues():
    for i in range(0, noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                signals[i].green -= 1
                signals[i].totalGreenTime += 1
            else:
                signals[i].yellow -= 1
        else:
            signals[i].red -= 1


def simulationTime():
    global timeElapsed, simTime
    while True:
        timeElapsed += 1
        time.sleep(1)
        if timeElapsed == simTime:
            totalVehicles = 0
            print('\n--- SIMULATION ENDED ---')
            print('Lane-wise Vehicle Counts')
            for i in range(noOfSignals):
                print(f'Lane {i+1} ({directionNumbers[i]}): {vehicles[directionNumbers[i]]["crossed"]}')
                totalVehicles += vehicles[directionNumbers[i]]['crossed']
            print(f'Total vehicles passed: {totalVehicles}')
            print(f'Total time passed: {timeElapsed}')
            print(f'Vehicles per unit time: {(float(totalVehicles)/float(timeElapsed)):.2f}')
            os._exit(1)


# Main Simulation Loop
if __name__ == "__main__":
    print("Starting Traffic Simulation...")
    print("Waiting for vehicle detections from app.py...")
    
    # Load detections
    load_detected_vehicles()
    
    thread4 = threading.Thread(name="simulationTime", target=simulationTime, args=())
    thread4.daemon = True
    thread4.start()
    
    thread2 = threading.Thread(name="initialization", target=initialize, args=())
    thread2.daemon = True
    thread2.start()
    
    black = (0, 0, 0)
    white = (255, 255, 255)
    
    screenWidth = 1400
    screenHeight = 800
    screenSize = (screenWidth, screenHeight)
    
    background = pygame.image.load('images/mod_int.png')
    
    screen = pygame.display.set_mode(screenSize)
    pygame.display.set_caption("YOLO Traffic Simulation with Real Detections")
    
    redSignal = pygame.image.load('images/signals/red.png')
    yellowSignal = pygame.image.load('images/signals/yellow.png')
    greenSignal = pygame.image.load('images/signals/green.png')
    font = pygame.font.Font(None, 30)
    
    clock = pygame.time.Clock()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        
        # Create vehicles from detections on first frame
        if not vehicles_created:
            create_vehicles_from_detections()
        
        screen.blit(background, (0, 0))
        
        for i in range(0, noOfSignals):
            if i == currentGreen:
                if currentYellow == 1:
                    if signals[i].yellow == 0:
                        signals[i].signalText = "STOP"
                    else:
                        signals[i].signalText = signals[i].yellow
                    screen.blit(yellowSignal, signalCoods[i])
                else:
                    if signals[i].green == 0:
                        signals[i].signalText = "SLOW"
                    else:
                        signals[i].signalText = signals[i].green
                    screen.blit(greenSignal, signalCoods[i])
            else:
                if signals[i].red <= 10:
                    if signals[i].red == 0:
                        signals[i].signalText = "GO"
                    else:
                        signals[i].signalText = signals[i].red
                else:
                    signals[i].signalText = "---"
                screen.blit(redSignal, signalCoods[i])
        
        signalTexts = ["", "", "", ""]
        for i in range(0, noOfSignals):
            signalTexts[i] = font.render(str(signals[i].signalText), True, white, black)
            screen.blit(signalTexts[i], signalTimerCoods[i])
            displayText = vehicles[directionNumbers[i]]['crossed']
            vehicleCountText = font.render(str(displayText), True, black, white)
            screen.blit(vehicleCountText, vehicleCountCoods[i])
        
        timeElapsedText = font.render(("Time Elapsed: " + str(timeElapsed)), True, black, white)
        screen.blit(timeElapsedText, (1100, 50))
        
        for vehicle in simulation:
            screen.blit(vehicle.currentImage, [vehicle.x, vehicle.y])
            vehicle.move()
        
        pygame.display.update()
        clock.tick(30)