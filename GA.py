import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import copy
import ast
import multiprocessing

# Step 2: Define the genetic algorithm parameters
POPULATION_SIZE = 50
NUM_GENERATIONS = 10
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.8  # or any value between 0 and 1 that makes sense for your simulation
SHEET_WIDTH = 100
SHEET_HEIGHT = 100
DELTA_X = 15
DELTA_Y = 15
ELITE_SIZE = 4

def load_shapes():
    shape_list = []
    with open("shapes.txt") as f:
        shape_list = ast.literal_eval(f.read())
    return shape_list

# Step 3: Define the fitness function
def fitness(chromosome, flag = False):
    # Calculate the wasted space in the nesting arrangement
    fitness_value = 0
    temp_mask = np.zeros((SHEET_HEIGHT, SHEET_WIDTH), dtype=np.uint8)  # test code
    for shape in chromosome:
        x, y = shape['position']
        w, h = shape['width'], shape['height']
        if shape['type'] == 'Rect':
            mask = np.zeros((SHEET_HEIGHT, SHEET_WIDTH))
            mask[max(round(y), 0):max(round(y + h), 0) + 1,
            max(round(x), 0):max(round(x + w), 0) + 1] = 1
            temp_mask[mask == 1] = 1  # test code
        elif shape['type'] == 'Circle':
            # rr, cc = plt.circle(y+h//2, x+w//2, shape['diameter']//2, image.shape)
            Y, X = np.ogrid[:SHEET_HEIGHT, :SHEET_WIDTH]
            dist_from_center = np.sqrt((X - (x + w / 2)) ** 2 + (Y - (y + h / 2)) ** 2)
            # make a mask that is less then Radius +0.5 (interger)
            mask = dist_from_center <= ((w / 2) + 0.5)
            temp_mask[mask == 1] = 1  # test code
        elif shape['type'] == 'Triangle':
            x_points = [x, x + w / 2, x + w, x]
            y_points = [y + h, y, y + h, y + h]
            Y, X = np.ogrid[:SHEET_HEIGHT, :SHEET_WIDTH]
            # x -> x+w//2| first edgex
            slope = (y_points[1] - y_points[0]) / (x_points[1] - x_points[0])
            diff = Y - (slope * X - slope * x_points[0] + y_points[0])
            mask_edge1 = diff >= -0.5

            # x -> x+w| second edge
            slope = (y_points[2] - y_points[0]) / (x_points[2] - x_points[0])
            diff = Y - (slope * X - slope * x_points[0] + y_points[0])
            mask_edge2 = diff <= 0.5

            # x+w//2 -> x+w| third edge
            slope = (y_points[2] - y_points[1]) / (x_points[2] - x_points[1])
            diff = Y - (slope * X - slope * x_points[1] + y_points[1])
            mask_edge3 = diff >= -0.5

            mask = mask_edge1 * mask_edge2 * mask_edge3

            temp_mask[mask != 0] = 1  # test code

    for row in range(SHEET_HEIGHT):
        for col in range(SHEET_WIDTH):
            if temp_mask[row][col] == 1:
                fitness_value = fitness_value + 1
    # if flag == True:
    #     print("chromosome in fitness function", chromosome)
    return fitness_value

# Step 4: Define the genetic algorithm functions
def initialize_population(pop_size, shapes):
    # Generate the initial population of chromosomes
    population = []
    for i in range(pop_size):
        chromosome = []
        k = 0
        for shape in shapes:
            for j in range(shape['Count ']):
                # Randomly generate a position for the shape
                if 'width' in shape:
                    x = random.randint(0, SHEET_WIDTH - shape['width'])
                    y = random.randint(0, SHEET_HEIGHT - shape['height'])
                    # Add the shape and its position to the chromosome
                    if shape['Shape'] == 'Circle':
                        chromosome.append(
                            {'type': shape['Shape'], 'width': shape['diameter'], 'height': shape['diameter'],
                             'position': (x, y)})
                    else:
                        chromosome.append(
                            {'type': shape['Shape'], 'width': shape['width'], 'height': shape['height'],
                             'position': (x, y)})
                else:
                    raise KeyError("Missing 'width' key in shape dictionary")
                if k > 10:
                    break
                k = k + 1
        population.append(chromosome)
    return population

def mutate(chromosome):
    # Perform mutation operation on the chromosome
    for i in range(len(chromosome)):
        rand = random.random()
        if rand < MUTATION_RATE:
            delta_x = random.uniform(0, DELTA_X)
            delta_y = random.uniform(0, DELTA_Y)
            current_x = chromosome[i]['position'][0]
            current_y = chromosome[i]['position'][1]
            width = chromosome[i]['width']
            height = chromosome[i]['height']
            rand = np.random.randint(0, 7)
            if rand == 0:
                chromosome[i]['position'] = (current_x + delta_x if current_x + delta_x < SHEET_WIDTH - width else current_x, current_y + delta_y if current_y + delta_y < SHEET_HEIGHT - height else current_y)
            elif rand == 1:
                chromosome[i]['position'] = (current_x, current_y + delta_y if current_y + delta_y < SHEET_HEIGHT - height else current_y)
            elif rand == 2:
                chromosome[i]['position'] = (current_x - delta_x if current_x - delta_x > 0 else current_x, current_y + delta_y if current_y + delta_y < SHEET_HEIGHT - height else current_y)
            elif rand == 3:
                chromosome[i]['position'] = (current_x - delta_x if current_x - delta_x > 0 else current_x, current_y)
            elif rand == 4:
                chromosome[i]['position'] = (current_x - delta_x if current_x - delta_x > 0 else current_x, current_y - delta_y if current_x - delta_y > 0 else current_y)
            elif rand == 5:
                chromosome[i]['position'] = (current_x, current_y - delta_y if current_y - delta_y > 0 else current_y)
            elif rand == 6:
                chromosome[i]['position'] = (current_x + delta_x if current_x + delta_x < SHEET_WIDTH - width else current_x, current_y - delta_y if current_y - delta_y > 0 else current_y)
            elif rand == 7:
                chromosome[i]['position'] = (current_x + delta_x if current_x + delta_x < SHEET_WIDTH - width else current_x, current_y)
            # chromosome[i]['position'] = (random.uniform(
            #     0, SHEET_WIDTH - chromosome[i]['width']), random.uniform(0, SHEET_HEIGHT - chromosome[i]['height']))
    return chromosome

def crossover(parent1, parent2):
    # Perform crossover operation on the two parents
    child1 = np.zeros_like(parent1)
    child2 = np.zeros_like(parent2)
    crossover_point = np.random.randint(len(parent1))
    child1[:crossover_point] = parent1[:crossover_point]
    child1[crossover_point:] = parent2[crossover_point:]
    child2[:crossover_point] = parent2[:crossover_point]
    child2[crossover_point:] = parent1[crossover_point:]
    offspring = []

    (parent_sup, parent_sub) = (parent1, parent2) if fitness(parent1) > fitness(parent2) else (parent2, parent1)
    for i in range(len(parent1)):
        if random.random() < CROSSOVER_RATE:
            offspring.append(parent_sup[i])
        else:
            offspring.append(parent_sub[i])
    # print(offspring)
    return offspring
    # return child1, child2

def select_parents(population):
    fitness_scores = evaluate_fitness(population)
    parent1_idx = roulette_wheel_selection(fitness_scores)
    parent2_idx = roulette_wheel_selection(fitness_scores)
    while parent2_idx == parent1_idx:
        parent2_idx = roulette_wheel_selection(fitness_scores)
    return population[parent1_idx], population[parent2_idx]

def evaluate_fitness(population):
    """
    Calculates the fitness score for each individual in the population.
    Returns a list of fitness scores, one for each individual.
    """
    # num_processes = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool()
    fitness_scores = []
    for individual in population:
        score = fitness(individual)
        violation = evaluate_constraints(individual)
        fitness_scores.append(score - violation)
    return fitness_scores

def roulette_wheel_selection(fitness_scores):
    """Perform roulette wheel selection to select parent"""
    total_fitness = sum(fitness_scores)
    selection_probabilities = [fitness_score /
                               total_fitness for fitness_score in fitness_scores]
    return np.random.choice(range(len(fitness_scores)), p=selection_probabilities)

def evaluate_constraints(chromosome):
    violations = 0

    for i in range(len(chromosome)):
        for j in range(i + 1, len(chromosome)):
            if is_overlap(chromosome[i], chromosome[j]):
                violations += 1

    return violations

def is_overlap(shape1, shape2):
    (x1, y1) = shape1['position']
    w1 = shape1['width']
    h1 = shape1['height']
    (x2, y2) = shape2['position']
    w2 = shape2['width']
    h2 = shape2['height']

    if x1 + w1 < x2 or x2 + w2 < x1:
        return False
    if y1 + h1 < y2 or y2 + h2 < y1:
        return False
    return True

def evolve(population):
    """
    Evolves the population by selecting parents, recombining their genomes, and mutating the offspring.
    Returns the new population.
    """
    best_chromosome = copy.deepcopy(max(population, key=lambda x: fitness(x)))
    for generation in range(NUM_GENERATIONS):
        print(f"Generation {generation+1}/{NUM_GENERATIONS}")

        # Select parents for reproduction
        population.sort(key=lambda x: fitness(x), reverse=True)

        new_population = []
        for i in range(POPULATION_SIZE // 2):
            parent1, parent2 = select_parents(population)
            offspring = crossover(parent1, parent2)
            new_population.append(copy.deepcopy(offspring))

        for i in range(POPULATION_SIZE // 2):
            new_population.append(copy.deepcopy(population[i]))

        # Mutate offspring
        for i in range(len(new_population)):
            new_population[i] = mutate(new_population[i])

        # Select next generationtemp_best_chromosome
        population = []

        # Evaluate fitness of offspring
        fitness_scores = evaluate_fitness(new_population)
        print(fitness_scores)
        for i in range(POPULATION_SIZE):
            idx = roulette_wheel_selection(fitness_scores)
            population.append(copy.deepcopy(new_population[idx]))
            del fitness_scores[idx]

        temp_best_chromosome = copy.deepcopy(max(population, key=lambda x: fitness(x)))
        if fitness(best_chromosome) < fitness(temp_best_chromosome):
            best_chromosome = copy.deepcopy(temp_best_chromosome)
        print("temp best chromosome: ", fitness(temp_best_chromosome))
        print("best chromosome:", fitness(best_chromosome, True))
        print("\n")

        # Print fitness of best chromosome in this generation
        # fitness_scores = evaluate_fitness(population)
        # print("fitness", fitness_scores)
        # print(f"Best fitness: {best_fitness}\n")

    # Return the final population
    return population, best_chromosome

def display(chromosome):
    image = np.zeros((SHEET_HEIGHT, SHEET_WIDTH, 3), dtype=np.uint8)
    # print("Height: ", image.shape[0], ", Width: ", image.shape[1])
    temp_mask = np.zeros(
        (image.shape[0], image.shape[1]), dtype=np.uint8)  # test code
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    plt.axes().set_aspect('equal')  # test code
    for i, shape in enumerate(chromosome):
        x, y = shape['position']
        w, h = shape['width'], shape['height']
        color = colors[i % len(colors)]
        if shape['type'] == 'Rectangle':
            image[max(round(y), 0):max(round(y + h), 0) + 1, max(round(x), 0):max(round(x + w), 0) + 1] = color
            mask = np.zeros((image.shape[0], image.shape[1]))
            mask[max(round(y), 0):max(round(y + h), 0) + 1,
            max(round(x), 0):max(round(x + w), 0) + 1] = 1
            image[mask == 1] = color
            temp_mask[mask == 1] = 1  # test code
            x_points = [x, x + w, x + w, x, x]  # test code
            y_points = [y, y, y + h, y + h, y]  # test code
            plt.fill(x_points, y_points)  # test code
        elif shape['type'] == 'Circle':
            # rr, cc = plt.circle(y+h//2, x+w//2, shape['diameter']//2, image.shape)
            Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
            dist_from_center = np.sqrt((X - (x + w / 2)) ** 2 + (Y - (y + h / 2)) ** 2)
            # make a mask that is less then Radius +0.5 (interger)
            mask = dist_from_center <= ((w / 2) + 0.5)
            image[mask == 1] = color
            temp_mask[mask == 1] = 1  # test code
            temp = Circle((x + w / 2, y + h / 2), w / 2)  # test code
            plt.gca().add_patch(temp)  # test code
        elif shape['type'] == 'Triangle':
            x_points = [x, x + w / 2, x + w, x]
            y_points = [y + h, y, y + h, y + h]
            # plt.fill(y_points, x_points, fill=True,
            #         edgecolor='black', facecolor='green')

            # y= slope*x - slope*x0+y0
            Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
            # x -> x+w//2| first edge
            slope = (y_points[1] - y_points[0]) / (x_points[1] - x_points[0])
            diff = Y - (slope * X - slope * x_points[0] + y_points[0])
            mask_edge1 = diff >= -0.5

            # x -> x+w| second edge
            slope = (y_points[2] - y_points[0]) / (x_points[2] - x_points[0])
            diff = Y - (slope * X - slope * x_points[0] + y_points[0])
            mask_edge2 = diff <= 0.5

            # x+w//2 -> x+w| third edge
            slope = (y_points[2] - y_points[1]) / (x_points[2] - x_points[1])
            diff = Y - (slope * X - slope * x_points[1] + y_points[1])
            mask_edge3 = diff >= -0.5

            mask = mask_edge1 * mask_edge2 * mask_edge3

            temp_mask[mask != 0] = 1  # test code
            # print(temp_mask)
            image[mask != 0] = color
            plt.fill(x_points, y_points)  # test code

    plt.xlim(0, SHEET_WIDTH)  # test code
    plt.ylim(0, SHEET_HEIGHT)  # test code
    plt.grid()  # test code
    plt.gca().invert_yaxis()  # test code
    plt.show()  # test code

# Step 5: Run the genetic algorithm
shapes = load_shapes()
population = initialize_population(POPULATION_SIZE, shapes)
# print(population)
init_best_chromosome = max(population, key=lambda x: fitness(x))
print(evaluate_fitness(population))
final_population, final_best_chromosome = evolve(population)
print(fitness(final_best_chromosome))
display(init_best_chromosome)
display(final_best_chromosome)
# np.savetxt("output_image.txt", temp_mask, fmt='%d')  # test code
# pass

