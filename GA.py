import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as patches
import copy
# Step 1: Define the shapes to be nested
shapes = []

# Add a rectangle
rect_width = 5
rect_height = 10
shapes.append({'type': 'rect', 'width': rect_width, 'height': rect_height})

# Add a circle
circle_diameter = 7
circle_radius = circle_diameter / 2
shapes.append({'type': 'circle', 'width': circle_diameter,
              'height': circle_diameter, 'radius': circle_radius})

# Add a triangle
tri_base = 8
tri_height = 4
shapes.append({'type': 'triangle', 'base': tri_base,
              'height': tri_height, 'width': tri_base})

# Step 2: Define the genetic algorithm parameters
POPULATION_SIZE = 40
NUM_GENERATIONS = 30
MUTATION_RATE = 0.4
CROSSOVER_RATE = 0.8  # or any value between 0 and 1 that makes sense for your simulation
SHEET_WIDTH = 20
SHEET_HEIGHT = 20
DELTA_X = 5
DELTA_Y = 5
ELITE_SIZE = POPULATION_SIZE * 0.1 // 2 * 2

# def load_shapes():


# Step 3: Define the fitness function

def fitness(chromosome, shapes, flag = False):
    # Calculate the wasted space in the nesting arrangement
    fitness_value = 0
    temp_mask = np.zeros((SHEET_HEIGHT, SHEET_WIDTH), dtype=np.uint8)  # test code

    for shape in chromosome:
        x, y = shape['position']
        w, h = shape['width'], shape['height']
        if shape['type'] == 'rect':
            mask = np.zeros((SHEET_HEIGHT, SHEET_WIDTH))
            mask[max(round(y), 0):max(round(y + h), 0) + 1,
            max(round(x), 0):max(round(x + w), 0) + 1] = 1
            temp_mask[mask == 1] = 1  # test code
        elif shape['type'] == 'circle':
            # rr, cc = plt.circle(y+h//2, x+w//2, shape['diameter']//2, image.shape)
            Y, X = np.ogrid[:SHEET_HEIGHT, :SHEET_WIDTH]
            dist_from_center = np.sqrt((X - (x + w / 2)) ** 2 + (Y - (y + h / 2)) ** 2)
            # make a mask that is less then Radius +0.5 (interger)
            mask = dist_from_center <= ((w / 2) + 0.5)
            temp_mask[mask == 1] = 1  # test code
        elif shape['type'] == 'triangle':
            x_points = [x, x + w // 2, x + w, x]
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
    if flag == True:
        print("chromosome in fitness function", chromosome)
    return fitness_value

# Step 4: Define the genetic algorithm functions

def initialize_population(pop_size, shapes):
    # Generate the initial population of chromosomes
    population = []
    for i in range(pop_size):
        chromosome = []
        for shape in shapes:
            # Randomly generate a position for the shape
            if 'width' in shape:
                x = random.randint(0, SHEET_WIDTH - shape['width'])
                y = random.randint(0, SHEET_HEIGHT - shape['height'])
                # Add the shape and its position to the chromosome
                chromosome.append(
                    {'type': shape['type'], 'width': shape['width'], 'height': shape['height'], 'position': (x, y)})
            else:
                raise KeyError("Missing 'width' key in shape dictionary")
        print(chromosome)
        population.append(chromosome)
    return population

def mutate(chromosome):
    # Perform mutation operation on the chromosome
    mutation_point1 = np.random.randint(len(chromosome))
    mutation_point2 = np.random.randint(len(chromosome))
    # chromosome[mutation_point1], chromosome[mutation_point2] = chromosome[mutation_point2], chromosome[mutation_point1]
    for i in range(len(chromosome)):
        rand = random.random()
        if rand < MUTATION_RATE:
            delta_x = random.uniform(0, DELTA_X)
            delta_y = random.uniform(0, DELTA_Y)
            current_x = chromosome[i]['position'][0]
            current_y = chromosome[i]['position'][1]
            width = chromosome[i]['width']
            height = chromosome[i]['height']
            # chromosome[i]['position'] = (delta_x + chromosome[i]['position'][0], delta_y + chromosome[i]['position'][1])
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
    offspring1 = []
    offspring2 = []

    for i in range(len(parent1)):
        if random.random() < CROSSOVER_RATE:
            offspring1.append(parent1[i])
            offspring2.append(parent2[i])
        else:
            offspring1.append(parent2[i])
            offspring2.append(parent1[i])

    return [offspring1, offspring2]

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
    fitness_scores = []
    for individual in population:
        score = fitness(individual, shapes)
        # print(score)
        fitness_scores.append(score)
    return fitness_scores

def roulette_wheel_selection(fitness_scores):
    """Perform roulette wheel selection to select parent"""
    total_fitness = sum(fitness_scores)
    selection_probabilities = [fitness_score /
                               total_fitness for fitness_score in fitness_scores]
    return np.random.choice(range(len(fitness_scores)), p=selection_probabilities)

def evolve(population, fitness_func, mutation_rate):
    """
    Evolves the population by selecting parents, recombining their genomes, and mutating the offspring.
    Returns the new population.
    """
    best_chromosome = copy.deepcopy(max(population, key=lambda x: fitness(x, shapes)))
    for generation in range(NUM_GENERATIONS):
        print(f"Generation {generation+1}/{NUM_GENERATIONS}")

        # Select parents for reproduction
        new_population = []
        for i in range(POPULATION_SIZE // 2):
            parent1, parent2 = select_parents(population)
            new_population.extend(crossover(parent1, parent2))

        # Mutate offspring
        for i in range(len(new_population)):
            new_population[i] = mutate(new_population[i])

        # Evaluate fitness of offspring
        fitness_scores = evaluate_fitness(new_population)
        # print("fitness", fitness_scores)
        # Select next generationtemp_best_chromosome
        population = []
        for i in range(POPULATION_SIZE):
            idx = roulette_wheel_selection(fitness_scores)
            population.append(new_population[idx])
            del fitness_scores[idx]

        temp_best_chromosome = copy.deepcopy(max(population, key=lambda x: fitness(x, shapes)))
        if fitness(best_chromosome, shapes) < fitness(temp_best_chromosome, shapes):
            best_chromosome = copy.deepcopy(temp_best_chromosome)
        print("best chromosome: ", best_chromosome)
        print("best_after", fitness(best_chromosome, shapes, True))
        print("\n")

        # Print fitness of best chromosome in this generation
        fitness_scores = evaluate_fitness(population)
        print("fitness", fitness_scores)
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
        if shape['type'] == 'rect':
            image[max(round(y), 0):max(round(y + h), 0) + 1, max(round(x), 0):max(round(x + w), 0) + 1] = color
            mask = np.zeros((image.shape[0], image.shape[1]))
            mask[max(round(y), 0):max(round(y + h), 0) + 1,
            max(round(x), 0):max(round(x + w), 0) + 1] = 1
            image[mask == 1] = color
            temp_mask[mask == 1] = 1  # test code
            x_points = [x, x + w, x + w, x, x]  # test code
            y_points = [y, y, y + h, y + h, y]  # test code
            plt.fill(x_points, y_points)  # test code
        elif shape['type'] == 'circle':
            # rr, cc = plt.circle(y+h//2, x+w//2, shape['diameter']//2, image.shape)
            Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
            dist_from_center = np.sqrt((X - (x + w / 2)) ** 2 + (Y - (y + h / 2)) ** 2)
            # make a mask that is less then Radius +0.5 (interger)
            mask = dist_from_center <= ((w / 2) + 0.5)
            image[mask == 1] = color
            temp_mask[mask == 1] = 1  # test code
            temp = Circle((x + w / 2, y + h / 2), w / 2)  # test code
            plt.gca().add_patch(temp)  # test code
        elif shape['type'] == 'triangle':
            x_points = [x, x + w // 2, x + w, x]
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
population = initialize_population(POPULATION_SIZE, shapes)
init_best_chromosome = max(population, key=lambda x: fitness(x, shapes))

# evaluate_fitness(population)
final_population, final_best_chromosome = evolve(
    population, lambda x: fitness(x, shapes), MUTATION_RATE)
# print(final_population)
# Step 6: Display the best nesting as an image
# best_chromosome = max(final_population, key=lambda x: fitness(x, shapes))
display(init_best_chromosome)
display(final_best_chromosome)
# np.savetxt("output_image.txt", temp_mask, fmt='%d')  # test code
# pass


def main():
    # Step 1: Load shape dimensions
    shape_dimensions = load_shape_dimensions()

    # Step 2: Set genetic algorithm parameters
    POPULATION_SIZE = 100
    NUM_GENERATIONS = 50
    ELITE_SIZE = 10
    MUTATION_RATE = 0.1
    TOURNAMENT_SIZE = 5

    # Step 3: Initialize population
    population = initialize_population(POPULATION_SIZE, shape_dimensions)

    for i in range(NUM_GENERATIONS):
        # Step 4: Evaluate fitness
        fitness_scores = evaluate_fitness(population, shape_dimensions)

        # Step 5: Select parents
        parents = select_parents(
            population, fitness_scores, ELITE_SIZE, TOURNAMENT_SIZE)

        # Step 6: Generate offspring through crossover and mutation
        offspring = []
        for j in range(POPULATION_SIZE - ELITE_SIZE):
            parent1, parent2 = parents[random.randint(
                0, len(parents) - 1)], parents[random.randint(0, len(parents) - 1)]
            child = crossover(parent1, parent2)
            child = mutate(child, MUTATION_RATE, shape_dimensions)
            offspring.append(child)

        # Step 7: Merge parents and offspring to form new population
        population = parents + offspring

        # Step 8: Evaluate fitness of new population
        fitness_scores = evaluate_fitness(population, shape_dimensions)

        # Step 9: Select top performers for next generation
        population = select_top_performers(
            population, fitness_scores, POPULATION_SIZE)

    # Step 10: Get best performing chromosome
    best_chromosome = get_best_chromosome(population, shape_dimensions)

    # Step 11: Generate nested layout
    nested_layout, image = generate_nested_layout(
        best_chromosome, shape_dimensions)

    # Step 12: Display nested layout
    fig, ax = plt.subplots()
    circle = Circle((0.5, 0.5), 0.2)
    polygon = patches.Polygon(zip(x_points, y_points),
                              facecolor='none', edgecolor='red')
    ax.add_patch(polygon)
    ax.add_patch(circle)
    plt.show()
