import random
import math

# Sample coordinates of cities (x, y)
CITIES = {
    0: (0, 0),
    1: (1, 3),
    2: (4, 3),
    3: (6, 1),
    4: (3, 0)
}

# Calculate distance between two cities
def calculate_distance(city1, city2):
    x1, y1 = CITIES[city1]
    x2, y2 = CITIES[city2]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Fitness function: Total distance of the tour (minimize)
def fitness(individual):
    total_distance = 0
    for i in range(len(individual) - 1):
        total_distance += calculate_distance(individual[i], individual[i + 1])
    # Return to the start point
    total_distance += calculate_distance(individual[-1], individual[0])
    return 1 / total_distance  # Inverse because we want to minimize the distance

# Create a random individual (random order of cities)
def create_individual():
    cities = list(CITIES.keys())
    random.shuffle(cities)
    return cities

# Create initial population
def create_population(pop_size):
    return [create_individual() for _ in range(pop_size)]

# Selection: Choose the top two best individuals based on fitness
def selection(population):
    population.sort(key=lambda x: fitness(x), reverse=True)
    return population[:2]

# Crossover (Ordered Crossover)
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted([random.randint(0, size - 1) for _ in range(2)])
    
    child1 = [None] * size
    child2 = [None] * size

    # Copy a slice from the first parent
    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]

    # Fill the remaining positions with the other parent's genes
    fill_child(child1, parent2)
    fill_child(child2, parent1)

    return child1, child2

def fill_child(child, parent):
    for gene in parent:
        if gene not in child:
            for i in range(len(child)):
                if child[i] is None:
                    child[i] = gene
                    break

# Mutation: Swap two cities in the tour
def mutate(individual):
    idx1, idx2 = random.sample(range(len(individual)), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

# Genetic Algorithm to solve TSP
def genetic_algorithm(pop_size, generations):
    population = create_population(pop_size)

    for generation in range(generations):
        # Selection: Select the best individuals
        parents = selection(population)

        # Crossover: Create children
        children = []
        for _ in range(pop_size // 2):
            child1, child2 = crossover(parents[0], parents[1])
            children.append(child1)
            children.append(child2)

        # Mutation: Randomly mutate some individuals
        for child in children:
            if random.random() < 0.1:  # 10% chance of mutation
                mutate(child)

        # New population
        population = children

        # Show the best distance in the current generation
        best_individual = max(population, key=lambda x: fitness(x))
        best_distance = 1 / fitness(best_individual)
        print(f"Generation {generation}: Best distance: {best_distance:.2f}")

    # Return the best solution
    best_individual = max(population, key=lambda x: fitness(x))
    return best_individual, 1 / fitness(best_individual)

# Run the Genetic Algorithm
population_size = 10
generations = 50

best_route, best_distance = genetic_algorithm(population_size, generations)
print("Best route:", best_route)
print("Best distance:", best_distance)