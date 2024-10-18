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




maze 
import random
import numpy as np

# Define the maze: 0 for open path, 1 for wall
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

start = (0, 0)  # Start point
goal = (4, 4)   # Goal point

# Genetic Algorithm Parameters
population_size = 100
generations = 1000
mutation_rate = 0.1

# Define moves
moves = ['up', 'down', 'left', 'right']

# Function to initialize population
def initialize_population():
    return [random.choices(moves, k=15) for _ in range(population_size)]

# Function to check if a position is valid
def is_valid(pos):
    x, y = pos
    if 0 <= x < len(maze) and 0 <= y < len(maze[0]) and maze[x][y] == 0:
        return True
    return False

# Function to compute fitness based on distance from goal
def fitness(individual):
    pos = list(start)
    for move in individual:
        if move == 'up':
            pos[0] -= 1
        elif move == 'down':
            pos[0] += 1
        elif move == 'left':
            pos[1] -= 1
        elif move == 'right':
            pos[1] += 1
        if not is_valid(pos):
            break
    return -abs(pos[0] - goal[0]) - abs(pos[1] - goal[1])

# Function to select parents for crossover
def select_parents(population):
    return random.choices(population, weights=[fitness(ind) for ind in population], k=2)

# Crossover function
def crossover(parent1, parent2):
    split = random.randint(1, len(parent1)-1)
    child1 = parent1[:split] + parent2[split:]
    child2 = parent2[:split] + parent1[split:]
    return child1, child2

# Mutation function
def mutate(individual):
    if random.random() < mutation_rate:
        individual[random.randint(0, len(individual)-1)] = random.choice(moves)
    return individual

# Main Genetic Algorithm loop
def genetic_algorithm():
    population = initialize_population()
    
    for generation in range(generations):
        population = sorted(population, key=lambda ind: fitness(ind), reverse=True)
        
        if fitness(population[0]) == 0:
            print(f"Solution found in generation {generation}")
            return population[0]
        
        new_population = population[:population_size//2]
        
        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))
        
        population = new_population
    
    print("No solution found.")
    return None

solution = genetic_algorithm()
if solution:
    print("Path found:", solution)
else:
    print("No path found.")


word
import random

def is_one_letter_diff(word1, word2):
    """Check if two words differ by exactly one letter."""
    return sum(c1 != c2 for c1, c2 in zip(word1, word2)) == 1

def initialize_population(start_word, target_word, population_size):
    """Create an initial population of sequences."""
    population = []
    for _ in range(population_size):
        individual = [start_word]
        while individual[-1] != target_word:
            next_word = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') if c == target_word[i] else c for i, c in enumerate(individual[-1]))
            individual.append(next_word)
        population.append(individual)
    return population

def fitness(individual, target_word):
    """Calculate fitness based on how close the individual is to the target."""
    current_word = individual[-1]
    if current_word == target_word:
        return float('inf')  # Highest fitness for the correct word
    return -sum(c1 != c2 for c1, c2 in zip(current_word, target_word))

def select_parents(population):
    """Select two parents from the population based on fitness."""
    weights = [fitness(ind) for ind in population]
    return random.choices(population, weights=weights, k=2)

def crossover(parent1, parent2):
    """Crossover between two parents to create two children."""
    split = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child1 = parent1[:split] + parent2[split:]
    child2 = parent2[:split] + parent1[split:]
    return child1, child2

def mutate(individual, valid_words):
    """Randomly mutate an individual."""
    if random.random() < 0.1:  # Mutation rate
        for i in range(len(individual)):
            if random.random() < 0.5:  # Mutate half the time
                possible_words = [word for word in valid_words if is_one_letter_diff(individual[i], word)]
                if possible_words:
                    individual[i] = random.choice(possible_words)
    return individual

def genetic_algorithm(start_word, target_word, valid_words, population_size=100, generations=1000):
    """Run the genetic algorithm to find the word ladder."""
    population = initialize_population(start_word, target_word, population_size)
    
    for generation in range(generations):
        population = sorted(population, key=lambda ind: fitness(ind, target_word), reverse=True)

        if fitness(population[0], target_word) == float('inf'):
            print(f"Solution found in generation {generation}: {population[0]}")
            return population[0]

        new_population = population[:population_size // 2]

        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, valid_words))
            new_population.append(mutate(child2, valid_words))

        population = new_population

    print("No transformation sequence exists.")
    return None

# Example usage
start = "hit"
target = "cog"
words = {"hot", "dot", "dog", "lot", "log", "cog"}

result = genetic_algorithm(start, target, words)
print("Transformation sequence:", result)





nqueen 
import random

def generate_initial_population(n, population_size):
    """Generate an initial population of N-Queens arrangements."""
    return [random.sample(range(n), n) for _ in range(population_size)]

def calculate_fitness(arrangement):
    """Calculate fitness based on non-threatening queens."""
    non_attacking_pairs = 0
    n = len(arrangement)
    
    for i in range(n):
        for j in range(i + 1, n):
            if arrangement[i] != arrangement[j] and abs(arrangement[i] - arrangement[j]) != j - i:
                non_attacking_pairs += 1
    
    return non_attacking_pairs

def select_parents(population):
    """Select two parents from the population based on fitness."""
    weights = [calculate_fitness(ind) for ind in population]
    return random.choices(population, weights=weights, k=2)

def crossover(parent1, parent2):
    """Crossover between two parents to create two children."""
    split = random.randint(1, len(parent1) - 1)
    child1 = parent1[:split] + parent2[split:]
    child2 = parent2[:split] + parent1[split:]
    return child1, child2

def mutate(individual):
    """Randomly mutate an individual by changing the position of a queen."""
    if random.random() < 0.1:  # Mutation rate
        idx = random.randint(0, len(individual) - 1)
        individual[idx] = random.randint(0, len(individual) - 1)
    return individual

def genetic_algorithm(n, population_size=100, generations=1000):
    """Run the genetic algorithm to solve the N-Queens problem."""
    population = generate_initial_population(n, population_size)

    for generation in range(generations):
        population = sorted(population, key=lambda ind: calculate_fitness(ind), reverse=True)

        if calculate_fitness(population[0]) == (n * (n - 1)) // 2:
            print(f"Solution found in generation {generation}: {population[0]}")
            return population[0]

        new_population = population[:population_size // 2]

        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))

        population = new_population

    print("No solution found.")
    return None

# Example usage
n = 8  # Number of queens
solution = genetic_algorithm(n)
print("Arrangement of queens:", solution)