# Genetic Tree Classifier

## Overview

This repository contains an implementation of a genetic algorithm for evolving decision trees for classification tasks. The decision tree structure is defined by the `Node` class, and the entire genetic algorithm is orchestrated by the `Genetic` class. The project was developed by Rafael Souza, João Augusto Fernandes, Thiago Ambiel Arthur Rezende, and João Pedro.

## Classes

### 1. `Node`

The `Node` class represents a node in the decision tree. Each node has attributes such as `name`, `value`, `cond_type`, `depth`, and `leaf`. The `decision` method is used to make decisions based on the node's conditions.

### 2. `Tree`

The `Tree` class is responsible for creating the initial decision tree and defining its structure. It uses the `Node` class to represent nodes. The decision tree is then evolved using a genetic algorithm, and the best tree is determined based on classification accuracy.

### 3. `Genetic`

The `Genetic` class implements the genetic algorithm for evolving decision trees. It includes methods for initializing the population, evaluating the fitness of individuals, and performing crossover and mutation operations to evolve the population.

## Usage

To use this implementation, follow these steps:

1. **Create a Tree:**

   ```python
   # Example usage
   tree = Tree(all_features, targets, max_depth=5)
   tree.create()
# Example usage
genetic_algorithm = Genetic(pop_size=10, epochs=100, model=tree, mutation_rate=4e-2)
# Example usage
genetic_algorithm.train(data, target, bar_train=True, desc='Training')
# Example usage
tree.show()


