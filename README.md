# Genetic Forest Classifier

## Overview

This repository contains an implementation of a genetic algorithm for evolving decision trees for classification tasks. The decision tree structure is defined by the `Node` class, and the entire genetic algorithm is orchestrated by the `Genetic` class. 

## Classes

### 1. `Node`

The `Node` class represents a node in the decision tree. Each node has attributes such as `name`, `value`, `cond_type`, `depth`, and `leaf`. The `decision` method is used to make decisions based on the node's conditions.

### 2. `Tree`

The `Tree` class is responsible for creating the initial decision tree and defining its structure. It uses the `Node` class to represent nodes. The decision tree is then evolved using a genetic algorithm, and the best tree is determined based on classification accuracy.

### 3. `GenForest`

The `GenForest` class implements a Genetic Forest, an ensemble of decision trees using genetic algorithms. It utilizes the `Tree`, `Node`, and `Genetic` classes to train the ensemble and make predictions.

## Usage

To use this implementation, follow these steps:

```python
# Import necessary libraries and classes
import numpy as np
from gen_forest import GenForest

# Example data
all_features = ["feature1", "feature2", "feature3"]
targets = np.array([0, 1, 0, 1, 1])

# Create a GenForest instance
genetic_forest = GenForest(features=all_features, targets=targets, n_species=5, n_features=3, n_agents=10, epochs=50, n_deaths=5, rounds_deaths=3, seed=123)

# Fit the GenForest model on training data
x_train = np.random.rand(100, 3)
y_train = np.random.randint(2, size=100)
genetic_forest.fit(x_train, y_train)

# Make predictions using the GenForest model
x_test = np.random.rand(10, 3)
predictions = genetic_forest.predict(x_test, bests=True)

# Evaluate accuracy
accuracy_value = genetic_forest.accuracy(predictions, y_test)
print("Accuracy:", accuracy_value)
```
### Contributors
- Rafael Souza
- João Augusto Fernandes
- Thiago Ambiel Arthur Rezende
- João Pedro Farjoun Silva
