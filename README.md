# genetic_algorithm

A Genetic Programming Algorithm developed from scratch to solve Symbolic Regression problems, 
implemented using Python 3.8.

Created as an assignment for the Natural Computing course at UFMG in the second term of 2020 (02/2020).

The `TP 1.ipynb` file is a Jupyter Notebook document that presents plots from the testing results (varying the evolution params) using the datasets `SR_circle.txt` and `SR_ellipse_noise.txt` (similar to the previous, but with noisy `y` values).

A fixed seed value was assigned (0) to ensure the tests' reproducibility.

## Instructions

* Install the required packages by running on terminal:

`pip install -r requirements.txt`

* Run a test by executing on terminal:

`python tp1.py "\path\to\dataset.txt"`

## Implementation

To represent individuals, a binary tree class was created, in which the endpoints can be one of the inputs to the problem or a real random constant in the range -1 to 1. The other nodes are operators (addition, subtraction, multiplication and protected division). To generate the initial population, the *half-and-half* method was chosen with a maximum height of 7 and a minimum of 3. 

The algorithm makes use of the genetic operators of mutation (choosing a random node to replace with a randomly generated subtree) and crossover (selecting by tournament two parents and swapping a subtree from a random node between them). The *Root Mean Square Error (RMSE)* was used as the fitness function, in which the best fitness will be the one closest to zero. In addition, the best individual from the previous generation is added to each subsequent generation (elitism).

To control bloating, the *Covariant Parsimony Pressure Method* detailed in [A Field Guide to Genetic Programming](https://www.researchgate.net/publication/216301261_A_Field_Guide_to_Genetic_Programming) in Section 11.3.2 was used, which at the time of selection will add the multiplied calculated coefficient to the individual's fitness by the size of the individual. In this way, individuals who grow a lot tend to be chosen less, reducing the risk of exorbitant growth. 

To improve the individuals' quality, a series of requirements were enforced to each individual before adding it to the population: the tree size needs to be smaller than the maximum size (7) raised to 4 (a value chosen based upon the algorithm performance/runtime results); the number of individual evaluation results that are duplicated needs to be less than the number of duplicate results from the original dataset (y values); and the approximate integer of the variance value of the results needs to be nonzero. These checks were added as it was observed that during the tests the tendency was for fitness to favor individuals who return very similar values regardless of the input. In addition, it was necessary to establish a limit for the size of the tree not to grow too large to the point of hindering the execution of evolution, while guaranteeing diversity in the population. 

To print the individual in tree format by levels, a solution found on the [stack overflow forum](https://stackoverflow.com/a/54074933) was adapted. It was included to facilitate the visualization of the tests, although it is not favorable when the tree is too large. 


