from GPTree import GPTree, np, randint
from copy import deepcopy
import re
from display import display 
import sys
from random import seed 
import seaborn as sns
import matplotlib.pyplot as plt

seed(0)
np.random.seed(0)

def gen_plot(df, y, save_plot=False):
    sns.set_theme(style="whitegrid", font_scale=1)

    a4_dims = (11.7, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)

    plot = sns.lineplot(
        data=df, y=y, x='Generation', style='Type', hue='Type', markers=True, dashes=False, linewidth = 3
    )
    plt.show()
    if save_plot:
        plt.savefig('compare_evolutions.svg', format="svg", bbox_inches='tight')

def generate_dataset(dataset_file):
    all_vars = []
    y_values = []
    variables = []
    with open(dataset_file, 'r') as reader:
        for line in reader:
            ds_list = list(filter(None, re.sub(r"\s+", '  ', line).split('  ')))
            ds_var = {}
            for i in range(len(ds_list) - 1):
                ds_var[chr(97+i)] = float(ds_list[i])
                variables.append(chr(97+i))
            all_vars.append(ds_var)
            y_values.append(float(ds_list[len(ds_list) - 1]))
    return all_vars, y_values, variables

def init_population(pop_size, max_depth_t, min_depth_t, max_tree_size, all_vars, y_values, variables): 
    pop = []
    pop_eval = []
    half_n_half = [True, False]
    while len(pop) < pop_size:
        for md in range(3, max_depth_t + 1):
            if len(pop) == pop_size:
                break
            for _grow in half_n_half:
                for _ in range(int(pop_size/10)):
                    if len(pop) == pop_size:
                        break
                    t = GPTree()
                    t.random_tree(variables, min_depth_t, grow=_grow, max_depth=md)
                    is_valid, evaluation = validate_tree(t, all_vars, y_values, max_tree_size)
                    if (is_valid):
                        pop.append(t)
                        pop_eval.append(evaluation)
    return pop, pop_eval

def validate_tree(individual, all_vars, y_values, max_tree_size):
    if individual.size() > max_tree_size:
        return False, None
    results = [individual.execute_tree(ds) for ds in all_vars]
    duplicates = len(results) - len(set(results))
    dataset_dup = len(y_values) - len(set(y_values))
    res_var = int(abs(np.var(results)))
    if duplicates > dataset_dup or res_var == 0:
        return False, None
    else:
        return True, results

def fitness(eval_v, y_values): #RMSE
    return np.mean((np.sqrt([((eval_v[i] - y_values[i]) ** 2) for i in range(len(y_values))])))

def selection(population, fitnesses, cov_parsimony, evaluation_v, ds_var, co_var, tour_size):
    tournament = [randint(0, len(population)-1) for _ in range(tour_size)]
    tournament_fitness = []
    for i in range(tour_size):
        parsimony_pen = population[tournament[i]].size() * cov_parsimony
        var_pen = 1 * co_var * np.var([evaluation_v[tournament[i]], ds_var])
        tournament_fitness.append(fitnesses[tournament[i]] + parsimony_pen + var_pen)
    tournament_index = tournament[tournament_fitness.index(min(tournament_fitness))]
    return deepcopy(population[tournament_index]), tournament_index

def treat_population(population, pop_evaluation, gen_fitnesses, pop_size):
    evaluation_mean_v, pop_tree_size = [], []
    for i in range (pop_size):
        evaluation_mean_v.append(np.var(pop_evaluation[i]))
        pop_tree_size.append(population[i].size())
    co_var = np.var(evaluation_mean_v)
    cov_parsimony = (np.cov(pop_tree_size, gen_fitnesses) / np.var(pop_tree_size))[0][1]
    gen_duplicates = len(gen_fitnesses) - len(set(gen_fitnesses))
    i_best_f = np.argmin(gen_fitnesses)
    i_worst_f = np.argmax(gen_fitnesses)
    gen_avg_f = np.mean(gen_fitnesses)
    return evaluation_mean_v, cov_parsimony, co_var, gen_avg_f, gen_duplicates, i_best_f, i_worst_f

def print_stats_gen(gen, worst_f_indv, worst_tree, best_f_indv, best_tree, avg_f_indv, worst_cross_indv, best_cross_indv, duplicates):
    print("\n\n__________________________________________\n")
    print(f"Gen {gen}")
    print(f"Worst f={worst_f_indv}")
    display(worst_tree)    
    print(f"\nBest f={best_f_indv}")
    display(best_tree)  
    print(f"\nGen {gen} Mean fitness: {avg_f_indv}")
    print(f"Worst children # from xover: {worst_cross_indv}")
    print(f"Best children # from xover: {best_cross_indv}")
    print(f"Duplicates: {duplicates}")

def print_final_stats(gen_size, gen_best, best_f, best_tree, gen_worst, worst_f, worst_tree, avg_f):
    print("\n\n__________________________________________\n")
    print(f"Total Gen #: {gen_size}")
    print(f"\nBest indv comes from Gen {gen_best} with f={best_f}")
    display(best_tree)
    print(f"\nWorst indv comes from Gen {gen_worst} with f={worst_f}")
    display(worst_tree) 
    print(f"\nMean f from all Gens: {avg_f}")

def evolution(input_path, gen_size=2, pop_size=10, max_depth=7, min_depth=3, cross_rate=0.6, mut_rate=0.3, tour_size=3, print_stats=True):

    max_tree_size = max_depth ** 4
    all_vars, y_values, variables = generate_dataset(input_path)
    ds_var = np.var(y_values)

    avg_f = []
    best_f = []
    worst_f = []
    best_tree = []
    worst_tree = []
    duplicates = []
    best_xover = []
    worst_xover = []

    for gen in range(gen_size):

        gen_worst_xover = 0
        gen_best_xover = 0

        if gen == 0:
            population, pop_evaluation = init_population(pop_size, max_depth, min_depth, max_tree_size, all_vars, y_values, variables)
            gen_fitnesses = [fitness(pop_evaluation[i], y_values) for i in range(pop_size)]
            evaluation_mean_v, cov_parsimony, co_var, gen_avg_f, gen_duplicates, i_best_f, i_worst_f = treat_population(population, pop_evaluation, gen_fitnesses, pop_size)
            
        else:
            next_gen_pop = [best_tree[gen - 1]]
            next_gen_f = [best_f[gen - 1]]
            next_gen_eva = [pop_evaluation[i_best_f]]
            
            while len(next_gen_pop) < pop_size:
                first_parent, fp_index = selection(population, gen_fitnesses, cov_parsimony, evaluation_mean_v, ds_var, co_var, tour_size)
                second_parent, sp_index = selection(population, gen_fitnesses, cov_parsimony, evaluation_mean_v, ds_var, co_var, tour_size)
                parents_f = (gen_fitnesses[fp_index] + gen_fitnesses[sp_index]) / 2
                aux_t = deepcopy(first_parent)
                first_parent.crossover(cross_rate, second_parent)
                first_parent.mutation(mut_rate, variables, min_depth)
                is_valid, evaluation = validate_tree(first_parent, all_vars, y_values, max_tree_size)
                if is_valid:
                    child_f = fitness(evaluation, y_values)
                    if child_f < parents_f:
                        gen_best_xover += 1
                    elif child_f > parents_f:
                        gen_worst_xover += 1
                    next_gen_pop.append(first_parent)
                    next_gen_f.append(child_f)
                    next_gen_eva.append(evaluation)
                if len(next_gen_pop) < pop_size:
                    second_parent.crossover(cross_rate, aux_t)
                    second_parent.mutation(mut_rate, variables, min_depth)
                    is_valid, evaluation = validate_tree(second_parent, all_vars, y_values, max_tree_size)
                    if is_valid:
                        child_f = fitness(evaluation, y_values)
                        if child_f < parents_f:
                            gen_best_xover += 1
                        elif child_f > parents_f:
                            gen_worst_xover += 1
                        next_gen_pop.append(second_parent)
                        next_gen_f.append(child_f)
                        next_gen_eva.append(evaluation)

            population = next_gen_pop
            pop_evaluation = next_gen_eva
            gen_fitnesses = next_gen_f
            evaluation_mean_v, cov_parsimony, co_var, gen_avg_f, gen_duplicates, i_best_f, i_worst_f = treat_population(population, pop_evaluation, gen_fitnesses, pop_size)

        avg_f.append(gen_avg_f)
        best_f.append(gen_fitnesses[i_best_f])
        worst_f.append(gen_fitnesses[i_worst_f])
        best_tree.append(deepcopy(population[i_best_f]))
        worst_tree.append(deepcopy(population[i_worst_f]))
        duplicates.append(gen_duplicates)
        worst_xover.append(gen_worst_xover)
        best_xover.append(gen_best_xover)

        if print_stats:
            print_stats_gen(gen, worst_f[gen], worst_tree[gen], best_f[gen], best_tree[gen], avg_f[gen], gen_worst_xover, gen_best_xover, gen_duplicates)

        if np.amin(best_f) == 0:
            break
        
    gen_worst = np.argmax(worst_f)
    gen_best = np.argmin(best_f)

    if print_stats:
        print_final_stats(gen_size, gen_best, np.amin(best_f), best_tree[gen_best], gen_worst, np.amax(worst_f), worst_tree[gen_worst], np.mean(avg_f))

    return avg_f, worst_f, best_f, duplicates, worst_xover, best_xover

def main():

    if len(sys.argv) != 2:
        print('Usage: tp1.py dataset.txt')
        sys.exit()

    input_path = sys.argv[1]
    avg_f, worst_f, best_f, duplicates, worst_xover, best_xover = evolution(input_path)
    print(len(avg_f))
    print(len(worst_f))
    print(len(best_f))
    print(len(duplicates))
    print(len(worst_xover))
    print(len(best_xover))

    
if __name__== "__main__":
  main()

