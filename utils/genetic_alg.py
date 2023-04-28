import numpy as np
import random
import pandas as pd

from utils.dataset import load_dataset, load_labels, convert_to_epochs, load_channels
from utils.features import time_series_features, hjorth_features
from utils.classifiers import KNN, SVM, NN
import utils.variables as v

# This project is extended and a library called PyGAD is released to build the genetic algorithm.
# PyGAD documentation: https://pygad.readthedocs.io
# Install PyGAD: pip install pygad
# PyGAD source code at GitHub: https://github.com/ahmedfgad/GeneticAlgorithmPython

def cal_pop_fitness(equation_inputs, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calcuates the sum of products between each input and 
    # its corresponding weight.
    # In our case this is 4*accuracy + 1*sensitivity + 1*specificity
    fitness = np.sum(pop*equation_inputs, axis=1)
    return fitness

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for 
    # producing the offspring of the next generation.
    parents = np.empty((num_parents, pop.shape[1]), dtype='<U5')
    print(parents)
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
    # Produces offspring with a random combination of the two parents' genes
    n_offspring = offspring_size[0]
    n_genes_in_person = int(offspring_size[1])

    offspring = np.empty(offspring_size, dtype='<U5')
    # The point at which crossover takes place between two parents. 
    # Usually it is at the center.
    crossover_point = round(n_genes_in_person/2)

    offspring_indx = 0
    for i in range(parents.shape[0]):
        for j in range(i+1, parents.shape[0]):
            gene_indx = 0
            while gene_indx < n_genes_in_person:
                if gene_indx < crossover_point:
                    rand_int = random.randint(0,7)
                    if parents[i][rand_int] not in offspring[offspring_indx]:
                        offspring[offspring_indx][gene_indx] = parents[i][rand_int]
                        gene_indx += 1
                else:
                    rand_int = random.randint(0,7)
                    if parents[j][rand_int] not in offspring[offspring_indx]:
                        offspring[offspring_indx][gene_indx] = parents[j][rand_int]
                        gene_indx += 1
            offspring_indx += 1
    return offspring


def make_init_pop(all_data, all_genes, num_genes_in_person, num_people):
    # Makes a random first population
    # Initialize empty population
    init_pop = np.empty([num_people,num_genes_in_person], dtype='<U5')
    person_index = 0

    while person_index!=num_people:
        # Initialize new person
        person = np.empty(num_genes_in_person, dtype='<U5')
        gene_index = 0

        while gene_index!=num_genes_in_person:
            # Gives a random index
            index = random.randint(0,len(all_genes)-1)
            # Checks if the gene is not already in the gene pool of the person
            if all_genes[index] not in person:
                person[gene_index] = all_genes[index]
                gene_index += 1
        init_pop[person_index] = person
        person_index +=1

    # Create labels to match the dataset
    # Creating labels
    subset_data = get_subset(all_data, all_genes, init_pop[0])
    dataset = convert_to_epochs(subset_data, num_genes_in_person, v.SFREQ)
    label = create_labels(dataset)
    
    return init_pop, label


def get_subset(data, all_genes, subset_genes):
    # Retrieves the data that belongs to the subset of genes
    subset_data = np.empty((120, 8, 3200))
    n_genes = 8

    j = 0
    for i in range(len(all_genes)):
        if j < (n_genes + 1) and all_genes[i] in subset_genes:
            subset_data[:,j,:] = data[:,i,:]
            j+=1
    return subset_data


def check_nan(array):
    # Checks if there is any NaN values in array
    # Used for debugging
    x = np.isnan(array)
    if True in x:
        print('NAN in array')
        return 0
    print('No NAN found')

def create_labels(dataset):
    # Loads labels into correct shape
    labels = load_labels()
    label = pd.concat([labels['t1_math'], labels['t2_math'],
                    labels['t3_math']]).to_numpy()
    label = label.repeat(dataset.shape[1])
    return label


def convert_pop_to_fitness(all_data, all_channels, current_pop, label, n_genes):
    # Calculates population fitness (accuracy, sensitivity, specificity)
    data = np.empty((3000, 16))
    new_pop_fitness = np.empty((len(current_pop),3))

    for i in range(len(current_pop)):
        subset_data = get_subset(all_data, all_channels, current_pop[i])
        dataset = convert_to_epochs(subset_data, n_genes, v.SFREQ)
        # hjorth features performs the best
        features = hjorth_features(dataset, n_genes,v.SFREQ)
        data = features
        
        # Use KNN or SVM
        new_pop_fitness[i] = KNN(data, label)

    return new_pop_fitness

def convert_parents_to_fitness(all_data, all_genes, parents, label, num_parents_mating, n_genes):
     # Calculates parents fitness (accuracy, sensitivity, specificity)
    new_pop_fitness = np.empty((num_parents_mating,3))
    data = np.empty((3000, 16))

    for i in range(num_parents_mating):
        
        subset_data = get_subset(all_data, all_genes, parents[i])
        dataset = convert_to_epochs(subset_data, n_genes, v.SFREQ)
        features = hjorth_features(dataset, n_genes,v.SFREQ)
        data = features
        print(f'Parent genes: {parents[i]} ')
        results = KNN(data, label)
        print(results)
        new_pop_fitness[i] = results

    return new_pop_fitness