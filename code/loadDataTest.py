import nibabel as nib
import torch
import numpy as np


# load the names of all the individuals in the dataset
with open('/home1/mforbush/grips2024/individuals.txt', 'r') as f : individuals = [int(ind) for ind in f.readlines()]


# load the first individual
individual = individuals[0]; print(individual)
dataPath = f'/public/home2/yqiao_group/yqiao/data/HCP/HCP_3T/{individual}/Diffusion/data/'  # HCP pre-processed data
dataFile = 'data.nii.gz'

img = nib.load(dataPath + dataFile)
data = img.get_fdata()

print('data shape', data.shape)

avgActivation = data.mean(axis=3)

print('shape of avg activation', avgActivation.shape)
print('avg activation', data.mean())


# avg shape of all the data
print(len(individuals), 'individuals')

shapes = []
problemIndividuals = []
for ind in individuals : 
    try : 
        shapes.append(nib.load(f'/public/home2/yqiao_group/yqiao/data/HCP/HCP_3T/{int(ind)}/Diffusion/data/' + dataFile).get_fdata().shape)
        print('loaded', ind)

    except Exception as e : 
        print('--> problem with', ind, e)
        problemIndividuals.append(ind)

print('avg shape of the data', np.array(shapes).mean(axis=0))

with open('/home1/mforbush/grips2024/problemIndividuals.txt', 'w') as f : f.write('\n'.join([f'{ind}' for ind in problemIndividuals]))