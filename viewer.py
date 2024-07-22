import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import argparse

# Load the data
#data = pd.read_csv('lesstox_min_data_all_generations.csv')
#data = pd.read_csv('moretox_min_data_all_generations.csv')
data = pd.read_csv('moretox_max_data_all_generations.csv')


def get_parents(row):
    if isinstance(row['Generation, method this molecule was created, and parent ID numbers'], str) and 'crossover' in row['Generation, method this molecule was created, and parent ID numbers']:
        parent_info = eval(row['Generation, method this molecule was created, and parent ID numbers'])
        return [int(float(parent_id.strip())) for parent_id in list(parent_info.values())[0].split(',')[1:]]
    return []

def trace_ancestry(molecule_id, max_depth=10):
    ancestry = []
    current_id = molecule_id
    visited = set()
    depth = 0
    
    while depth < max_depth:
        if current_id in visited:
            print(f"Warning: Cycle detected at Molecule ID {current_id}")
            break
        
        visited.add(current_id)
        parent_df = data.loc[data['compound_id'] == current_id]
        
        if parent_df.empty:
            print(f"Warning: Molecule ID {current_id} not found in the dataset.")
            break
        
        ancestry.append(current_id)
        parents = get_parents(parent_df.iloc[0])
        if not parents:
            break
        
        current_id = parents[0]  # We'll follow the first parent for simplicity
        depth += 1
    
    return list(reversed(ancestry))

def create_molecule_image(smiles, fitness):
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, size=(300, 300))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Fitness: {fitness:.3f}")

def visualize_evolution(start_range, end_range, min_lineage_length=2, output_file='molecule_evolution_large_plot.png'):
    start_molecule_ids = range(start_range, end_range + 1)

    valid_lineages = []
    for start_molecule_id in start_molecule_ids:
        ancestry = trace_ancestry(start_molecule_id)
        if len(ancestry) >= min_lineage_length:
            valid_lineages.append((start_molecule_id, ancestry))

    if not valid_lineages:
        print(f"No valid lineages found with at least {min_lineage_length} molecules.")
        return

    plt.figure(figsize=(20, 5 * len(valid_lineages)))

    for idx, (start_molecule_id, ancestry) in enumerate(valid_lineages):
        num_molecules = len(ancestry)

        for i, molecule_id in enumerate(ancestry):
            molecule_data = data[data['compound_id'] == molecule_id]
            if molecule_data.empty:
                print(f"Warning: Molecule ID {molecule_id} not found in the dataset.")
                continue
            
            molecule_data = molecule_data.iloc[0]
            smiles = molecule_data['smiles']
            fitness = molecule_data['fitness']
            
            plt.subplot(len(valid_lineages), num_molecules, idx * num_molecules + i + 1)
            create_molecule_image(smiles, fitness)
            plt.title(f"ID: {molecule_id}\nFitness: {fitness:.3f}")

        # Print the evolution path
        print(f"\nEvolution path for starting molecule {start_molecule_id}:")
        for i, molecule_id in enumerate(ancestry):
            molecule_data = data[data['compound_id'] == molecule_id]
            if molecule_data.empty:
                print(f"Generation {i+1}: Molecule ID {molecule_id} not found in the dataset.")
            else:
                molecule_data = molecule_data.iloc[0]
                print(f"Generation {i+1}: Molecule ID {molecule_id}, Fitness: {molecule_data['fitness']:.3f}")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nLarge plot has been saved as '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize molecule evolution")
    parser.add_argument("--start", type=int, default=0, help="Start of molecule ID range")
    parser.add_argument("--end", type=int, default=194, help="End of molecule ID range")
    parser.add_argument("--min-length", type=int, default=4, help="Minimum lineage length to include")
    parser.add_argument("--output", type=str, default="molecule_evolution_large_plot.png", help="Output file name")

    args = parser.parse_args()

    visualize_evolution(args.start, args.end, args.min_length, args.output)