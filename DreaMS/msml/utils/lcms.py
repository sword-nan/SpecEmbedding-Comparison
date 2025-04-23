import pandas as pd

# TODO: all code from data/MassIVE/scripts?
# TODO: blank samples removal
# TODO: instrument/scpecies cleaning


def standartize_species(species: pd.Series):

    # Lowercase
    species = species.str.lower()

    # Add NCBITaxon suffix if known from other entries
    ncbi_suffix = ' (NCBITaxon:'.lower()
    species_to_ncbi = {s.split(ncbi_suffix)[0]: s.split(ncbi_suffix)[1] for s in species.unique().tolist() if isinstance(s, str) and ncbi_suffix in s}
    species = species.apply(lambda s: s if s not in species_to_ncbi else s + ncbi_suffix + species_to_ncbi[s])

    # Manually merge similar species
    species_merged = [
        (['Homo sapiens (NCBITaxon:9606)', 'homo sapiens', 'human', 'Human'], 'Human'),
        (['Mus musculus domesticus', 'Mus musculus (NCBITaxon:10090)', 'Rattus norvegicus (NCBITaxon:10116)', 'Rattus (NCBITaxon:10114)', 'C57BL/6N', 'Mus sp. (NCBITaxon:10095)', 'mice', 'Mice'], 'Mice'),
        (['Ocean Environmental Samples', 'environmental samples <Bacillariophyta> (NCBITaxon:33858)', 'environmental samples <Verrucomicrobiales> (NCBITaxon:48468)', 'environmental samples <delta subdivision> (NCBITaxon:34033)'], 'Environmental')
    ]
    species_merge_map = {}
    for k, v in species_merged:
        for s in k:
            species_merge_map[s.lower()] = v.lower()
    species = species.apply(lambda s: species_merge_map[s] if s in species_merge_map else s)

    # Other
    species = species.rename({'': 'other'})

    return species