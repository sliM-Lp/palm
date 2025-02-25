from Bio.PDB import PDBParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB.Polypeptide import Polypeptide
import os



def extract_sequence(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)
    sequences = []
    for model in structure:
        for chain in model:
            sequence = ""
            for residue in chain:
                if residue.get_resname() in Polypeptide.three_to_one:
                    sequence += Polypeptide.three_to_one(residue.get_resname())
            if sequence:
                seq_record = SeqRecord(Seq(sequence), id=chain.id, description="")
                sequences.append(seq_record)
    return sequences


scop_pdb_path = '../data/scop40pdb/pdb'
sequences = dict()

for file in os.listdir(scop_pdb_path):
    pdb_path = os.path.join(scop_pdb_path,file)
    sequences = extract_sequence(pdb_path)

    sequences[file.split('.pdb')[0]] = sequences[0]
    for seq_record in sequences:
        print(seq_record.format("fasta"))

if __name__ == "__main__":
    pdb_file = "your_pdb_file.pdb"
    sequences = extract_sequence(pdb_file)
    for seq_record in sequences:
        print(seq_record.format("fasta"))