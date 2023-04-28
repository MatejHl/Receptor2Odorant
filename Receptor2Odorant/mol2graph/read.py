import pandas
from Bio.SeqIO.FastaIO import SimpleFastaParser

def read_fasta(filepath):
    with open(filepath) as fasta_file:
        ids = []
        seq = []
        for title, sequence in SimpleFastaParser(fasta_file):
            ids.append(title)
            seq.append(sequence)
    return pandas.Series(data = seq, index = ids)