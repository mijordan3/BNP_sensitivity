import csv
import string
import re
import argparse

parser = argparse.ArgumentParser(description='Convert structure to plink format.')
parser.add_argument('--input_file', type=argparse.FileType('r'),
                   help='An input file in structure format')
parser.add_argument('--output_file_base', type=str,
                   help='The basename for the output file.')

args = parser.parse_args()
ped_handle = open('{}.ped'.format(args.output_file_base), 'w')
fam_handle = open('{}.fam'.format(args.output_file_base), 'w')
map_handle = open('{}.map'.format(args.output_file_base), 'w')
missing_handle = open('{}.missing'.format(args.output_file_base), 'w')

rs_number = args.input_file.readline().split()
region_number = args.input_file.readline().split()
chromosome_number = args.input_file.readline().split()
snp_position = args.input_file.readline().split()
core_region_indicator = args.input_file.readline().split()

individuals = [ line.split() for line in args.input_file.readlines() ]

if (len(individuals) % 2 != 0):
    raise IOError('There are not an even number of rows (there should be two per individual)')

n_loci = len(rs_number)

print('Processing {0} individuals and {1} loci.\n'.format(len(individuals), n_loci))

# Write a ped file.
for i in range(int(len(individuals) / 2)):
    row1 = individuals[2 * i]
    row2 = individuals[2 * i + 1]
    row_id = row1[0]

    if (row_id != row2[0]):
        raise IOError('An individual had an odd number of observations')

    if (len(row1) != 7 + n_loci or len(row2) != 7 + n_loci):
        raise IOError('Wrong number of loci for individual {row_id} ({n})'.format(row_id=row_id, n=len(row1) - 7))


    # We will only use the IndividualID column of the ped format.
    start_cols = '{row_id} {row_id} {row_id} {row_id} 1 -9'.format(row_id=row_id)
    ped_handle.write(start_cols)
    fam_handle.write(start_cols + '\n')

    for locus in range(n_loci):
        allele1 = row1[locus + 7]
        allele2 = row2[locus + 7]

        # If one allele at a locus is missing, they must both be, and this is
        # encoded as a 0.
        if (allele1 == '?') or (allele2 == '?'):
            #print 'Warning: at locus {}, only one allele is missing.'.format(locus)
            allele1 = '0'
            allele2 = '0'
            missing_handle.write('{}\n'.format(locus))

        ped_handle.write(' {0} {1}'.format(allele1, allele2))
    ped_handle.write('\n')

ped_handle.close()
fam_handle.close()
missing_handle.close()

# Write a map file.
for locus in range(n_loci):
  # I don't know what the genetic distance is supposed to be.
  map_handle.write('{0} {1} {2} {2}\n'.format(chromosome_number[locus],
                                              rs_number[locus],
                                              snp_position[locus]))
map_handle.close()