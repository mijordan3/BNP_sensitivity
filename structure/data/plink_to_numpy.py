# adapted from
# https://github.com/rgiordan/fastStructure/blob/master/read_bed_file.py

import numpy as np
import parse_bed
import getopt
import sys
import pdb

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def parseopts(opts):

    """
    parses the command-line flags and options passed to the script
    """

    params = { 'format': 'bed' }

    for opt, arg in opts:
        if opt in ["--input"]:
            params['inputfile'] = arg

        elif opt in ["--format"]:
            params['format'] = arg

        elif opt in ["--output"]:
            params['outputfile'] = arg

    return params

def checkopts(params):

    """
    checks if some of the command-line options passed are valid.
    In the case of invalid options, an exception is always thrown.
    """

    if params['format'] not in ['bed','str']:
        print("%s data format is not currently implemented")
        raise ValueError

    if not 'inputfile' in params:
        print("an input file needs to be provided")
        raise KeyError

    if not 'outputfile' in params:
        print("an output file needs to be provided")
        raise KeyError

def usage():

    """
    brief description of various flags and options for this script
    """

    print("\nHere is how you can use this script\n")
    print("Usage: python %s"%sys.argv[0])
    print("\t --input=<file>")
    print("\t --format={bed,str} (default: bed)")
    print("\t --output=<file>")


if __name__=="__main__":

    # parse command-line options
    argv = sys.argv[1:]
    smallflags = "K:"
    bigflags = ["input=", "format=", "output="]
    try:
        opts, args = getopt.getopt(argv, smallflags, bigflags)
        if not opts:
            usage()
            sys.exit(2)
    except getopt.GetoptError:
        print("Incorrect options passed")
        usage()
        sys.exit(2)

    params = parseopts(opts)

    # check if command-line options are valid
    try:
        checkopts(params)
    except (ValueError,KeyError):
        sys.exit(2)

    # load data
    if params['format']=='bed':
        G = parse_bed.load(params['inputfile'])
    elif params['format']=='str':
        G = parse_str.load(params['inputfile'])

    # convert to one-hot encoding
    G_one_hot = get_one_hot(G, nb_classes = 4)
    # G = 3 means that data is missing
    # see https://github.com/rajanil/fastStructure/blob/master/parse_bed.pyx
    # one-hot-encoding will be all zeros -- won't enter the loglikelihood
    # so we only take the first three columns of the one-hot-encoding
    G_one_hot = G_one_hot[:, :, 0:3]
    np.savez(params['outputfile'],
                g_obs_raw=G,
                g_obs = G_one_hot)

    # # Write the genome file.
    # handle = open('%s.genome' % (params['outputfile']), 'w')
    # handle.write('\n'.join(['  '.join(['%d' % i for i in g]) for g in G])+'\n')
    # handle.close()