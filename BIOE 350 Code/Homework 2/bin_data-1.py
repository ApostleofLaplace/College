def bin_data(counts,nbins=10):
    '''
    Uses numpy.histogram to bin the data and returns bin centers
    and counts in each bin, normalized so the sum of the counts
    is equal to 1.

    counts should be a list of integers
    '''
    # use np.histogram to do the counting and set up the bins
    y,x = np.histogram(counts,nbins,density=True)
    # use the bin edges to compute bin centers (last value is junk so drop it)
    bin_centers = (x + np.roll(x,-1))/2

    # Get differences between bin_centers
    bin_diffs = np.diff(bin_centers)

    bin_centers = bin_centers[:-1]

    # return bin centers as x and PDF as y
    return bin_centers,y, bin_diffs
