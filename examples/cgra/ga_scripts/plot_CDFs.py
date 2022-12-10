import matplotlib.pyplot as plt
import argparse
import numpy

def load_cdf_from_file(f):
    with open(f) as fle:
        lines = ''.join(fle.readlines())
        data = []
        # Should only be one-line files?
        for item in lines.split(','):
            if item.strip():
                try:
                    data.append(abs(int(item.strip())))
                except:
                    # Plenty of reasons this could fail -- mostly due to

                    # incomplete runs
                    pass

        print ("Loaded ", len(data), "items")
        return data

def compute_cdf(data):
    sorted_data = sorted(data)
    x_points = range(0, max(sorted_data))
    cdf = [0.0] * len(x_points)
    sum_so_far = 0
    cdf_pointer = 0
    value_per_point = 1 #Treated as int to avoid FP accum issues.
    for point in sorted_data:
        cdf_pointer = int(float(len(x_points)) * float(point) / float(max(sorted_data))) - 1
        sum_so_far += value_per_point

        cdf[cdf_pointer] = float(sum_so_far) / float(len(sorted_data))

    return x_points, cdf


def plot_datas(datas, names):
    # First, compute the CDF from the raw data
    cdfs = []
    xvals = []
    for data in datas:
        xvalus, cdf = (compute_cdf(data))
        xvals.append(xvalus)
        cdfs.append(cdf)

    xvs_max = 0
    for i in range(len(cdfs)):
        plt.plot(xvals[i], cdfs[i], label=names[i])
        xvs_max = max(max(xvals[i]), xvs_max)

    plt.ylim([0.0, 1.0])
    plt.xlim([0,  xvs_max])
    plt.ylabel('CDF')
    plt.xlabel('InitializationInterval')
    plt.legend()
    plt.savefig('cdfs.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Alternate between files and names.
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()

    datas = []
    names = []
    name = False
    for file in args.files:
        if name:
            names.append(file)
        else:
            data = load_cdf_from_file(file)
            datas.append(data)
        name = not name

    plot_datas(datas, names)
