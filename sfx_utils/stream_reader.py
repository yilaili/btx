def c2(a):
    return m.cos(a) * m.cos(a)

def s2(a):
    return m.sin(a) * m.sin(a)

# Return 1/d for hkl in cell, in 1/Angstroms
def resolution(scell, shkl):

    a = float(scell[0])*10.0
    b = float(scell[1])*10.0
    c = float(scell[2])*10.0  # nm -> Angstroms

    al = m.radians(float(scell[3]))
    be = m.radians(float(scell[4]))
    ga = m.radians(float(scell[5])) # in degrees

    h = int(shkl[0])
    k = int(shkl[1])
    l = int(shkl[2])

    pf = 1.0 - c2(al) - c2(be) - c2(ga) + 2.0*m.cos(al)*m.cos(be)*m.cos(ga)
    n1 = h*h*s2(al)/(a*a) + k*k*s2(be)/(b*b) + l*l*s2(ga)/(c*c)
    n2a = 2.0*k*l*(m.cos(be)*m.cos(ga) - m.cos(al))/(b*c)
    n2b = 2.0*l*h*(m.cos(ga)*m.cos(al) - m.cos(be))/(c*a)
    n2c = 2.0*h*k*(m.cos(al)*m.cos(be) - m.cos(ga))/(a*b)

    return m.sqrt((n1 + n2a + n2b + n2c) / pf)

def read_stream(filedir='', exp='cxilv4418', run=10, tag='post1'):
    # code probably adapted from crystfel (can not remember)
    keyword=f'{exp}_{run:04d}_{tag}'
    data = []
    n=0
    in_list = 0
    cell = []

    f = open(f'{filedir}{keyword}.stream')
    for line in f:

        if line.find("Cell parameters") != -1:
            cell[0:3] = line.split()[2:5]
            cell[3:6] = line.split()[6:9]
            continue
        if line.find("Reflections measured after indexing") != -1:
            in_list = 1
            continue
        if line.find("End of reflections") != -1:
            in_list = 0
        if in_list == 1:
            in_list = 2
            continue
        elif in_list != 2:
            continue

        # From here, we are definitely handling a reflection line

        # Add reflection to list
        columns = line.split()
        n += 1
        try:
            data.append([resolution(cell, columns[0:3]),columns[3],columns[4],columns[5],columns[6]])
        except:
            print("Error with line: "+line.rstrip("\r\n"))
            print("Cell: "+str(cell))

        if n%1000==0:
            sys.stdout.write("\r%i predicted reflections found" % n)
            sys.stdout.flush()

    return np.asarray(data,dtype=float)


def make_figure(filedir='', exp='cxilv4418', run=10, det='jungfrau4M', tag='post1',
                savefig=False, figsize=12, dpi=300):
    keyword = f'{exp}_{run:04d}_{det}'
    peakogram_bins = [500, 500]

    fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
    fig.suptitle(f'{keyword}_{tag}')
    gs = fig.add_gridspec(2, 2)

    stream_data = read_stream(filedir=filedir, exp=exp, run=run, tag=tag)

    irow = 0
    ax1 = fig.add_subplot(gs[irow, 0:])
    ax1.set_title(f'Peakogram ({stream_data.shape[0]} reflections)',
                  fontdict={'fontsize': 8})
    y = np.log10(stream_data[:, 3])
    x = stream_data[:, 0]
    H, xedges, yedges = np.histogram2d(y, x, bins=peakogram_bins)
    im = ax1.pcolormesh(yedges, xedges, H, cmap='gray', norm=LogNorm())
    plt.colorbar(im)
    ax1.set_xlabel("1/d (A^-1)")
    ax1.set_ylabel("log(peak intensity)")


    irow += 1
    ax2 = fig.add_subplot(gs[irow, 0])
    im = ax2.hexbin(stream_data[:, 1], stream_data[:, 3],
                    gridsize=100, mincnt=1,
                    norm=LogNorm(), cmap='gray')
    ax2.plot(stream_data[:, 3], stream_data[:, 3], label='y=x', linewidth=0.5)
    ax2.plot(-stream_data[:, 3], stream_data[:, 3], label='y=-x', linewidth=0.5)
    ax2.set_xlabel('sum in peak')
    ax2.set_ylabel('max in peak')
    ax2.legend()
    plt.colorbar(im)

    ax3 = fig.add_subplot(gs[irow, 1])
    im = ax3.hexbin(stream_data[:, 2], stream_data[:, 3],
                    gridsize=100, mincnt=1,
                    norm=LogNorm(), cmap='gray')
    ax3.plot(stream_data[:, 3], stream_data[:, 3], label='y=x', linewidth=0.5)
    ax3.set_xlabel('sig(I)')
    ax3.set_yticks([])
    ax3.legend()
    plt.colorbar(im)

    if savefig:
        plt.savefig(f'{filedir}{tag}/{keyword}.png')
    else:
        plt.show()
        plt.tight_layout()