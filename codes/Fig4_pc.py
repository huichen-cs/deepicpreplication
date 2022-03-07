import pandas as pd
import sys
from matplotlib import pyplot as plt

def main(argv):
    if len(argv) < 3:
        print('Usage: {} pc_segment_file.csv pc_segment_fig.pdf')
        sys.exit(0)
    if len(argv) > 3:
        factor = float(argv[3])
    else:
        factor = 1
    if len(argv) > 4:
        nbins=int(argv[4])
    else:
        nbins=5
    df = pd.read_csv(argv[1])
    assert 'la' in df.columns #LA
    assert 'nf' in df.columns #NF
    plt.figure(figsize=(4*0.25/0.20, 3.5))
    x = df['la'] # NF
    y = df['nf']
    plt.plot(x, y, 'k-')
    plt.locator_params(axis='x', nbins=nbins)
    plt.xticks(fontsize=14*0.25/0.20*factor)
    plt.yticks(fontsize=14*0.25/0.20*factor)    
    plt.xlabel('LA', fontsize=16*0.25/0.20*factor)
    plt.ylabel('NF', fontsize=16*0.25/0.20*factor)
    plt.tight_layout()
    plt.savefig(argv[2], bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    main(sys.argv)



