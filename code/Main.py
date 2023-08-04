import matplotlib.pyplot as plt
from bb_analyzer import BbAnalyzer
import sys

def main():
    if not sys.argv[0]:
        print("No directory specified!")
        exit(-1)

    myBa = BbAnalyzer("/home/antonio123/workspace/Github_projects/DefectAnalysis/data/_annotations.coco.json", "/home/antonio123/workspace/Github_projects/DefectAnalysis/data/unlabeled_images/*.jpg")
    """
    fig1 = plt.figure()
    myBa.createSdHistogram(bins=[25, 20])
    plt.title("Spatial defect centroid distribution (bins = 25 X 20)")
    fig2 = plt.figure()
    myBa.createSdHistogram(bins=[50, 40])
    plt.title("Spatial defect centroid distribution (bins = 50 X 40)")
    fig3 = plt.figure()
    myBa.createSdHistogram(bins=[90, 72])
    plt.title("Spatial defect centroid distribution (bins = 90 X 72)")
    """
    fig1 = plt.figure()
    myBa.createDefSizePerImgHistogram()
    if len(sys.argv) > 1:
        fig1.savefig(sys.argv[1] % 'defect_size_per_image_dist')
        #fig2.savefig(sys.argv[1] % 'real_centroid_distribution_(50X40)')
        #fig3.savefig(sys.argv[1] % 'real_centroid_distribution_(90X72)')
    else:
        plt.show()

if __name__ == "__main__":
        main()
