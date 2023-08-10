import matplotlib.pyplot as plt
from bb_analyzer import BbAnalyzer
import sys, utils

def main():
    if not sys.argv[0]:
        print("No directory specified!")
        exit(-1)

    myBa = BbAnalyzer("/home/antonio123/workspace/Github_projects/DefectAnalysis/data/_annotations.coco.json", "/home/antonio123/workspace/Github_projects/DefectAnalysis/data/unlabeled_images/*.jpg")
    figures = []

    figures.append(plt.figure())
    myBa.createCdHist(bins=[90, 72])
    plt.title("Spatial defect centroid distribution(bins = [90X72])")

    figures.append(plt.figure())
    myBa.createCdHist(bins=[50, 40])
    plt.title("Spatial defect centroid distribution(bins = [50X40])")

    figures.append(plt.figure())
    myBa.createPixelHist()

    figures.append(plt.figure())
    myBa.createKdePixelHist()

    figures.append(plt.figure())
    myBa.createDefPixNumHist()

    figures.append(plt.figure())
    myBa.createDefNumImgHist()

    figures.append(plt.figure())
    myBa.createPixAvgSizeHist()

    if len(sys.argv) > 1:
        size = len(figures)
        names = [str(i) for i in range(size)]
        filenames = [("./results/" + name + ".png") for name in names]

        for i in range(size):
            figures[i].savefig(filenames[i])

        utils.saveAsExcel(sys.argv[1], filenames)

    else:
        plt.show()

if __name__ == "__main__":
        main()
