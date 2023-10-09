import matplotlib.pyplot as plt
from bb_analyzer import BbAnalyzer
import sys, utils

#runtima analysis zapijne između 0 i 1 te 2 i 3

from tqdm import tqdm

def main():
    if not sys.argv[0]:
        print("No directory specified!")
        exit(-1)

    myBa = BbAnalyzer("/home/antonio123/workspace/Github_projects/StatisticalImageAnalysis/data/ROI_images_0.5/*.jpg", "/home/antonio123/workspace/Github_projects/StatisticalImageAnalysis/data/all_images/*.png", "/home/antonio123/workspace/Github_projects/StatisticalImageAnalysis/data/all_images_anns.json")
    figures = []
    """progress_bar = tqdm(total=100)
    figures.append(plt.figure())
    print("Creating weighted centroid distribution")
    myBa.createWeightCdhist(bins=[200, 160])"""
    progress_bar = tqdm(total=100)

    figures.append(plt.figure())
    print("Creating spatial defect centroid distribution(bins = [120X96])")
    myBa.createCdHist(bins=[120, 96])
    plt.title("Spatial defect centroid distribution(bins = [120X96])")
    progress_bar.update(10)

    progress_bar.update(10)
    figures.append(plt.figure())
    myBa.createAccDefPixDist()

    progress_bar.update(10)
    figures.append(plt.figure())
    myBa.createPixelHist()

    progress_bar.update(10)
    figures.append(plt.figure())
    myBa.createKdePixelHist()

    progress_bar.update(10)
    figures.append(plt.figure())
    myBa.createDefPixNumHist()

    progress_bar.update(10)
    figures.append(plt.figure())
    myBa.createDefNumImgHist()

    progress_bar.update(10)
    figures.append(plt.figure())
    myBa.createPixDistSizeHist()

    progress_bar.update(10)
    figures.append(plt.figure())
    myBa.showMeasuers()
    progress_bar.update(20)

    if len(sys.argv) > 1:
        size = len(figures)
        names = [str(i) for i in range(size)]
        filenames = [("./results/" + name + ".png") for name in names]

        for i in range(size):
            figures[i].savefig(filenames[i])
        if sys.argv[1] == "original":
            utils.savingInDatasetStatstistics(1, filenames)
        else: utils.saveAsExcel(sys.argv[1], filenames)
    else:
        plt.show()

if __name__ == "__main__":
        main()
