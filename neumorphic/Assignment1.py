import argparse
import numpy as np
import matplotlib.pyplot as plt

class Stimulai:
    def __init__(self, sSize):
        self.data = None
        self.sSize = sSize
    def generateData(self):
        self.data = np.random.random(self.sSize)
        
class response:
    def __init__(self, rSize, prob, samp):
        self.data = None
        self.rSize = rSize
        self.prob = prob
        self.samp = samp
    def generateData(self):
        self.data = np.random.binomial(n=1, p=self.prob, size=self.rSize)

def simulate(sSIZE, rSIZE, prob, sampW):
    sti = Stimulai(sSIZE)
    sti.generateData()
    res = response(rSIZE, prob, sampW)
    res.generateData()
    spikes = np.where(res.data == 1)
    samples = []
    for i in range(len(spikes[0])):
        if spikes[0][i]+sampW < sSIZE:
            samples.append(sti.data[spikes[0][i]:spikes[0][i]+sampW])
			
    avgSti = np.mean(samples, axis = 0)
    smoothSti = np.convolve(sti.data, avgSti, "same")
    bin = sSIZE//10
    histSti, bins = np.histogram(smoothSti, bins=bin)
    StiGivenR, _ = np.histogram(smoothSti[spikes[0]], bins=bins)
    ResGSti = (StiGivenR/histSti)*prob
    fig, axs = plt.subplots(2, 2)
	
    axs[0, 0].plot(sti.data)
    axs[0, 0].set_title('Stimulas')
    axs[0, 1].plot(res.data)
    axs[0, 1].set_title('Spiked Respose')
    axs[1, 0].plot(avgSti)
    axs[1, 0].set_title('Averaged Samples')
    axs[1, 1].plot(smoothSti)
    axs[1, 1].set_title('Filtered Stimulas')
    plt.show()
	
    plt.bar(bins[:-1], histSti, width=np.diff(bins))
    plt.title('Filtered Stimulas Histogram')
    plt.show()
    plt.bar(bins[:-1], StiGivenR, width=np.diff(bins))
    plt.title('Stimulas Given Respose Histogram')
    plt.show()
    plt.bar(bins[:-1], ResGSti, width=np.diff(bins))
    plt.title('Respose Given Stimulas Histogram')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = '-------------------<Neuromorphic Assignment>-------------------')
    parser.add_argument('-n', type=int, default = 1000, help = '[Stimulas Length]')
    parser.add_argument('-m', type=int, default = 10, help='[Window Size]')
    parser.add_argument('-p', type=float, default = 0.1, help='[Probablity]')
    args = parser.parse_args()
    n = args.n
    m = args.m
    p = args.p
    simulate(n, n, p, m)
