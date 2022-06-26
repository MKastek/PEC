import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


class PEC:

    def __init__(self,PEC_file,wavelength,transitions,Ne_samples_amount, Te_samples_amount):

        self.Ne_samples_amount = Ne_samples_amount
        self.Te_samples_amount = Te_samples_amount

        self.ISEL, self.num_of_Ne_axes, self.num_of_Te_axes, self.num_of_lines_to_read_with_axes, self.sum_of_axes, self.Ne, self.Te = self.read_first_line_of_file(PEC_file)
        self.matrix_Te_Ne = np.empty([self.num_of_Te_axes, self.num_of_Ne_axes],dtype='float')
        self.num_of_lines_to_read_with_const_Ne = int(np.ceil(self.num_of_Ne_axes / 8))

        if self.num_of_lines_to_read_with_const_Ne == 1.0:
            self.num_of_lines_to_read_with_const_Ne += 1

        start = self.num_of_lines_to_read_with_axes + 2
        self.move = self.num_of_Ne_axes * self.num_of_lines_to_read_with_const_Ne + self.num_of_lines_to_read_with_axes + 1
        end = self.num_of_lines_to_read_with_const_Ne*self.num_of_Ne_axes+start

        self.xnew, self.ynew = np.logspace(np.log10(self.Ne[0]),np.log10(self.Ne[-1]), num=Ne_samples_amount), np.logspace(np.log10(self.Te[0]),np.log10(self.Te[-1]), num=Te_samples_amount)
        self.PEC_arr = [self.read_PEC(PEC_file, start+self.move*i, end+self.move*i, self.num_of_lines_to_read_with_const_Ne, self.matrix_Te_Ne.copy()) for i in range(self.ISEL)]

        self.transition_df = self.get_transition_df(PEC_file,1)

        self.indexes = []

        for transition in transitions:
            self.indexes.append(self.transition_df.index[(self.transition_df['wavelength'] == wavelength) & (self.transition_df['type'] == transition)].tolist()[0])
        self.PEC_selected_arr = [self.PEC_arr[index] for index in self.indexes]
        self.plot(self.indexes)

    def read_first_line_of_file(self,filepath):
        with open(filepath) as file:
            first_line = file.readline().strip().split()
            second_line = file.readline().strip().split()
            num_of_Ne_axes, num_of_Te_axes = int(second_line[2]), int(second_line[3])
            sum_of_axes = num_of_Te_axes + num_of_Ne_axes
            num_of_lines_to_read_with_axes = int(np.ceil(sum_of_axes / 8))

            ISEL = int(first_line[0])

            data = []
            iter = num_of_lines_to_read_with_axes
            while iter > 0:
                for item in file.readline().strip().split():
                    data.append(item)
                iter -= 1

            Ne = [float(item) for item in data[:num_of_Ne_axes]]
            Te = [float(item) for item in data[num_of_Ne_axes:]]
        return ISEL, num_of_Ne_axes, num_of_Te_axes, num_of_lines_to_read_with_axes, sum_of_axes, Ne, Te

    def read_PEC(self, filepath, start_line, stop_line, num_of_lines_to_read_with_const_Ne, matrix_Te_Ne):

        with open(filepath) as file:
            data = np.array([item.split() for item in file.read().strip().splitlines()[start_line:stop_line]],
                            dtype=object)
            data = np.split(data, 1)

            iter = 0
            for i in range(0, int(len(data[0]))-1, num_of_lines_to_read_with_const_Ne):
                line_data = np.array([])
                for j in range(num_of_lines_to_read_with_const_Ne):
                    line_data = np.concatenate((line_data, data[0][i + j]))
                matrix_Te_Ne[:, iter] = line_data
                iter += 1

        f = interpolate.interp2d(self.Ne, self.Te, matrix_Te_Ne.astype(np.float64), kind='linear')

        return f(self.xnew, self.ynew)

    def get_transition_df(self,filepath,start_line):
        wavelength = []
        type = []
        for i in range(self.ISEL):
            with open(filepath) as file:
                first_line = file.readlines()[start_line+i*self.move].strip().split()
                wavelength.append(float(first_line[0]))
                type.append(first_line[9])
        return pd.DataFrame({'wavelength':wavelength,'type':type})

    def plot(self,indexes):

        for index in indexes:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(self.xnew, self.ynew)

            ax.plot_wireframe(X, Y, self.PEC_arr[index], rstride=1)
            plt.title(f"PEC - {self.transition_df['type'][index]}")
            ax.set_xlabel('Ne [1/cm3]')
            ax.set_ylabel('Te [eV]')
            ax.set_zlabel('PEC')
            plt.show()

    def analyse_pec(self):
        interpolated_pec_df = np.zeros([len(self.indexes),self.Ne_samples_amount, self.Te_samples_amount, 3])
        for transition in range(len(self.indexes)):
            for i in range(self.Ne_samples_amount):
                interpolated_pec_df[transition,i,:,:] = np.array([[self.xnew[i]]*self.Ne_samples_amount,self.ynew,self.PEC_selected_arr[transition][:,i]]).T
        return interpolated_pec_df.shape,{index+1: self.transition_df['type'][index] for index in self.indexes}


if __name__ == "__main__":
    C = PEC(PEC_file='pec_C.dat',wavelength=33.7,transitions=['EXCIT','RECOM'],Ne_samples_amount=230,Te_samples_amount=230)
    df, list = C.analyse_pec()
    print(list)
    # out put
    # main
    # FA - Fractional Abundance
    #B = PEC(PEC_file='pec_B.dat', wavelength=194.3, transitions=['EXCIT', 'RECOM'], Ne_samples_amount=50,Te_samples_amount=50)
    #N = PEC(PEC_file='pec_N.dat', wavelength=133.8, transitions=['EXCIT', 'RECOM','CHEXC'], Ne_samples_amount=50,Te_samples_amount=50)
    #O = PEC(PEC_file='pec_O.dat', wavelength= 102.4, transitions=['EXCIT', 'RECOM'], Ne_samples_amount=50, Te_samples_amount=50)


