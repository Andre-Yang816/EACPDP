import matplotlib.pyplot as plt
import pandas as pd
import os
plt.rc('font', family='Times New Roman')
class BoxPlot:
    def __init__(self, phome,cl_name):
        self.color_dict = {1: 'red', 2: "green", 3: 'blue', 4: 'yellow', 5: 'purple', 6: 'orange', 7: 'pink', 8: 'gray',
                           9: 'gray', 10: 'gray', 11: 'gray', 12: 'gray', 13: 'gray', 14: 'gray', 15: 'gray',
                           16: 'gray', 17: 'gray', 18: 'gray', 19: 'gray', 20: 'gray', 21: 'gray', 22: 'gray',
                           23: 'gray',
                           24: 'gray', 25: 'gray', 26: 'gray', 27: 'gray', 28: 'gray', 29: 'gray', 30: 'gray',
                           31: 'gray', 32: 'gray', 33: 'gray', 34: 'gray', 35: 'gray'}

        # 9: 'olive', 10: 'plum', 11: 'c', 12: 'aqua',13: 'cyan', 14: 'teal', 15: 'skyblue', 16: 'darkblue',
        # 17: 'deepskyblue', 18: 'indigo', 19: 'darkorange', 20: 'lightcoral'}
        self.home = phome
        self.parse()
    # After passing the r program will get a txt version of which group each algorithm belongs to,
    # here the txt is converted to a csv file
    def parseTxtToCSV(self, header, txt_path):
        # The function is to convert the grouped txt to a color csv and return the path to generate the color csv
        # Note that the first line of this txt is invalid data, the data starts from the second line,
        # and each line is a key-value pair
        if not os.path.exists(txt_path):
            print("txt file：{0} not exit!".format(txt_path))
            exit()
        method = []
        rank = []
        lines = open(txt_path, "r", encoding="utf-8").readlines()
        for line in lines:
            l = line.replace("\n", "").replace('"', "").split(" ")
            if l[0] != 'x':
                method.append(l[0])
                rank.append(int(l[1]))
        if len(header) != len(method) or len(header) != len(rank):
            print("lenth not equal ！")
            exit()
        header_rank = []
        for head in header:
            if head in method:
                index = method.index(head)
                header_rank.append(rank[index])
        data = []
        data.append(header)
        data.append(header_rank)
        return data
        pass

    def parse(self):
        results_path = self.home
        for file in os.listdir(results_path):
            header = ['None1', 'BF1', 'PF1', 'KF1', 'DFAC1', 'TCA1', 'BDA1', 'JDA1', 'JPDA1','TNB1',
                    'None2', 'BF2', 'PF2', 'KF2', 'DFAC2', 'TCA2', 'BDA2', 'JDA2', 'JPDA2','TNB2',
                    'None3', 'BF3', 'PF3', 'KF3', 'DFAC3', 'TCA3', 'BDA3', 'JDA3', 'JPDA3','TNB3',
                    'None4', 'BF4', 'PF4', 'KF4', 'DFAC4', 'TCA4', 'BDA4', 'JDA4', 'JPDA4','TNB4']
            txt_file_path = self.home + '{0}'.format(file)
            color_data = self.parseTxtToCSV(header, txt_file_path)
            csv_name = file.strip('.txt')
            csv_path = '../target/{0}/{1}'.format(cl_name,csv_name)
            all_data = pd.read_csv(csv_path)
            all_data = all_data.iloc[1:, 1:].values
            colors_nums = color_data[1]
            print("colors_nums from color_csv:", colors_nums)
            fig, ax = plt.subplots(
                figsize=(12, 2))
            ax.tick_params(direction='in')
            figure = ax.boxplot(all_data,
                                notch=False,  # notch shape
                                sym='r+',  # blue squares for outliers
                                vert=True,  # vertical box aligmnent
                                meanline=True,
                                showmeans=True,
                                patch_artist=False,
                                showfliers=False
                                )
            colors = [self.color_dict[int(i)] for i in colors_nums]
            for i in range(0, len(colors)):
                k = figure['boxes'][i]
                k.set(color=colors[i])
                k = figure['means'][i]
                k.set(color=colors[i], linewidth=0)
                k = figure['medians'][i]
                k.set(color=colors[i], linewidth=2)
                k = figure['whiskers'][2 * i:2 * i + 2]
                for w in k:
                    w.set(color=colors[i], linestyle='--')
                k = figure['caps'][2 * i:2 * i + 2]
                for w in k:
                    w.set(color=colors[i])
            plt.xlim((0, 40.5))
            lenheader = len(header) + 1
            plt.xticks([y for y in range(1, lenheader)], header, rotation=45, weight='heavy', fontsize=7.5)
            plt.yticks(fontsize=10)
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.axvline(10.5, color='black', linestyle=':')
            plt.axvline(20.5, color='black', linestyle=':')
            plt.axvline(30.5, color='black', linestyle=':')
            if csv_name.strip('.csv') == 'IFA' or csv_name.strip('.csv') == 'Popt':
                plt.ylabel("{0}".format(csv_name.strip('.csv')), fontsize=10)
            else:
                plt.ylabel("{0}@20%".format(csv_name.strip('.csv')), fontsize=10)
            plt.title("                       Label/LOC                   "  
                      "                         Prob/LOC                   "
                      "                           CBS+                     "
                      "                            Prob"

                      , fontsize=11, loc='left')
            if not os.path.exists('../pictures/{0}/'.format(cl_name)):
                os.makedirs('../pictures/{0}/'.format(cl_name))
            output_file_path = '../pictures/{0}/{1}.pdf'.format(cl_name,csv_name.strip('.csv'))
            foo_fig = plt.gcf()
            foo_fig.savefig(output_file_path, format='pdf', dpi=1000, bbox_inches='tight')
            plt.clf()
            plt.close()


if __name__ == '__main__':
    classifiers = ['LR', 'NB', 'KNN', 'RF', 'DT', 'MLP']
    for cl_name in classifiers:
        BoxPlot(r".../output/{0}/".format(cl_name),cl_name)