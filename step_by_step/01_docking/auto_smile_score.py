import os
import glob
import subprocess
import pandas as pd

dir_ls = glob.glob(f'./auto_vina/*')

smi_score_dict = {}


for i in dir_ls:
    new_ls = []
    result = i + '/' + i.split('/')[-1] + '_result.txt'
    smi_file = '/Users/jongseo/Desktop/laidd/laidd_run/DB/Enamine/target/' + os.path.basename(result).replace('_result.txt', '.smi')

    with open (result, 'r') as score_file:
        for i in score_file:
            if i.startswith('   1 '):
                sc = float(i[5:20].lstrip(' ').rstrip(' '))

        with open (smi_file, 'r') as s:
            l = s.readlines()
            for i in l:
                dt = i.split('\t\t')
                _smi = dt[0]
                _name = dt[1].lstrip().rstrip()
                
                new_ls.append(_smi)
                new_ls.append(sc)

                smi_score_dict[_name] = new_ls


df = pd.DataFrame.from_dict(data=smi_score_dict, orient='index')
df.to_csv('smi_score.csv')



# for qt in qt_ls:
#     qt = qt
#     score = qt.replace('_out.pdbqt', '_result.txt')

#     qt_to_smi_cmd = ['obabel', '-ipdbqt', qt, '-osmi']
    
#     try:
#         result_smi = subprocess.run(qt_to_smi_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         smi = result_smi.stdout.decode('UTF-8').split('\t')[0]
        

#         with open (score, 'r') as score_file:
#             for i in score_file:
#                 if i.startswith('   1 '):
#                     sc = float(i[5:20].lstrip(' ').rstrip(' '))

#             smi_score_dict[smi] = sc

#     except: 
#         pass

    
# df = pd.DataFrame.from_dict(data=smi_score_dict, orient='index')

# df.to_csv('smi_score.csv')