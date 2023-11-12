import torch
import numpy as np
import rdkit
import rdkit.Chem as Chem
import numpy as np
import rdkit.Chem.AllChem as AllChem
import torch
import torch_geometric
from torch_geometric.data import Data
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn.functional import gelu
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import glob
import os
import time


# 네트워크 구성
class myGCN(torch.nn.Module):
    def __init__(self, in_channel=35, hidden_layer_size=70):
        super().__init__()
        self.conv1 = GCNConv(in_channel, hidden_layer_size) 
        # 가장 기본적인 graph convolution model, https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
        self.bn1 = BatchNorm(hidden_layer_size)
        self.conv2 = GCNConv(hidden_layer_size, hidden_layer_size)
        self.bn2 = BatchNorm(hidden_layer_size)
        self.conv3 = GCNConv(hidden_layer_size, hidden_layer_size)
        self.bn3 = BatchNorm(hidden_layer_size)

        self.lin1 = Linear(hidden_layer_size, int(hidden_layer_size/2))
        #self.lin1_bn = BatchNorm(int(hidden_layer_size/2))
        self.lin2 = Linear(int(hidden_layer_size/2), 1)

    def forward(self, data):
        x, edge_index, batch, edge_weight = data.x, data.edge_index, data.batch, data.edge_weight

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.gelu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.gelu(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.gelu(x)

        # READOUT
        x = global_mean_pool(x, batch) # 전체의 node feature의 평균 값을 취한다. # [batch_size, hidden_channels]
        x = self.lin1(x) # 70 dim -> 35-dim
        #x = self.lin1_bn(x)
        x = F.elu(x)

        x = self.lin2(x) # 35 dim -> 1-dim

        return x
    

# 분자 구조를 그래프로 표현하기
def convert_mol_to_graph(mol, use_pos = False):

        #mol2 = Chem.AddHs(mol)
        mol2 = Chem.RemoveHs(mol)

        n_bonds = len(mol2.GetBonds()) # 분자의 공유 결합 개수
        n_atoms = len(mol2.GetAtoms()) # 분자의 원자 개수

        node_attr = []
        #### node 속성 계산 시작 ####
        # RDKit으로 계산할 수 있는 Atom의 속성은 아래 링크에서 확인할 수 있다.
        # https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Atom
        #
        #.                0.     1.     2.     3.     4.     5.     6.     7.     8.        9.    10.
        #                'H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl' 'Br'    'I'
        valid_atoms = {'H': 0, 'B':1, 'C':2, 'N':3, 'O':4, 'F':5, 'P':6, 'S':7, 'Cl':8, 'Br':9, 'I':10}

        for atm_id in range(n_atoms):
                # Select an atom.
                atm = mol2.GetAtomWithIdx(atm_id)

                # Atom symbol check (9-dim)
                sym = atm.GetSymbol()
                atm_one_hot = [0] * len(valid_atoms) # 0이 9개 들어있는 리스트를 만든다.
                idx = valid_atoms[sym] # sym에 해당하는 원소 기호가 몇 번째에 있는지?
                atm_one_hot[idx] = 1     # 해당되는 원소의 위치만 1로 바꾼다.

                # Check hybridization (7-dim)
                hybrid = atm.GetHybridization()
                hybrid_one_hot = [0] * 7 # [0, 0, 0, 0, 0, 0, 0]
                if hybrid == Chem.HybridizationType.SP3:
                    hybrid_one_hot[0] = 1
                elif hybrid == Chem.HybridizationType.SP2:
                    hybrid_one_hot[1] = 1
                elif hybrid == Chem.HybridizationType.SP:
                    hybrid_one_hot[2] = 1
                elif hybrid == Chem.HybridizationType.S:
                    hybrid_one_hot[3] = 1
                elif hybrid == Chem.HybridizationType.SP3D:
                    hybrid_one_hot[4] = 1
                elif hybrid == Chem.HybridizationType.SP3D2:
                    hybrid_one_hot[5] = 1
                else: # hybridization이 제대로 정의되지 않은 나머지의 모든 경우.
                    hybrid_one_hot[6] = 1

                # aromatic 인지 아닌지?    (True/False)
                if atm.GetIsAromatic():
                    arom = 1
                else:
                    arom = 0

                # ring 안에 존재하는지 아닌지? (True/False)
                if atm.IsInRing():
                    ring_flag = 1
                else:
                    ring_flag = 0

                # Degree (공유 결합의 개수)    (6-dim, one-hot)
                # 0, 1, 2, 3, 4, >=5
                degree_one_hot = [0, 0, 0, 0, 0, 0]
                degree = atm.GetTotalDegree()
                if degree >= 5: # 5개 이상의 공유 결합을 가지는 원자.
                    degree_one_hot[5]=1
                else:
                    degree_one_hot[degree]=1

                # Number of hydrogens (5-dim, one-hot)
                # 결합되어 있는 수소의 개수.
                # 0, 1, 2, 3, >=4
                num_h = atm.GetTotalNumHs()
                hydrogen_one_hot = [0, 0, 0, 0, 0]
                if num_h >= 4:
                    hydrogen_one_hot[4] = 1
                else:
                    hydrogen_one_hot[num_h] = 1

                # Chirality (4-dim, one-hot)
                chiral = atm.GetChiralTag()
                if chiral == Chem.rdchem.ChiralType.CHI_OTHER:
                    chiral_one_hot = [1, 0, 0, 0]
                # Counter-clock-wise (반시계)
                elif chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
                    chiral_one_hot = [0, 1, 0, 0]
                # Clockwise (시계방향)
                elif chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
                    chiral_one_hot = [0, 0, 1, 0]
                # Chirality 정의되지 않음.
                elif chiral == Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
                    chiral_one_hot = [0, 0, 0, 1]

                # 원자 특성 계산 [원자 symbol one-hot, 공유 결합 개수, 전체 valence의 개수 (explicit + implicit), is an atom aromatic (True/False)? ]
                # 더 추가 가능!
                # in total 25-dim.
                # 11-dim, 7-dim, 6-dim, 5-dim, 4-dim, 4-dim = 35-dim
                attr = atm_one_hot + \
                                hybrid_one_hot + \
                                degree_one_hot + \
                                hydrogen_one_hot + \
                                chiral_one_hot + \
                                [arom, ring_flag, atm.GetFormalCharge(), atm.GetNumRadicalElectrons()]

                #print(atm_id, attr)
                node_attr.append(attr)

        #### node 속성 계산 완료 ####

        edge_index = []
        edge_attr = []
        edge_weight = []
        for edge_idx in range(n_bonds): # 전체 공유 결합에 대해서 loop을 돌린다.

                bond = mol2.GetBondWithIdx(edge_idx) # 각 공유 결합에 대해서 시작 atom과 끝 atom의 인덱스를 확인.
                edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]) # undirected graph를 만들기 위해서 순서를 바꿔서 edge를 2번 넣어준다.

                # BondType (4-dimensional one-hot)
                btype = bond.GetBondType() # 공유 결합의 종류.
                if btype == Chem.rdchem.BondType.SINGLE:
                        bond_one_hot = [1, 0, 0, 0]
                        edge_weight.extend([1.0, 1.0])
                elif btype == Chem.rdchem.BondType.AROMATIC:
                        bond_one_hot = [0, 1, 0, 0]
                        edge_weight.extend([1.5, 1.5])
                elif btype == Chem.rdchem.BondType.DOUBLE:
                        bond_one_hot = [0, 0, 1, 0]
                        edge_weight.extend([2.0, 2.0])
                elif btype == Chem.rdchem.BondType.TRIPLE:
                        bond_one_hot = [0, 0, 0, 1]
                        edge_weight.extend([3.0, 3.0])

                # BondStereo (6-dimensional one-hot)
                stype = bond.GetStereo()
                if stype == Chem.rdchem.BondStereo.STEREOANY:
                    stereo_one_hot = [1, 0, 0, 0, 0, 0]
                elif stype == Chem.rdchem.BondStereo.STEREOCIS:
                    stereo_one_hot = [0, 1, 0, 0, 0, 0]
                elif stype == Chem.rdchem.BondStereo.STEREOE:
                    stereo_one_hot = [0, 0, 1, 0, 0, 0]
                elif stype == Chem.rdchem.BondStereo.STEREONONE:
                    stereo_one_hot = [0, 0, 0, 1, 0, 0]
                elif stype == Chem.rdchem.BondStereo.STEREOTRANS:
                    stereo_one_hot = [0, 0, 0, 0, 1, 0]
                elif stype == Chem.rdchem.BondStereo.STEREOZ:
                    stereo_one_hot = [0, 0, 0, 0, 0, 1]

                # Is this bond included in a ring?
                if bond.IsInRing():
                    ring_bond = 1
                else:
                    ring_bond = 0

                # Is this bond a conjugated bond?
                if bond.GetIsConjugated():
                    conjugate = 1
                else:
                    conjugate = 0

                # In total 12-dimensional edge attribute
                # bond-type (4-dim), bondstereo (6-dim), (ring, conjugate)
                # Can you image more?
                attr = bond_one_hot + stereo_one_hot + [ring_bond, conjugate] # 12 차원의 공유 결합 속성.

                # 분자는 undirected graph이므로 edge가 두 번 정의된다.
                # 그러므로 동일한 attribute를 두 번 넣어주어야 한다.
                edge_attr.append(attr)
                edge_attr.append(attr)
        #### edge 속성 계산 완료 ####


        # PyTorch Tensor로 변환.
        edge_attr = torch.tensor(edge_attr, dtype = torch.float)
        node_attr = torch.tensor(node_attr, dtype = torch.float)
        edge_index = torch.tensor(edge_index, dtype = torch.long)
        edge_index = edge_index.t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype = torch.float)


        # 만일 3D 좌표 정보를 사용한다면
        if use_pos:
                val = AllChem.EmbedMolecule(mol2)
                if val !=0:
                    print(f"Error while generating 3D: {Chem.MolToSmiles(mol)}")
                    return None

                pos_list = [] # this is optional
                for atm_id in range(n_atoms):
                    # Get Atomic Position.
                    atm_pos = mol2.GetConformer(0).GetAtomPosition(atm_id)
                    crd = [atm_pos.x, atm_pos.y, atm_pos.z]
                    pos_list.append(crd)

                pos = torch.tensor(pos_list, dtype=torch.float)
        else:
            pos = None
        #print(edge_attr.shape)
        #print(node_attr.shape)
        #print(edge_index.shape)

        return edge_index, node_attr, edge_attr, pos, edge_weight



# 원하지 않는 atom 이 포함되어 있는지 확인
def check_atoms(mol):
    """
    This function checks whether all atoms are valid
    """
    valid_atoms = ('H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I')
    flag = True
    for atm in mol.GetAtoms(): # 분자 안에 존재하는 모든 atom에 대해서 loop을 수행.
        if atm.GetSymbol() not in valid_atoms:
            flag = False
            break
    return flag


# 데이터를 텐서 형태로 전환
def _data_to_tensors (inp):
    
    result = convert_mol_to_graph(inp)
    
    if result is None:
        return None

    else:
        edge_index, node_attr, edge_attr, pos, edge_weight = result
        return edge_index, node_attr, edge_attr, pos, edge_weight


# custom dataset 구성
class MyData(Data):
    def __init__ (self, **kwargs):
        super(MyData, self).__init__(**kwargs)

    def __cat_dim__(self, key, value, *args):
        if key == 'smi':
            return None

        else:
            return super(MyData, self).__cat_dim__(key, value)


# 데이터 전처리 함수 (학습데이터와 SMILE 을 함께 가지고 다닐 수 있도록 구성)
def data_preproc (smi_list):
    processed_data_list = []

    for smi in tqdm(smi_list):
        smi = smi.strip()
        mol = Chem.MolFromSmiles(smi)

        if not mol or not check_atoms(mol):
            continue

        result = _data_to_tensors(mol)

        if result is None:
            continue

        edge_index, node_attr, edge_attr, pos, edge_weight = result

        data = MyData(
            x = node_attr,
            edge_index = edge_index,
            edge_attr = edge_attr,
            y = torch.tensor([0.0]), # dummy
            smi = smi
        )

        processed_data_list.append(data)

    return processed_data_list


# 추론 및 결과를 저장
def inference (model, loader, device, output_file):
    model.eval()

    with torch.no_grad():
        with open (output_file, 'w') as writer:
            writer.write(f'smi,score\n')
            for data in loader:
                data = data.to(device)
                out = model(data)
                scores = out.view(-1).cpu().numpy()

                for smi, score in zip (data.smi, scores):
                    writer.write(f'{smi},{score}\n')


if __name__ == '__main__':
    # device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('mps') (<< macOS 용)

    # 학습했던 모델 로드
    model = torch.load('./example/Best_GCN_model_v1.pt')

    # SMILES 가 들어있는 파일 로딩
    smi_files = glob.glob('./example/mcule-instock_11.smi')

    # 추론/평가모드로 모델을 전환
    model.eval()
    model.to(device)
    start_init = time.time()

    # 각 SMILE 파일에 대해 추론을 수행하고, 진행과정을 출력 
    for i in smi_files:
        output_file = os.path.basename(i).replace('.smi', '.csv')
        print (os.path.basename(i), '---- processing')
        with open (i, 'r') as f:
            smi_list = f.readlines()
            print (smi_list)
            dataset = data_preproc(smi_list)
            loader = DataLoader(dataset, batch_size=1024, num_workers=4, shuffle=False, pin_memory=True)
            inference (model, loader, device, output_file)

    print ('Total elapsed time: ', time.time() - start_init, 'sec')
