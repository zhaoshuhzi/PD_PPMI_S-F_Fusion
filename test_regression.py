# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
# y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# torch can only train on Variable, so convert them to Variable
# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('ignore')
# Import Data
csv_dir = '/media/lhj/Momery/PD_predictDL/Data/1.csv'
df = pd.read_csv(csv_dir)
df_select = df.loc[df.Group.isin(['G1', 'G2','G3']), :]
r1 = df.loc[df.Group.isin(['G1']),:].corr().iat[1,0]
r2 = df.loc[df.Group.isin(['G2']),:].corr().iat[1,0]
r3 = df.loc[df.Group.isin(['G3']),:].corr().iat[1,0]
# Plot
sns.set_style("white")
gridobj = sns.lmplot(x='TruSc', y='PreSc', hue='Group', data=df_select)
# Decorations
gridobj.set(xlim=(-0.5, 25), ylim=(-1.5, 25))
plt.title("Pearson Correlation G1 "+str(r1)+'  G2  '+str(r2)+'  G3  '+str(r3), fontsize=13)
plt.show()