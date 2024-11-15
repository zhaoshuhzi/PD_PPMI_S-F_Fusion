import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def drawScatter(df):
    df_select1 = df.loc[df.Group.isin(['G1']), :]
    df_select2 = df.loc[df.Group.isin(['G2']), :]
    df_select3 = df.loc[df.Group.isin(['G3']), :]
    r1 = df.loc[df.Group.isin(['G1']),:].corr().iat[1,0]
    r2 = df.loc[df.Group.isin(['G2']),:].corr().iat[1,0]
    r3 = df.loc[df.Group.isin(['G3']),:].corr().iat[1,0]
    # Plot
    sns.set_style("white")
    gridobj1 = sns.lmplot(x='TruSc', y='PreSc', hue='Group', data=df_select1)
    gridobj2 = sns.lmplot(x='TruSc', y='PreSc', hue='Group', data=df_select2)
    gridobj3 = sns.lmplot(x='TruSc', y='PreSc', hue='Group', data=df_select3)
    # Decorations
    gridobj1.set(xlim=(8, 43), ylim=(-1.5, 43))
    gridobj2.set(xlim=(8, 43), ylim=(-1.5, 43))
    gridobj3.set(xlim=(8, 43), ylim=(-1.5, 43))
    plt.title("Pearson Correlation G1 "+str(r1)+'  G2  '+str(r2)+'  G3  '+str(r3), fontsize=13)
    plt.show()
    
## read all txt
for ii in range(2,4):
    txt_dir = '/media/lhj/Momery/PD_predictDL/Data/Log/VoReMa/79_2/VoReMa'+str(ii)+'.txt'
    df1 = pd.read_csv(txt_dir,delimiter="\t")
    drawScatter(df1)