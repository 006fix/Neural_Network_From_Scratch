
import Data_Repositories.Source_Data as Source_Pull
import Function_Library.Preprocessing_Functions as Preproc_Func
import Classes.Neural_Network_Creation as NN_Creator
import pandas as pd
import openpyxl

base_data = Source_Pull.get_base_data()
print("base_data is retrieved, proceeding with the code")

print("gathering test dataset")
train_test_split_no = 60000
X_train = base_data.data.values[:train_test_split_no].T
y_train = base_data.target[:train_test_split_no].values.astype(int)
y_train_holder = []
for y in y_train:
    y_train_holder_pre = Preproc_Func.generate_onehot(y, 10)
    y_train_holder.append(y_train_holder_pre)
print("test dataset generation complete")

print("now gathering test dataset")
X_test = base_data.data.values[train_test_split_no:]
y_test = base_data.target[train_test_split_no:].values.astype(int)
y_test_holder = []
for y in y_test:
    y_test_holder_pre = Preproc_Func.generate_onehot(y, 10)
    y_test_holder.append(y_test_holder_pre)
print("test dataset generation complete")
print(X_train.shape, X_test.shape)

trial_NN = NN_Creator.Neural_Network(X_train, X_test, [15,12], 'relu', 10, 20)
trial_NN.initialize_parameters()
holder_val = trial_NN.parameters
key_list = []
val_list = []
for key in holder_val:
    holdval = holder_val[key]
    key_list.append(key)
    val_list.append(holdval)
print(key_list[0])
print(val_list[0])
df_set = [key_list, val_list]

df = pd.DataFrame(df_set)
outpath1 = r"C:\Users\pyeac\Desktop\holder_output\provis_data.xlsx"
df.to_excel(outpath1)
