import pandas as pd
import random 
import json

def simple_same_name(name1, name2):
    # 定义需要移除的分隔符
    separators = [',', '-', '_', ' ', '/', '.', ':', ';', '(', ')', '[', ']', '{', '}']
    # 定义需要移除的连词
    connectors = ['and', '&']

    # 处理第一个名称
    name1_processed = name1.lower()
    for sep in separators:
        name1_processed = name1_processed.replace(sep, '')

    # 处理第二个名称
    name2_processed = name2.lower()
    for sep in separators:
        name2_processed = name2_processed.replace(sep, '')

    # 移除连词
    for connector in connectors:
        name1_processed = name1_processed.replace(connector, '')
        name2_processed = name2_processed.replace(connector, '')

    # 比较处理后的名称是否相等
    return name1_processed == name2_processed

# fewshot_num = {'Ganglioneuroblastoma, intermixed': 15, 
#                     'Differentiating neuroblastoma': 15,
#                     'Poorly differentiated neuroblastoma':15}
# fewshot_num= {'Classic medulloblastoma':15, 
#                     'Desmoplastic nodular medulloblastoma': 15,
#                     'Large Cell/Anaplastic medulloblastoma': 5}
# fewshot_num= {'Pure fetal hepatoblastoma with low mitotic activity':15,
#                       'Epithelial mixed fetal and embryonal hepatoblastoma':15,
#                       'Epithelial macrotrabecular pattern of hepatoblastoma': 5, 
#                       'Mixed epithelial and mesenchymal hepatoblastoma': 15,
#                     }
# fewshot_num= {'Mixed blastemal and stromal nephroblastoma': 15,
#                             'Nephroblastoma, blastemal type': 5,
#                             'Mixed blastemal, epithelial and stromal nephroblastoma': 15,
#                             'Mixed blastemal and epithelial nephroblastoma': 15,
#                             'Nephroblastoma, stromal type': 15
#                             }

fewshot_num= {'Normal': 15,
            'Tumor': 15,
            }
# fewshot_num = {'CC':15, 
#                'EC': 15, 
#                'HGSC':15, 
#                'LGSC': 15,
#                'MC':15
#                }

csv_root = '/ailab/user/zhouxiao/WSI_proc_code/MI-Zero-main/xinhua_src/data_csvs/final_csvs/'
# csv_root = '/ailab/user/zhouxiao/WSI_proc_code/MI-Zero-main/src/data_csvs/'
shenmu_csv_path = csv_root + 'shenmu_all_label_for_subtyping_V5_filter.csv'
suimu_csv_path = csv_root + 'suimu_all_label_for_subtyping_V3.csv'
ganmu_csv_path = csv_root + 'ganmu_all_label_for_subtyping_filter.csv'
# shenzangmu_csv_path = csv_root + 'shenzangmu_all_label_for_subtyping_V2_filter.csv'
shenzangmu_csv_path = csv_root + 'shenzangmu_all_label_for_detection_V2_filter_rebuttal.csv'

# ubc_csv_path = csv_root + 'other_ubc_test.csv'

proc_tumor = 'shenzangmu'

proc_table = pd.read_csv(eval(proc_tumor + '_csv_path'))

proc_data= dict()
for index, each_row in proc_table.iterrows():
    sid = each_row['slide_id']
    subtype = each_row['Diagnosis']  #'Diagnosis' for xinhua, 'OncoTreeCode' for UBC
    
    if subtype not in proc_data:
        proc_data[subtype] = [sid]
    else:
        proc_data[subtype].append(sid)

# with open('./annotation_as_train.json') as f:
#     train_dict = json.load(f)


final_data = dict()
final_data['train_IDs'] = dict()
final_data['test_IDs'] = dict()
for idx, (k,v) in enumerate(proc_data.items()):
    random.seed(idx)
    random.shuffle(v)
    
    v = [str(item) for item in v]
    
    train_data = v[0:fewshot_num[k]] 
    test_data = v[fewshot_num[k]:]
    # train_data = [item.split('.tiff')[0] for item in list(train_dict[proc_tumor][k].keys())]
    # test_data = [item for item in v if item not in train_data]
    
    if k not in final_data['train_IDs']:
        final_data['train_IDs'][k] = train_data
    if k not in final_data['test_IDs']:
        final_data['test_IDs'][k] = test_data
        
json_str = json.dumps(final_data, indent=2)with open('./xinhua_'+ proc_tumor + '_det_dataset_division.json', 'w') as json_file:
    json_file.write(json_str)

test = 1