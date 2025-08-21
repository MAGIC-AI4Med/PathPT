import h5py
import json

templates = ['CLASSNAME.',
            'a photomicrograph showing CLASSNAME.',
            'a photomicrograph of CLASSNAME.',
            'an image of CLASSNAME.',
            'an image showing CLASSNAME.',
            'an example of CLASSNAME.',
            'CLASSNAME is shown.',
            'this is CLASSNAME.',
            'there is CLASSNAME.',
            'a histopathological image showing CLASSNAME.',
            'a histopathological image of CLASSNAME.',
            'a histopathological photograph of CLASSNAME.',
            'a histopathological photograph showing CLASSNAME.',
            'shows CLASSNAME.',
            'presence of CLASSNAME.',
            'CLASSNAME is present.',
            'an H&E stained image of CLASSNAME.',
            'an H&E stained image showing CLASSNAME.',
            'an H&E image showing CLASSNAME.',
            'an H&E image of CLASSNAME.',
            'CLASSNAME, H&E stain.',
            'CLASSNAME, H&E.'
            ]


brca_names = {'Invasive Ductal Carcinoma': ['breast invasive ductal carcinoma',
                            'invasive ductal carcinoma of the breast',
                           ],
                    'Invasive Lobular Carcinoma': ['breast invasive lobular carcinoma',
                            'invasive lobular carcinoma of the breast',
                            ],
                    'Normal': ['normal breast tissue',
                               'breast normal tissue',
                               'breast non-cancerous tissue'
                               ]

                    }
nsclc_names = {'Lung Adenocarcinoma': ['lung adenocarcinoma',
                             'adenocarcinoma of the lung',
                            #  'LUAD',
                             ],
                    'Lung Squamous Cell Carcinoma': ['lung squamous cell carcinoma',
                             'squamous cell carcinoma of the lung',
                            #  'LUSC'
                             ],
                    'Normal': ['normal lung tissue',
                               'lung normal tissue',
                               'lung non-cancerous tissue'
                               ]
                    }


brain_names = {'Glioblastoma': ['brain glioblastoma',
                            'glioblastoma of the brain',
                            ],
                'Astrocytoma': ['brain astrocytoma',
                                'astrocytoma of the brain',
                                    ],
                'Oligodendroglioma':['brain oligodendroglioma',
                                    'oligodendroglioma of the brain',
                ],
                'Normal': ['normal brain tissue',
                            'brain normal tissue',
                            'brain non-cancerous tissue',
                            ]
                }


camelyon_names = {'Tumor': ['tumor tissue',
                                  'tumor epithelial tissue',
                                  'cancerous tissue',
                                  'breast tumor tissue',
                                  'breast tumor epithelial tissue',
                                  'breast cancerous tissue',
                                ],
                        'Normal': ['normal tissue',
                                    'non-cancerous tissue'
                                    'normal breast tissue',
                                    'breast non-cancerous tissue',
                                    'benign breast tissue',
                                    'benign tissue'
                                    ]
                }

panda_names = {'Tumor': ['tumor tissue',
                                  'tumor epithelial tissue',
                                  'cancerous tissue',
                                  'prostate tumor tissue',
                                  'prostate tumor epithelial tissue',
                                  'prostate cancerous tissue',
                                ],
                        'Normal': ['normal tissue',
                                    'non-cancerous tissue'
                                    'normal prostate tissue',
                                    'prostate non-cancerous tissue',
                                    'benign prostate tissue',
                                    'benign tissue'
                                    ]
                }

aggc_names = {'Tumor': ['tumor tissue',
                        'tumor epithelial tissue',
                        'cancerous tissue',
                        'prostate tumor tissue',
                        'prostate tumor epithelial tissue',
                        'prostate cancerous tissue',
                                ],
                        'Normal': ['normal tissue',
                                    'non-cancerous tissue'
                                    'normal prostate tissue',
                                    'prostate non-cancerous tissue',
                                    'benign prostate tissue',
                                    'benign tissue'
                                    ]
                }


ubc_names = {'CC': ['ovarian clear cell carcinoma',
                            'clear cell carcinoma of the ovary',
                            ],
                    'EC': ['ovary endometrioid carcinoma',
                             'endometrioid carcinoma of the ovary',
                             ],
                    'HGSC':['high-grade ovary serous carcinoma',
                            'high-grade serous carcinoma of the ovary',
                    ],
                    'LGSC': ['low-grade ovary serous carcinoma',
                            'low-grade serous carcinoma of the ovary'
                               ],
                    'MC': ['ovarian mucinous carcinoma',
                            'mucinous carcinoma of the ovary'
                               ],
                    'Normal': ['normal ovarian tissue',
                               'ovary normal tissue',
                               'ovary non-cancerous tissue'
                               ]
                }

ebrains_names = {'Glioblastoma__IDH-wildtype': ['glioblastoma, IDH-wildtype', 'glioblastoma without IDH mutation', 'glioblastoma with retained IDH', 'glioblastoma, IDH retained'],
                      'Transitional_meningioma':['transitional meningioma',' meningioma, transitional type','meningioma of transitional type','meningioma, transitional'],
                      'Anaplastic_meningioma':['anaplastic meningioma','meningioma, anaplastic type','meningioma of anaplastic type','meningioma, anaplastic'],
                      'Pituitary_adenoma':['pituitary adenoma','adenoma of the pituitary gland','pituitary gland adenoma','pituitary neuroendocrine tumor','neuroendocrine tumor of the pituitary','neuroendocrine tumor of the pituitary gland'],
                      'Oligodendroglioma__IDH-mutant_and_1p_19q_codeleted':['oligodendroglioma, IDH-mutant and 1p/19q codeleted','oligodendroglioma','oligodendroglioma with IDH mutation and 1p/19q codeletion'],
                      'Haemangioma':['hemangioma','haemangioma of the CNS','hemangioma of the CNS','haemangioma of the central nervous system','hemangioma of the central nervous system'],
                      'Ganglioglioma':['gangliocytoma','glioneuronal tumor','circumscribed glioneuronal tumor'],
                      'Schwannoma':['schwannoma','Antoni A','Antoni B','neurilemoma'],
                      'Anaplastic_oligodendroglioma__IDH-mutant_and_1p_19q_codeleted':['anaplastic oligodendroglioma, IDH-mutant and 1p/19q codeleted','anaplastic oligodendroglioma','anaplastic oligodendroglioma with IDH mutation and 1p/19q codeletion'],
                      'Anaplastic_astrocytoma__IDH-wildtype':['anaplastic astrocytoma, IDH-wildtype','anaplastic astrocytoma without IDH mutation','anaplastic astrocytoma, IDH retained','anaplastic astrocytoma with retained IDH'],
                      'Pilocytic_astrocytoma':['pilocytic astrocytoma','juvenile pilocytic astrocytoma','spongioblastoma','pilomyxoid astrocytoma'],
                      'Angiomatous_meningioma':['angiomatous meningioma','meningioma, angiomatous type','meningioma of angiomatous type','meningioma, angiomatous'],
                      'Haemangioblastoma':['haemangioblastoma','capillary hemangioblastoma','lindau tumor','angioblastoma'],
                      'Gliosarcoma':['gliosarcoma','gliosarcoma variant of glioblastoma'],
                      'Adamantinomatous_craniopharyngioma':['adamantinomatous craniopharyngioma','craniopharyngioma'],
                      'Anaplastic_astrocytoma__IDH-mutant':['anaplastic astrocytoma, IDH-mutant','anaplastic astrocytoma with IDH mutation','anaplastic astrocytoma with mutant IDH','anaplastic astrocytoma with mutated IDH'],
                      'Ependymoma':['ependymoma','subependymoma','myxopapillary ependymoma'],
                      'Anaplastic_ependymoma':['anaplastic ependymoma','ependymoma, anaplastic','ependymoma, anaplastic type'],
                      'Glioblastoma__IDH-mutant':['glioblastoma, IDH-mutant','glioblastoma with IDH mutation','glioblastoma with mutant IDH','glioblastoma with mutated IDH'],
                      'Atypical_meningioma':['atypical meningioma','meningioma, atypical type','meningioma of atypical type','meningioma, atypical'],
                      'Metastatic_tumours':['metastatic tumors','metastases to the brain','metastatic tumors to the brain','brain metastases','brain metastatic tumors'],
                      'Meningothelial_meningioma':['meningothelial meningioma','meningioma, meningothelial type','meningioma of meningothelial type','meningioma, meningothelial'],
                      'Langerhans_cell_histiocytosis':['langerhans cell histiocytosis','histiocytosis X','eosinophilic granuloma','Hand-Sch¨uller-Christian disease','Hashimoto-Pritzker disease','Letterer-Siwe disease'],
                      'Diffuse_large_B-cell_lymphoma_of_the_CNS':['diffuse large B-cell lymphoma of the CNS','DLBCL','DLBCL of the CNS','DLBCL of the central nervous system'],
                      'Diffuse_astrocytoma__IDH-mutant':['diffuse astrocytoma, IDH-mutant','diffuse astrocytoma with IDH mutation','diffuse astrocytoma with mutant IDH','diffuse astrocytoma with mutated IDH'],
                      'Secretory_meningioma':['secretory meningioma','meningioma, secretory type','meningioma of secretory type','meningioma, secretory'],
                      'Haemangiopericytoma':['haemangiopericytoma','solitary fibrous tumor','hemangiopericytoma','angioblastic meningioma'],
                      'Fibrous_meningioma':['fibrous meningioma','meningioma, fibrous type','meningioma of fibrous type','meningioma, fibrous'],
                      'Lipoma':['lipoma','CNS lipoma','lipoma of the CNS','lipoma of the central nervous system'],
                      'Medulloblastoma__non-WNT_non-SHH':['medulloblastoma, non-WNT/non-SHH','medulloblastoma','medulloblastoma group 3','medulloblastoma group 4'],
                      'Normal': ['normal brain tissue', 'brain normal tissue','brain non-cancerous tissue']
        
}

shenmu_names = {'Ganglioneuroblastoma, nodular(and other composite neuroblastic tumours)':['Ganglioneuroblastoma, nodular',
                                                                                           'nodular ganglioneuroblastoma'],
                'Ganglioneuroblastoma, intermixed': ['ganglioneuroblastoma, intermixed',
                        'intermixed ganglioneuroblastoma',    
                        'intermixed schwannian stroma-rich ganglioneuroblastoma',
                        ],
                'Differentiating neuroblastoma': ['differentiating neuroblastoma',
                        'neuroblastoma, differentiating type',
                        ],
                'Poorly differentiated neuroblastoma': ['poorly differentiated neuroblastoma',
                            'neuroblastoma, poorly differentiated type',
                            ],
                'Normal': ['normal nerve tissue',
                           'nerve normal tissue',
                        'nerve non-cancerous tissue',
                        ]
                    }

suimu_names = {'Large Cell/Anaplastic medulloblastoma': ['large cell medulloblastoma',
                'anaplastic medulloblastoma',
                            ],
               'Desmoplastic nodular medulloblastoma': ['desmoplastic nodular medulloblastoma',
                        'nodular medulloblastoma',
                        ],
                'Classic medulloblastoma': ['classic medulloblastoma',
                        'classic medulloblastoma, childhood',
                        ],
                'Normal': ['normal brain tissue',
                           'brain normal tissue',
                        'brain non-cancerous tissue',
                        ]
                    }

ganmu_names = {'Epithelial macrotrabecular pattern of hepatoblastoma': ['epithelial macrotrabecular pattern of hepatoblastoma',
                'macrotrabecular hepatoblastoma',
                            ],
               'Mixed epithelial and mesenchymal hepatoblastoma': ['mixed epithelial and mesenchymal hepatoblastoma',
                        'hepatoblastoma with mixed epithelial and mesenchymal pattern',
                        ],
                'Epithelial mixed fetal and embryonal hepatoblastoma': ['epithelial mixed fetal and embryonal hepatoblastoma',
                        'epithelial hepatoblastoma with mixed fetal and embryonal pattern',
                        ],
                'Pure fetal hepatoblastoma with low mitotic activity': ['pure fetal hepatoblastoma with low mitotic activity',
                           'well-differentiated fetal hepatoblastoma',
                        ],
                'Normal': ['normal liver tissue',
                        'liver normal tissue',
                        'liver non-cancerous tissue',
                        ]
                    }

shenzangmu_names = {'Nephroblastoma, blastemal type': ['nephroblastoma, blastemal type', 
                                                       'blastemal nephroblastoma' ],
                    
                'Nephroblastoma, stromal type': ['nephroblastoma, stromal type', 
                                                 'stromal nephroblastoma'],
                
                'Mixed blastemal and epithelial nephroblastoma': ['mixed blastemal and epithelial nephroblastoma', 
                                                                  'nephroblastoma with mixed blastemal and epithelial pattern'],
                
                'Mixed blastemal and stromal nephroblastoma': ['mixed blastemal and stromal nephroblastoma', 
                                                                  'nephroblastoma with mixed blastemal and stromal pattern'],
                
                'Mixed blastemal, epithelial and stromal nephroblastoma': ['mixed blastemal, epithelial and stromal nephroblastoma', 
                                                                           'nephroblastoma with mixed blastemal, epithelial and stromal pattern'],
                
                'Normal': ['normal kidney tissue',
                        'kidney normal tissue',
                        'kidney non-cancerous tissue',
                        ],
                }


sarc_names = {'Dedifferentiated liposarcoma':['dedifferentiated liposarcoma',
                                                   'dedifferentiated liposarcoma of soft tissue',
                                                   'liposarcoma, dedifferentiated'],
                   'Leiomyosarcoma (LMS)':['leiomyosarcoma',
                                           'leiomyosarcoma, malignant'],
                   'Malignant Peripheral Nerve Sheath Tumors (MPNST)':['malignant peripheral nerve sheath tumor', 
                                                                       'malignant neurilemmoma', 
                                                                       'neurofibrosarcoma, malignant'],
                   'Myxofibrosarcoma':['myxofibrosarcoma',
                                       'fibromyxosarcoma',
                                       'fibromyxoid sarcoma'],
                   'Undifferentiated Pleomorphic Sarcoma':['undifferentiated pleomorphic sarcoma',
                                                           'undifferentiated pleomorphic soft tissue sarcoma'],
                   'Normal':['normal soft tissue',
                              'non-cancerous soft tissue']
}

thym_names = {'Thymoma, Type A':['thymoma, spindle cell',
                                 'thymoma, type A',
                                      ],
                   'Thymoma, Type AB':['thymoma, mixed type',
                                       'thymoma, type AB'
                                       ],
                   'Thymoma, Type B1':['thymoma, lymphocyte-rich', 
                                       'thymoma, lymphocytic',
                                       'thymoma, type B1'
                                       ],
                   'Thymoma, Type B2':['thymoma, cortical',
                                       'thymoma, type B2'
                                       ],
                   'Normal':['normal thymus tissue',
                             'thymus normal tissue',
                              'non-cancerous thymus tissue']
}

ucs_names = {'Uterine Carcinosarcoma-MMMT-Heterologous Type': ['uterine Carcinosarcoma, heterologous type', 
                                                                      'malignant mixed müllerian tumor, heterologous type'],
    'Uterine Carcinosarcoma-MMMT-Homologous Type': ['uterine carcinosarcoma, homologous type', 
                                                     'malignant mixed müllerian tumor, heterologous type'],
                   'Normal':['normal uterus tissue',
                             'thymux uterus tissue',
                              'non-cancerous uterus tissue']
}


def read_h5(path):
    with h5py.File(path, 'r') as f:
        coords = f['coords'][:]
        features = f['features'][:]
    return coords, features

def load_all_prompts(dataset_name, classnames):
    
    data_names = eval(dataset_name.lower() + '_names')
    assert set(list(data_names.keys())) == set(classnames)
    prompt_lst = []
    for i, each_cls in enumerate(classnames):
        prompt_class_lst = []
        for each_name in data_names[each_cls]:
            for each_template in templates:
                prompt_class_lst.append(each_template.replace('CLASSNAME', each_name))
        prompt_lst.append(prompt_class_lst)
    
    return prompt_lst


def load_prompts(dataset_name, subtype_names):
    prompt_names = eval(dataset_name + '_names')
    prompt_list = []
    classname_list = []
    for sn in subtype_names:
        prompts = []
        subtype_syns = prompt_names[sn]
        for each_syn in subtype_syns:
            for each_template in templates:
                prompts.append(each_template.replace('CLASSNAME', each_syn))
        prompt_list.append(prompts)
        classname_list.append(subtype_syns)
        
    return prompt_list,classname_list
    

def load_prompts_from_template(json_path, classnames_example=None): # example with normal
    with open(json_path, 'r') as file:
        data = json.load(file)

    prompt_list = []
    classname_list = []

    if classnames_example:
        classnames_key = [classname[:3].lower() for classname in classnames_example]

    for key, value in data.items():
        classnames = value['classnames']
        template = value['templates']
        prompts = []
        classes = []
        if classnames_example:
            key_dict = {key[:3].lower():value for key, value in classnames.items()}
            for key in classnames_key:
                class_value = key_dict[key]
                new_string = template.replace('CLASSNAME', class_value)
                prompts.append(new_string)
                classes.append(class_value)
        else:    
            for class_name_type, class_value in classnames.items(): #normal 0, tumor 1
                new_string = template.replace('CLASSNAME', class_value)
                prompts.append(new_string)
                classes.append(class_value)
        prompt_list.append(prompts)
        classname_list.append(classes)
    return prompt_list, classname_list