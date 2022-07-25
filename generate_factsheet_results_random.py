import os
import json

import random
import math
import time
import datetime
import copy


import pandas as pd
import numpy as np

import cv2

import gc

#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from textwrap import wrap
import seaborn as sns


'''params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
plt.rcParams.update(params)'''



from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import random


# # Define Errors

# In[3]:


class FactSheetErrors:
    def FileError(self,file):
        raise IOError('[-] File Not Found : '+file)
        exit
    def DirectoryError(self,directory):
        raise IOError('[-] Directory Not Found : '+directory)
        exit
    def ColumnError(self,column):
        raise ValueError('[-] Column Not Found : '+column)
        exit
    def DebugError(self):
        raise ValueError('[-] Debug Super Categories are None')
        exit
    def DebugInvalidSCError(self,debug_super_category):
        raise ValueError('[-] Debug Invalid Super Category : '+debug_super_category)
        exit
err = FactSheetErrors()


def make_set_my_didactic(df, train_split, test_split, SEED, majority):
    
    df2 = df.copy()
    train = []
    train += df2[df2.GROUP.isin(majority)].groupby('GROUP').sample(train_split[0], random_state=SEED).index.to_list()
    train += df2[~df2.GROUP.isin(majority)].groupby('GROUP').sample(train_split[1], random_state=SEED).index.to_list()

    df3 = df2[~df2.index.isin(train)].copy()

    test = []
    test += df3[df3.GROUP.isin(majority)].groupby('GROUP').sample(test_split[0], random_state=SEED).index.to_list()
    test += df3[~df3.GROUP.isin(majority)].groupby('GROUP').sample(test_split[1], random_state=SEED).index.to_list()

    df2 = df2[df2.index.isin(train+test)]

    df2.loc[df2.index.isin(train), "SPLIT"] = 0
    df2.loc[df2.index.isin(test), "SPLIT"] = 1

    assert (df2[df2.index.isin(train)].SPLIT == 0).sum() == len(train)
    assert (df2[df2.index.isin(test)].SPLIT == 1).sum() == len(test)

    df2 = df2.sample(frac = 1)
    #print(df2)


    return df2

matr = np.zeros((3,3))
matr = pd.DataFrame(matr, columns = ["Car", "Plane", "Boat"], index = ["Ground", "Sky", "Water"])

def factsheet_rd(DATASET_PATH, PREDICTIONS_PATH, TRAIN_SPLIT, TEST_SPLIT, majority = ["Car_Ground", "Plane_Sky", "Boat_Water"], make_set = make_set_my_didactic, mat = matr, SET_SEED = 42, CSV_PATH = None, EPOCHS = 10, ONLY_LAST = True, PRETRAIN = True, GROUP_ACC = False, TO_SPLIT = False, show = False, GENERATE_IMAGESHEET = False):
    # Path of the dataset which contains images, labels.csv and info.json    

    if not os.path.exists(PREDICTIONS_PATH):
        os.makedirs(PREDICTIONS_PATH)

    IMAGES_PER_CATEGORY = -1
    # maximum limit on episodes/super-categories
    # can be none or an intege

    MAX_EPISODES = None

    # Normalize the input images according to the way neural networks were pretrained on ImageNet
    USE_NORMALIZATION = False

    # To generate an imagesheet : a pdf document with all the images per category/class
    #GENERATE_IMAGESHEET = False

    # Debug Flag
    DEBUG_MODE = False

    # Debug Categories
    DEBUG_SUPER_CATEGORIES = None



    # True super Categories
    TRUE_SUPER_CATEGORIES = None

    # seed for generating super-categories by the same random combination of categories
    SEED = 42

    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(False)

    # In[6]:


    # Path of the json.info file
    JSON_PATH = os.path.join(DATASET_PATH, "info.json")

    CSV_PATH = CSV_PATH if CSV_PATH else os.path.join(DATASET_PATH, "labels.csv")


    # # Create Prediction Dir

    # In[7]:


    if not os.path.exists(PREDICTIONS_PATH):
        os.makedirs(PREDICTIONS_PATH)


    # # Check Directories and Files

    # In[8]:


    # Dataset Directory
    if not os.path.exists(DATASET_PATH):
        err.DirectoryError(DATASET_PATH)
        
    #Check JSON file
    if not os.path.isfile(JSON_PATH):
        err.FileError(JSON_PATH)

    # Check CSV File
    '''if not os.path.isfile(CSV_TRAIN):
        err.FileError(CSV_TRAIN)'''

    # Check Predictions Directory
    if not os.path.exists(PREDICTIONS_PATH):
        err.DirectoryError(PREDICTIONS_PATH)


    # # Read JSON

    # In[9]:


    f = open (JSON_PATH, "r")
    info = json.loads(f.read())


    # # Settings from JSON

    # In[10]:


    # True if CSV is tab separated otherwise false
    CSV_WITH_TAB = False#info["csv_with_tab"]

    # Path of the directory where images to be used in this experiement are saved
    IMAGE_PATH = os.path.join(DATASET_PATH, "images")

    '''if info["images_in_sub_folder"]:
        IMAGE_PATH = os.path.join(DATASET_PATH, "images")
    else:  
        IMAGE_PATH = DATASET_PATH'''



    # category column name in csv
    CATEGORY_COLUMN = info["category_column_name"]

    # Super category column name in csv
    SUPER_CATEGORY_COLUMN = info["super_category_column_name"]


    # image column name in csv
    IMAGE_COLUMN = info["image_column_name"]


    # # Check True Super Categories

    # In[11]:


    TRUE_SUPER_CATEGORIES = info["has_super_categories"]


    # ### Time

    # In[12]:


    t0_start = time.time()
    timestamp = str(datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f"))


    # # Load CSV

    # In[13]:


    if CSV_WITH_TAB:
        data = pd.read_csv(CSV_PATH, sep="\t", encoding="utf-8") 
    else:
        data = pd.read_csv(CSV_PATH)

    data = make_set(data, TRAIN_SPLIT, TEST_SPLIT, SET_SEED, majority)

    # In[14]:


    if show:
        print("Data Shape : ", data.shape)

    data = data.sample(frac = 1, random_state=SEED)


    # In[17]:

    if DEBUG_MODE:
        if DEBUG_SUPER_CATEGORIES is None:
            err.DebugError()
        else:
            DEBUG_SUPER_CATEGORIES = [s_c.strip() for s_c in DEBUG_SUPER_CATEGORIES.split(',')]


    # # Categoris

    # In[20]:


    categories = data[CATEGORY_COLUMN].unique()
    total_categories = len(categories)
    ##print(CATEGORIES_TO_COMBINE)

    #if CATEGORIES_TO_COMBINE == None:
        #CATEGORIES_TO_COMBINE = len(categories)

    CATEGORIES_TO_COMBINE = total_categories

    # # Super Categories

    # In[21]:


    if TRUE_SUPER_CATEGORIES:
        super_categories = data[SUPER_CATEGORY_COLUMN].value_counts().index.values
        iterations_needed = None
    else:
        random.Random(SEED).shuffle(categories)
        iterations_needed = math.ceil(total_categories/CATEGORIES_TO_COMBINE)
        
        if show:
            print("Iterations required : ", iterations_needed)
            print("Categories to combine togather : ", CATEGORIES_TO_COMBINE)
        super_categories = np.array_split(categories,iterations_needed)
        
    total_super_categories = len(super_categories)


    # # Statistics

    # In[22]:

    if show:
        print("Total Super-Categories : ", total_super_categories)
        print("Total Categories/Classes : ", total_categories)


    # # Filter Debug Categories

    # In[23]:


    if DEBUG_MODE:
        filtered_super_categories = []
        for s_c in DEBUG_SUPER_CATEGORIES:

            if TRUE_SUPER_CATEGORIES:
                if s_c in super_categories:
                    filtered_super_categories.append(s_c)
                else:
                    err.DebugInvalidSCError(s_c)

            else:
                if int(s_c) in range(0,total_super_categories):
                    filtered_super_categories.append(int(s_c))
                else:
                    err.DebugInvalidSCError(s_c)


    # # Preparing Super_Categories

    # In[24]:


    def prepare_single_super_data(index, super_category):
        super_dict = {}
        
        if TRUE_SUPER_CATEGORIES:
            if IMAGES_PER_CATEGORY == -1:
                super_category_df = data[data[SUPER_CATEGORY_COLUMN] == super_category].sample(frac=1, random_state=SEED)
            else:
                super_category_df = data[data[SUPER_CATEGORY_COLUMN] == super_category].groupby(CATEGORY_COLUMN).sample(n=IMAGES_PER_CATEGORY, random_state=SEED)
            super_dict['super_category'] = super_category
        else:
            if IMAGES_PER_CATEGORY == -1:
                    super_category_df = data[data[CATEGORY_COLUMN].isin(super_category)].sample(frac=1, random_state=SEED)
            else:
                super_category_df = data[data[CATEGORY_COLUMN].isin(super_category)].groupby(CATEGORY_COLUMN).sample(n=IMAGES_PER_CATEGORY, random_state=SEED)
            super_dict['super_category'] = str(index)
        
        
        super_category_df['label_cat'] = super_category_df[CATEGORY_COLUMN].astype('category')
        
        
        super_dict['categories'] = super_category_df['label_cat'].cat.categories.values
        super_dict['images'] = super_category_df[CATEGORY_COLUMN].value_counts().values
        
        
        if TO_SPLIT:
            train_data, valid_data = train_test_split(
                super_category_df, test_size=0.35, 
                random_state=420, shuffle=True, 
                stratify=super_category_df[CATEGORY_COLUMN]
            )
        else:
            train_data = super_category_df[super_category_df['SPLIT'] == 0]
            valid_data = super_category_df[super_category_df['SPLIT'] == 1]

        #print(valid_data)

        
        super_dict['train_labels'] = train_data[CATEGORY_COLUMN].values
        super_dict['valid_labels'] = valid_data[CATEGORY_COLUMN].values
        
        super_dict['train_labels_num'] =  train_data['label_cat'].cat.codes.values
        super_dict['valid_labels_num'] = valid_data['label_cat'].cat.codes.values
        
        
        super_dict['train_data'] = train_data[IMAGE_COLUMN].values
        super_dict['valid_data'] = valid_data[IMAGE_COLUMN].values
        
        if GROUP_ACC:
            super_dict['train_group'] = train_data["BINS"].values
            super_dict['valid_group'] = valid_data["BINS"].values
        
        return super_dict


    # In[25]:


    super_data = []
    for index, super_category in enumerate(super_categories):
        #Debug Mode
        if DEBUG_MODE:
            if TRUE_SUPER_CATEGORIES: 
                if super_category in filtered_super_categories:
                    super_data.append(prepare_single_super_data(index, super_category))
            else:
                if index in filtered_super_categories:
                    super_data.append(prepare_single_super_data(index, super_category))   
                    
        # Not Debug Mode
        else:
            super_data.append(prepare_single_super_data(index, super_category))


    # # Baseline 

    # In[26]:


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # ### DATA

    # In[27]:


    class ImgDataset(Dataset):
        def __init__(self, dataset_images, dataset_labels, transform):

            # Transforms
            self.transform = transform
            
            self.images = dataset_images
            self.labels = dataset_labels

            self.data_len = len(self.images)

        def __getitem__(self, index):
            
            
            single_img = self.images[index]
            img_transformed = torch.from_numpy(single_img).long()
            img_transformed = img_transformed.permute(2, 0, 1)
            img_transformed = torch.from_numpy(np.array(img_transformed)).float() / 255.

            
            single_label = self.labels[index]
            single_label = single_label.astype(torch.LongTensor)
            
            return img_transformed, single_label

        def __len__(self):
            return self.data_len


    # # Make Dataset

    # In[28]:


    def make_dataset(super_data_set, batch_size=64):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        if not USE_NORMALIZATION:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        
        
        train_ds=ImgDataset(super_data_set['train_images'],super_data_set['train_labels_num'], transform)
        valid_ds=ImgDataset(super_data_set['valid_images'],super_data_set['valid_labels_num'], transform)
        
        if show:
            print("############################################")
            print("============================================")
            print("=== Super-Category: ", super_data_set['super_category'])
            print("============================================")
            print("############################################")
            print()
            print("Total Categories: ", len(super_data_set['categories']))
            print("Total Images: ", super_data_set['images'].sum())
            print("Train Data: ", len(train_ds))
            print("Validation Data: ", len(valid_ds))
            print()
        
        data_stats = {
            "total_images" : super_data_set['images'].sum(),
            "train_images" : len(train_ds),
            "valid_images" : len(valid_ds)
        }
        
        
        dataloaders = {
            'val':DataLoader(
                valid_ds, 
                batch_size=batch_size,
                shuffle=False,
            ),
            'train':DataLoader(
                train_ds, 
                batch_size=batch_size,
                shuffle=False,
            )
        }

        dataset_sizes = {
            'val': len(valid_ds),
            'train':len(train_ds)
        }
        
        return dataloaders, dataset_sizes, data_stats


    # # Model

    # In[29]:


    def getModel(only_train_last_layer=ONLY_LAST, number_of_classes=2):
        model = models.resnet18(pretrained=PRETRAIN)

        if only_train_last_layer:
            for param in model.parameters():
                param.requires_grad = False
            
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, number_of_classes)
        return model.to(device)


    def getModelAlex(only_train_last_layer=ONLY_LAST, number_of_classes=2):
        model = models.alexnet(pretrained=PRETRAIN)

        if only_train_last_layer:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, number_of_classes)
        return model.to(device)


    # # Training Loop

    # In[30]:


    def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=EPOCHS):
        
        if show:
            print("--------------------------------------------")
            print("Training")
            print("--------------------------------------------")
        
        since = time.time()
        
        best_model_wts = copy.deepcopy(model.state_dict())
        
        best_val_acc = 0.0
        best_train_acc = 0.0
        
        loss_history = []
        score_history = []
        
        
        train_loss, train_score, valid_loss, valid_score = [], [], [], []

        if show:
            print("Epoch: ", end=" ")
        for epoch in range(num_epochs):
            if show:
                print(epoch, end=" ")
            
            train_predictions, train_ground, valid_predictions, valid_ground = [], [], [], []
            train_predicted_probabilities, valid_predicted_probabilities = [],[]
            

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        
    #                     inputs = inputs.permute(0, 3, 1, 2)
    #                     inputs = torch.from_numpy(np.array(inputs)).float() / 255.
                        
                        outputs = model(inputs)
                        probabilities = F.softmax(outputs, dim=1)
                        _, preds = torch.max(outputs, 1)
                     
                        loss = criterion(outputs, labels)
                        
                        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                    # saving prediction and ground truth for future use
                    # saving prediction and ground truth for future use
                    if phase == 'train':
                        train_predictions += list(preds.cpu().numpy())
                        train_ground += list(labels.cpu().numpy())
                        train_predicted_probabilities += list(probabilities.cpu().detach().numpy())
                    else:
                        valid_predictions += list(preds.cpu().numpy())
                        valid_ground += list(labels.cpu().numpy())
                        valid_predicted_probabilities += list(probabilities.cpu().detach().numpy())
                    
                    
                # end dataloader loop

                if phase == 'train':
                    scheduler.step()
    #                 loss_history.append(running_loss)
    #                 score_history.append(running_corrects)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()

    #             print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                
                loss_history, score_history = (train_loss, train_score) if phase == 'train' else (valid_loss, valid_score)
                loss_history.append(epoch_loss)
                score_history.append(epoch_acc)

                
                if phase == 'train' and epoch_acc > best_train_acc:
                     best_train_acc = epoch_acc
                
                # deep copy the model
                if phase == 'val' and epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            # end phase loop
            

        time_elapsed = time.time() - since
        if show:
            print()
            print()
        
        training_time = '{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
        
        if show:
            print('Training complete in: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Best train Acc: {:.2f}'.format(best_train_acc))
            print('Best val Acc: {:.2f}'.format(best_val_acc))
            print()

        # load best model weights
        model.load_state_dict(best_model_wts)

        return dict(   
            model = model,
            train_loss = train_loss,
            train_score = train_score,
            train_best_score = round(best_train_acc,2),
            train_ground = train_ground,
            train_predictions = train_predictions,
            train_predicted_probabilities = np.array(train_predicted_probabilities),
            valid_loss = valid_loss,
            valid_score = valid_score,
            valid_best_score = round(best_val_acc,2),
            valid_ground = valid_ground,
            valid_predictions = valid_predictions,
            valid_predicted_probabilities =  np.array(valid_predicted_probabilities),
            training_time = training_time
            
        )


    # In[ ]:





    # # Plot accuracy and loss

    # In[31]:


    def plot_train_results(train_results, super_category):
        
        
        standard_error = train_results['standard_error']
        y_upper = train_results["valid_score"] + standard_error
        y_lower = train_results["valid_score"] - standard_error
        

        if show:
            print("--------------------------------------------")
            print("Results")
            print("--------------------------------------------")
        
        fig = plt.figure(figsize=(20,8))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(0,len(train_results["train_score"])), train_results["train_score"], label='train')

        plt.plot(range(0,len(train_results["valid_score"])), train_results["valid_score"], label='valid')
        
        
        kwargs = {'color': 'black', 'linewidth': 1, 'linestyle': '--', 'dashes':(5, 5)}
        plt.plot(range(0,len(train_results["valid_score"])), y_lower, **kwargs)
        plt.plot(range(0,len(train_results["valid_score"])), y_upper, **kwargs, label='validation SE (68% CI)')
        
        
        plt.title('Accuracy Plot - ' + super_category, fontsize=20)
        plt.ylabel('Accuracy', fontsize=16)
        plt.xlabel('Training Epochs', fontsize=16)
        plt.ylim(0, 1)
        plt.legend()


        plt.subplot(1, 2, 2)
        plt.plot(range(0,len(train_results["train_loss"])), train_results["train_loss"], label='train')
        plt.plot(range(0,len(train_results["valid_loss"])), train_results["valid_loss"], label='valid')
        

        
        plt.title('Loss Plot - ' + super_category, fontsize=20)
        plt.ylabel('Loss', fontsize=16)
        plt.xlabel('Training Epochs', fontsize=16)
        max_train_loss = max(train_results["train_loss"])
        max_valid_loss = max(train_results["valid_loss"])
        y_max_t_v = max_valid_loss if max_valid_loss > max_train_loss else max_train_loss
        ylim_loss = y_max_t_v if y_max_t_v > 1 else 1
        plt.ylim(0, ylim_loss)
        plt.legend()

      
        if show:
            plt.show()
        
        super_category_path = os.path.join(PREDICTIONS_PATH,super_category, "train_results.png")
        fig.savefig(super_category_path, dpi=fig.dpi)
        
        
        
    def get_error_bar(best_score, valid_examples):
        
        if show:
            print("--------------------------------------------")
            print("Standard Error")
            print("--------------------------------------------")
            

        
        err = np.sqrt((best_score * (1-best_score))/valid_examples)
        err_rounded_68 = round(err,2)
        err_rounded_95 = round((err_rounded_68 * 2),2)
       
        if show:
            print('Error (68% CI): +- ' + str(err_rounded_68))
            print('Error (95% CI): +- ' + str(err_rounded_95))
            print()
        return err_rounded_68
        


    # # Plot Confusion Matrix

    # In[32]:


    def plot_confusion_matrix(grounds, preds, super_category, categories):
        if show:
            print("--------------------------------------------")
            print("Confusion Matrix")
            print("--------------------------------------------")
        
        num_cat = []
        for ind, cat in enumerate(categories):
            if show:
                print("Class {0} : {1}".format(ind, cat))
            num_cat.append(ind)
        if show:
            print()
        
        
        
        cm = confusion_matrix(grounds, preds, labels=num_cat)
        
        figsize = (10,8) if len(categories) <= 15 else (20,20)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

        # labels, title and ticks
        ax.set_title('Confusion Matrix - '+ super_category, fontsize=20)
        ax.set_xlabel('Predicted labels', fontsize=16)
        ax.set_ylabel('True labels', fontsize=16)
        
        ax.xaxis.set_ticklabels(num_cat)
        ax.yaxis.set_ticklabels(num_cat)
        
        #plt.pause(0.1)
        if show:
            plt.show()
        super_category_path = os.path.join(PREDICTIONS_PATH,super_category, "confusion_matrix.png")
        fig.savefig(super_category_path, dpi=fig.dpi)
        


    # # Plot Sample Images

    # In[33]:


    def plot_sample_images(super_category, categories, grounds, train_images,train_image_names):
        if show:
            print("--------------------------------------------")
            print("Sample Images - ", super_category)
            print("--------------------------------------------")

        
        uniques, indexes = np.unique(grounds, return_index=True)
        rows = math.ceil(len(indexes)/5)
        fig = plt.figure(figsize=(30,7*rows))
        
        k=0
        for i in range(0, len(indexes)):
            fig.add_subplot(rows,5,k+1)
            plt.axis('off')
            
            lbl_alph = categories[uniques[i]]
            lbl_int = str(uniques[i])
            lbl_file = train_image_names[uniques[i]]
            
            title_lbl = "\n".join(wrap("{}({})".format(lbl_alph,lbl_int),30))
            title_file = "\n".join(wrap(lbl_file,30))
            title = title_lbl+"\n"+title_file
            
            plt.title(title)
            plt.imshow(train_images[indexes[i]])
            k += 1
        #plt.pause(0.1)
        if show:
            plt.show()
        if show:
            print()   
        
        super_category_path = os.path.join(PREDICTIONS_PATH,super_category, "sample_images.png")
        fig.savefig(super_category_path, dpi=fig.dpi, bbox_inches='tight')


    # # Plot Wrongly Classified Images

    # In[34]:


    def plot_wrongly_classified_images(super_category, categories, grounds, preds, valid_images, valid_image_names):
        
        if show:
            print("--------------------------------------------")
            print("Wrongly Classified Images - ", super_category)
            print("--------------------------------------------")
        
        fig = plt.figure(figsize=(30,20))
        k=0
        
        for i in range(0, len(grounds)):
            if grounds[i] != preds[i]:
                
               
                fig.add_subplot(1,5,k+1)
                plt.axis('off')
                
                
                orig_lbl_alph = categories[grounds[i]]
                orig_lbl_int = str(grounds[i])
               
                
                pred_lbl_alph = categories[preds[i]]
                pred_lbl_int = str(preds[i])
                pred_lbl_file = valid_image_names[preds[i]]
                
                org_title_lbl = "\n".join(wrap("Orig lbl : {}({})".format(orig_lbl_alph,orig_lbl_int),30))
                pred_title_lbl = "\n".join(wrap("Pred lbl : {}({})".format(pred_lbl_alph,pred_lbl_int),30))
                pred_title_file = "\n".join(wrap(pred_lbl_file,30))
                
                title = org_title_lbl+"\n"+pred_title_lbl+"\n"+pred_title_file
                
               
                plt.title(title)
                plt.imshow(valid_images[i])
                k += 1
            if k == 5:
                break
        #plt.pause(0.1)
        if show:
            plt.show()
        if show:
            print()
        
        super_category_path = os.path.join(PREDICTIONS_PATH,super_category, "wrongly_classified_images.png")
        fig.savefig(super_category_path, dpi=fig.dpi, bbox_inches='tight')
        
    def get_wrongly_classified_images_indexes(grounds, preds):
        wrong_images_indexes = []
        for i in range(0, len(grounds)):
            if grounds[i] != preds[i]:
                wrong_images_indexes.append(i)
        return wrong_images_indexes


    # # Plot AUC

    # In[35]:


    def autodl_auc(outputs, targets, predictions):
        
        targets = np.asarray(targets)
        predictions = np.asarray(predictions)
        
        numclass = targets.max()+1
        
        boolean_array = np.zeros((len(outputs),numclass), dtype=bool)
        
        for labelindex in range(numclass):
            boolean_array[:,labelindex]= (targets == labelindex)
        
        
        auc = roc_auc_score(boolean_array, outputs)
        auc_1 = 2*auc-1
        
        
        return round(auc, 2), round(auc_1, 2)
       


    def plot_auc(outputs, targets, predictions, autodl_auc_0,autodl_auc_1, super_category, categories):
        
        if show:
            print("--------------------------------------------")
            print("Average AUC")
            print("--------------------------------------------")
        
        if show:
            print("AUC : ", autodl_auc_0)
            print("2*AUC-1 : ", autodl_auc_1)
            print()
        

        
        targets = np.asarray(targets)
        predictions = np.asarray(predictions)
        
        
        
        numclass = len(categories) #targets.max()+1 
        
        scores_auroc = []
        scores_auroc_1 =[]

        for labelindex in range(numclass):
            binary_targets = (targets == labelindex)
            binary_predictions = (predictions == labelindex)
           
            selected_outputs = outputs[:,labelindex]
      
            auroc = roc_auc_score(binary_targets, selected_outputs)
            
            scores_auroc.append(auroc)
            scores_auroc_1.append(2*auroc-1)
            
        
        
        
        
        
        #save categories auc  
        categories_auc_textfile_path = os.path.join(PREDICTIONS_PATH, super_category, 'categories_auc.txt') 
        with open(categories_auc_textfile_path, 'w', encoding="utf-8") as f:
            for i,cat in enumerate(categories):
                single_auroc = round(scores_auroc[i], 2)
                
                f.write("%s : %s\n" %(cat,single_auroc))
        
        
        
        
        
        
        average_auc = round(np.mean(scores_auroc), 2)
        average_auc_1 = round(np.mean(scores_auroc_1), 2)
        if average_auc_1 == 0.0:
            average_auc_1 = 0
        
        
        
        # Plot AUC Score
        ymin = np.min(scores_auroc_1) if np.min(scores_auroc_1) < 0  else 0
        width = 0.2
        
       
        fig_height = 8 if len(categories) <= 5 else 16
        fig = plt.figure(figsize=(3*numclass,fig_height))
        
        
        plt.bar(np.arange(numclass), scores_auroc, width,  label='AUC')
        plt.bar(np.arange(numclass)+width, scores_auroc_1, width, label='2*AUC-1')
        plt.hlines(y=0.0, xmin=-width, xmax=numclass-1+width*2, linewidth=1, linestyles='-', color='black')
        plt.hlines(y=average_auc, xmin=-width, xmax=numclass-1+width*2, linewidth=2, linestyles='--', color='b', label='Average AUC : %0.2f'%average_auc)
        plt.hlines(y=average_auc_1, xmin=-width, xmax=numclass-1+width*2, linewidth=2, linestyles='--', color='r', label='Average 2*AUC-1 : %0.2f'%average_auc_1)
        plt.title('AUC Score - ' + super_category, fontsize=20)
        plt.ylabel('AUC Score', fontsize=16)
        plt.xlabel('Classes', fontsize=16)
        plt.ylim(ymin,1)
        plt.xticks(np.arange(numclass) + width / 2, np.arange(numclass))
        plt.legend()
    #     #plt.show()
        #plt.pause(0.1)

        if show:
            plt.show()
        
        super_category_path = os.path.join(PREDICTIONS_PATH,super_category, "auc.png")
        fig.savefig(super_category_path, dpi=fig.dpi)
        
        
        
        
        # histogram
        fig = plt.figure(figsize=(8,5))
        title = 'AUC Histogram - '+super_category
        plt.title(title, fontsize=20)
        plt.xlabel('AUC Score', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        plt.hist(scores_auroc, alpha=0.5, ec='black')
        #plt.pause(0.1)
        if show:
            plt.show()
        super_category_path = os.path.join(PREDICTIONS_PATH,super_category, "auc_histogram.png")
        fig.savefig(super_category_path, dpi=fig.dpi)
        
        


    # # Plot ROC Curves

    # In[36]:


    def plot_roc_curves(train_outputs, train_targets, train_predictions, valid_outputs, valid_targets, valid_predictions, super_category):
        
        if show:
            print("--------------------------------------------")
            print("ROC Curves")
            print("--------------------------------------------")
        
        
       
        train_targets = np.asarray(train_targets)
        train_predictions = np.asarray(train_predictions)
        valid_targets = np.asarray(valid_targets)
        valid_predictions = np.asarray(valid_predictions)
        
        
        numclass = train_predictions.max()+1 
        
        
        train_auc_curves = []
        valid_auc_curves = []

        for labelindex in range(numclass):
            train_binary_targets = (train_targets == labelindex)
            train_binary_predictions = (train_predictions == labelindex)
            
            valid_binary_targets = (valid_targets == labelindex)
            valid_binary_predictions = (valid_predictions == labelindex)
           
            selected_train_outputs = train_outputs[:,labelindex]
            selected_valid_outputs = valid_outputs[:,labelindex]
            
            train_fpr, train_tpr, _ = roc_curve(train_binary_targets, selected_train_outputs)
            valid_fpr, valid_tpr, _ = roc_curve(valid_binary_targets, selected_valid_outputs)
           
            train_auc_curves.append([train_fpr,train_tpr])
            valid_auc_curves.append([valid_fpr,valid_tpr])
            
      
        
        
        
         # Plot ROC Curves

        fig_height = 8 if numclass <= 5 else 16
        fig_width =  16 if numclass <= 5 else 30
        fig = plt.figure(figsize=(fig_width,fig_height))
       

       
        plt.subplot(1,2,1)
        for idx, auc_cur in enumerate(train_auc_curves): 
            plt.plot(auc_cur[0], auc_cur[1], marker='.',  label='Class:'+str(idx))
        plt.plot([0,1], [0,1], linestyle='--', color='black')
        plt.title('Train ROC Curves - ' + super_category, fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.legend()
        
        
        plt.subplot(1,2,2)
        for idx, auc_cur in enumerate(valid_auc_curves): 
            plt.plot(auc_cur[0], auc_cur[1], marker='.',  label='Class:'+str(idx))
        plt.plot([0,1], [0,1], linestyle='--', color='black')
        plt.title('Valid ROC Curves - ' + super_category, fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.legend()
        
        #plt.pause(0.1)
        if show:
            plt.show()
        
        super_category_path = os.path.join(PREDICTIONS_PATH,super_category, "roc_curves.png")
        fig.savefig(super_category_path, dpi=fig.dpi)
        


    # # Save Predictions

    # In[37]:

    def group_accuracy(training_result, single_super_data, prefix):
        gr = {key:0 for key in np.unique(single_super_data[prefix+"_group"])}
        #print(gr)
        total = gr.copy()
        for i in range(len(single_super_data[prefix+"_data"])):
            idx = single_super_data[prefix+"_group"][i]
            gr[idx]+= training_result[prefix+'_ground'][i] == training_result[prefix+'_predictions'][i]
            total[idx] += 1
        
        for i in gr:
            gr[i] = gr[i]/total[i]
            #print(f"{prefix} Group {i}  Accuracy : {gr[i]}")

        return gr


    def group_class_accuracy(training_result, single_super_data, prefix):

        class_name = np.unique(single_super_data[prefix+'_labels'])
        group_name = np.unique(single_super_data[prefix+"_group"])

        #gr = {key:0 for key in group_name}
        #gr2 = {key:0 for key in group_name}

        #gr = {f:gr.copy() for f in class_name}
        #total = {f:gr2.copy() for f in class_name}
        gr = mat.copy()
        total = mat.copy()
        #print(gr)

        #convert_gr = {k:i for (i,k) in enumerate(group_name)}
        #convert_cl = {k:i for (i,k) in enumerate(class_name)}
        #res = np.zeros((3,3))

        print()

        for i in range(len(single_super_data[prefix+"_data"])):
            grp = single_super_data[prefix+"_group"][i]
            #cl = training_result[prefix+'_labels'][i]
            cl = single_super_data[prefix+'_labels'][i]
            #c,g = convert_cl[cl], convert_gr[grp]
            #res[c,g] += training_result[prefix+'_ground'][i] == training_result[prefix+'_predictions'][i]

            gr[cl][grp] += training_result[prefix+'_ground'][i] == training_result[prefix+'_predictions'][i]
            total[cl][grp] += 1

        ##print(training_result[prefix+'_ground'])
        ##print(single_super_data[prefix+'_labels'])

        ##print("True Group :", res)
        ##print("Group : ", gr)
        ##print("Total : ", total)
        #gr = pd.DataFrame(gr)
        #total = pd.DataFrame(total)

        gr = gr.div(total)

        #print(gr)

        return gr



    def plot_group_confusion_matrix(cm, title):
        from matplotlib.patches import Rectangle

        cm = cm.round(2)
        figsize = (10,8) if len(categories) <= 15 else (20,20)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()

        sns.heatmap(cm, annot=True, fmt='g', ax=ax, vmin=0, vmax=1, annot_kws={"size": 20});  #annot=True to annotate cells, ftm='g' to disable scientific notation
        
        
        ax.collections[0].colorbar.ax.tick_params(labelsize=16)
        
        # labels, title and ticks
        ax.set_title(title, fontsize=25)
        ax.set_xlabel('Class label Y ', fontsize=20)
        ax.set_ylabel('Spurious feature bin Z', fontsize=20)

        c = cm.to_numpy()
        m = np.amin(c)
        indices = []
        for i in range(c.shape[0]):
            for n in range(c.shape[1]):
                if c[i,n] == m:
                    indices.append((n,i))
        for ind in indices:
            ax.add_patch(Rectangle(ind,1,1, fill=False, edgecolor='blue', lw=3))

        idx = cm.index
        col = cm.columns

        ax.xaxis.set_ticklabels(col, fontsize=16)
        ax.yaxis.set_ticklabels(idx, fontsize=16)
        
        if show:
            plt.show()
        super_category_path = os.path.join(PREDICTIONS_PATH, title.replace(" ", "_") + ".png")
        fig.savefig(super_category_path, dpi=fig.dpi)


    def save_predictions(super_category, categories, single_super_data, training_result):

        train_dic_for_df = {}
        valid_dic_for_df = {}



        train_dic_for_df['ground_truth'] = training_result['train_ground']
        train_dic_for_df['predictions'] = training_result['train_predictions']
        train_dic_for_df['train_data'] = single_super_data['train_data']
        train_dic_for_df['train_labels'] = single_super_data['train_labels']

        valid_dic_for_df['ground_truth'] = training_result['valid_ground']
        valid_dic_for_df['predictions'] = training_result['valid_predictions']
        valid_dic_for_df['valid_data'] = single_super_data['valid_data']
        valid_dic_for_df['valid_labels'] = single_super_data['valid_labels']

        train_probability_array = list(training_result['train_predicted_probabilities'].T)
        valid_probability_array = list(training_result['valid_predicted_probabilities'].T)
        for idx, prob in enumerate(train_probability_array):
            key = 'prob_'+str(idx)
            train_dic_for_df[key] = prob
        for idx, prob in enumerate(valid_probability_array):
            key = 'prob_'+str(idx)
            valid_dic_for_df[key] = prob


        train_df = pd.DataFrame(train_dic_for_df)
        valid_df = pd.DataFrame(valid_dic_for_df)
        

        
        csv_train = os.path.join(PREDICTIONS_PATH, super_category, 'train.csv') 
        csv_valid = os.path.join(PREDICTIONS_PATH, super_category, 'valid.csv') 
        

        #save train valid_predictions_ground_probabilities in CSV  
        train_df.to_csv(csv_train, index=False)
        valid_df.to_csv(csv_valid, index=False)

        
        #save categories  
        categories_textfile_path = os.path.join(PREDICTIONS_PATH, super_category, 'categories.txt') 
        with open(categories_textfile_path, 'w', encoding="utf-8") as f:
            for item in categories:
                f.write("%s\n" % item)
        
        #save category logs
        category_logfile_path = os.path.join(PREDICTIONS_PATH, super_category, 'logs.txt') 
        with open(category_logfile_path, 'w') as f:
            f.write("Total Images : %s\n" % training_result['total_images'])
            f.write("Training Images : %s\n" % training_result['train_images'])
            f.write("Validation Images : %s\n" % training_result['valid_images'])
            f.write("Training Time : %s\n" % training_result['training_time'])
            f.write("Best Train Accuracy : %s\n" % training_result['train_best_score'])
            f.write("Best Valid Accuracy : %s\n" % training_result['valid_best_score'])
            f.write("AUC : %s\n" % training_result['AUC'])
            f.write("2*AUC-1 : %s\n" % training_result['AUC_1'])
            f.write("Standard Error : %s\n" % training_result['standard_error'])
            

        
        if show:
            print("Saved Results for Super-Category : ", super_category)
            print()
            print()
            
        #save super-category
        super_categories_textfile_path = os.path.join(PREDICTIONS_PATH, 'super_categories.txt') 
        with open(super_categories_textfile_path, 'a', encoding="utf-8") as f:
            f.write("%s\n" % super_category)
            
            
    def save_overall_logs():
        #save logs 
        overall_logsfile_path = os.path.join(PREDICTIONS_PATH, 'logs.txt') 
        with open(overall_logsfile_path, 'w') as f:
            f.write("Total Super Categories : %s\n" % total_super_categories)
            f.write("Total Categories : %s\n" % total_categories)
            f.write("Iterations Needed : %s\n" % iterations_needed)
            f.write("Classes to Combine : %s\n" % CATEGORIES_TO_COMBINE)
            
        if show:
            print("#################################")
            print("Saved Overall logs of experiment")
            print("#################################")
            
            
            
    def make_super_category_directory(super_category):
        
        super_category_path = os.path.join(PREDICTIONS_PATH,super_category)
        if not os.path.exists(super_category_path):
            os.makedirs(super_category_path)
            
        
        


    # # Generate overall histogram

    # In[38]:


    def get_range(dictionary, begin, end):
        return dict(e for i, e in enumerate(dictionary.items()) if begin <= i <= end)


    def generate_overall_auc_histogram_and_desc_auc_plot():
        
        #Read super_Categories
        super_categories_textfile_path = os.path.join(PREDICTIONS_PATH, 'super_categories.txt') 
        with open(super_categories_textfile_path, 'r') as f:
            super_categories = f.read().splitlines()
            
        #Read Categories AUC    
        categories_dic = {}
        for sup_cat in super_categories:
            categories_auc_textfile_path = os.path.join(PREDICTIONS_PATH, sup_cat, 'categories_auc.txt') 
            with open(categories_auc_textfile_path, 'r') as f:
                categories_auc = f.read().splitlines()
            for item in categories_auc:
                arr_split = item.split(' : ')
                categories_dic[arr_split[0]] = float(arr_split[1])
                
        #Sort in descending order        
        sorted_categories_dic = dict(sorted(categories_dic.items(), key=lambda x: x[1], reverse=True))
        
        
        if show:
            print("##########################################")
            print("Saving Overall AUC Histogram")
            print("##########################################")
            print()
            print()
        
        #plot histogram
        fig = plt.figure(figsize=(15,8))
        plt.title('All CategoriesAUC Histogram', fontsize=20)
        plt.xlabel('AUC Score', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        plt.hist(list(sorted_categories_dic.values()), alpha=0.5, ec='black')
        if show:
            plt.show()
        overall_categoris_auc_histogram_path = os.path.join(PREDICTIONS_PATH, "overall_auc_histogram.png")
        fig.savefig(overall_categoris_auc_histogram_path, dpi=fig.dpi)
        
        
        
        
        
        if show:
            print("##########################################")
            print("Saving Desc AUC Plot")
            print("##########################################")
            print()
            print()
        
        total_cat = len(sorted_categories_dic)
        cat_per_plot = 30
        plots_needed = math.ceil(len(sorted_categories_dic)/cat_per_plot)

        #print("Total Categories : ", total_cat)
        #print("Plots Needed : ", plots_needed)
        #print("Categories per plot : ", cat_per_plot)
        
        fig, axs = plt.subplots(plots_needed,1, figsize=(20, plots_needed*cat_per_plot))
        fig.subplots_adjust(hspace = .1, wspace=.001)
        
        if type(axs) is np.ndarray:
            axs = axs.ravel()
        else:
            axs = [axs]

        for i in range(plots_needed):
            begin = i*cat_per_plot
            end = (i*cat_per_plot+cat_per_plot-1) if (i*cat_per_plot+cat_per_plot-1) < total_cat else total_cat 
            slice_i = get_range(sorted_categories_dic, begin, end)

            axs[i].barh(list(slice_i.keys()), list(slice_i.values()))
            axs[i].set_title("All Categories AUC - Slice "+ str(i+1), fontsize=20)
            axs[i].xaxis.set_tick_params(rotation=90)
            axs[i].set_xlabel("AUC Score")
            axs[i].set_ylabel("Category")
            axs[i].invert_yaxis()
            ymin,ymax=axs[i].get_ylim()
            axs[i].vlines(0.5, ymin=ymin, ymax=ymax, linestyles ="--", colors ="r")


        descending_categoris_auc_path = os.path.join(PREDICTIONS_PATH, "descending_auc.png")
        fig.savefig(descending_categoris_auc_path, dpi=fig.dpi)
        


    # # Generate Image Sheet

    # In[39]:


    def generate_image_sheet():
        
        if show:
            print("#################################")
            print("Generating Imagesheet")
            print("#################################")
            
        rows = math.ceil(60/5)
        imagesheet_path = PREDICTIONS_PATH +'/imagesheet.pdf'
        with PdfPages(imagesheet_path) as pdf:
            
            #for cat in categories:
            for cat in data.GROUP.unique():
                #category_images = data[data[CATEGORY_COLUMN] == cat].groupby(CATEGORY_COLUMN).sample(frac=1, random_state=SEED)[IMAGE_COLUMN].values
                category_images = data[data["GROUP"] == cat].groupby(CATEGORY_COLUMN).sample(frac=1, random_state=SEED)[IMAGE_COLUMN].values
                fig = plt.figure(figsize=(30,5*rows))
                k=0
                for i, image_name in enumerate(category_images):
                    fig.add_subplot(rows,5,k+1)
                    
                    file = IMAGE_PATH+"/"+image_name
                    img = cv2.imread(file)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    title = "\n".join(wrap(image_name,30))
                  
                    
                    plt.title(title)
                    plt.axis('off')
                    plt.imshow(img_rgb)
                    
                    k += 1

                fig.suptitle("\n"+cat, fontsize=40)
                pdf.savefig(fig)
                #plt.pause(0.1)
                if show:
                    plt.show()
        
                plt.close(fig)
                
                


    # In[ ]:


    # # Magic Happens here

    # In[40]:

    def delete_bg(im, mask):
        mask = (mask > 255/2)*255
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i,j,0] == 0:
                    im[i,j] = np.zeros(3)+255
        return im

    def load_images(single_super_data):
            
        train_images = []
        valid_images = []
        
        for image_name in single_super_data['train_data']:
            file = IMAGE_PATH+"/"+image_name
            img = cv2.imread(file)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            train_images.append(img_rgb)

        for image_name in single_super_data['valid_data']:
            file = IMAGE_PATH+"/"+image_name
            img = cv2.imread(file)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            valid_images.append(img_rgb)
            
        single_super_data['train_images'] = train_images
        single_super_data['valid_images'] = valid_images
        return single_super_data



    def train_single_super_category(single_super_data):
        
        single_super_data = load_images(single_super_data)
        

        dataloaders, dataset_sizes, data_stats = make_dataset(single_super_data)
        model = getModelAlex(number_of_classes=len(single_super_data['categories']))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        result = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, num_epochs=EPOCHS)
        
        
        #data statistics
        result['total_images'] = data_stats['total_images']
        result['train_images'] = data_stats['train_images']
        result['valid_images'] = data_stats['valid_images']
        
        
        
        # make directory for storing results
        make_super_category_directory(single_super_data['super_category'])

        #Accuracy, Loss and Error
        result['standard_error'] = get_error_bar(result['valid_best_score'], len(single_super_data['valid_images']))
        plot_train_results(result, single_super_data['super_category'])
        
        if GROUP_ACC:
            group_accuracy(result, single_super_data, prefix = 'train')
            group_accuracy(result, single_super_data, prefix = 'valid')
            cm_train = group_class_accuracy(result, single_super_data, prefix = 'train')
            plot_group_confusion_matrix(cm_train, "Training Group Accuracy")
            cm_test = group_class_accuracy(result, single_super_data, prefix = 'valid')
            plot_group_confusion_matrix(cm_test, "Test Group Accuracy")
            #plot_wrong_class_group(cm_test, result, single_super_data)

                
        #Confusion Matrix
        plot_confusion_matrix(result['valid_ground'], result['valid_predictions'], single_super_data['super_category'], single_super_data['categories'])
        
        #AUC
        autodl_auc_0 , autodl_auc_1 = autodl_auc(result['valid_predicted_probabilities'], result['valid_ground'], result['valid_predictions'])
        result['AUC'] = autodl_auc_0
        result['AUC_1'] = autodl_auc_1
        plot_auc(result['valid_predicted_probabilities'], result['valid_ground'], result['valid_predictions'],autodl_auc_0, autodl_auc_1, single_super_data['super_category'], single_super_data['categories'])
        
        
        #ROC Curves
        plot_roc_curves(result['train_predicted_probabilities'], result['train_ground'], result['train_predictions'],
                result['valid_predicted_probabilities'], result['valid_ground'], result['valid_predictions']
                , single_super_data['super_category'])
        
        #Sample Images
        plot_sample_images(single_super_data['super_category'], single_super_data['categories'], result['train_ground'], single_super_data['train_images'], single_super_data['train_data'])
        
        #Wrong Images
        plot_wrongly_classified_images(single_super_data['super_category'], single_super_data['categories'], result['valid_ground'], result['valid_predictions'], single_super_data['valid_images'], single_super_data['valid_data'])
        
        #Save Predictions
        ##print("here 222222222222222: ", result)
        save_predictions(single_super_data['super_category'], single_super_data['categories'], single_super_data, result)

        
        
        #CleanUp
        del dataloaders
        del model
        del criterion
        del optimizer
        del exp_lr_scheduler
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result, cm_train, cm_test


    # ### Training starts here

    # In[41]:


    super_results = {}


    # In[42]:


    # save overall logs
    save_overall_logs()


    # In[43]:


    #generate image sheet
    if GENERATE_IMAGESHEET:
        generate_image_sheet()


    # In[44]:


    # loop over all random super-categories to get results per super-category
    for index, singlee_super_data in enumerate(super_data):
        if MAX_EPISODES is None or index < MAX_EPISODES:
            super_results[singlee_super_data['super_category']], cm_train, cm_test = train_single_super_category(singlee_super_data)
            my_model = super_results[singlee_super_data['super_category']]['model']
            valid_set = singlee_super_data['valid_data']
            plt.close('all')


    # In[45]:


    #generate over all hostogram and auc desc plot
    generate_overall_auc_histogram_and_desc_auc_plot()
     


    # In[46]:


    if show:
        print("The whole process done in {:.2f} s.".format(time.time() - t0_start))

    return cm_train, cm_test, data


# In[ ]:




