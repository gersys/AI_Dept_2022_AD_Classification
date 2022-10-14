from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd
import json


class DataFrame_wrapper():
    def __init__(self, dataframe, filename):
        self.dataframe = dataframe
        self.filename = filename


class CIVET_DATASET(Dataset):
    """
    Civet Dataset Preprocesser.
    
    w/ APOE: use APOE_e4
    w/ demo: use yellow columns { binarize(성별) | normalize(검사시나이) | normalize(교육연수) }
    
    """
    
    def __init__(self,
                 frame_type,
                 fold_idxs,
                 xlsx_dataset,
                 xlsx_dataset_kfold,
                 xlsx_dataset_demo_apoe,
                 verbose=False):
        
        """
        frame_type:             type=str. "GT" or "VFI". if "GT": Use data from xlsx_dataset_demo_apoe, if "VFI": use data from xlsx_dataset
        fold_idx:               type=list(str) or type=list(int). Contains list of columns to read in xlsx_dataset_kfold
        xlsx_dataset:           type=str or type=DataFrame_wrapper. if str: Reads xlsx file , else: assign directly.
        xlsx_dataset_demo_apoe: type=str or type=DataFrame_wrapper. if str: Reads xlsx file , else: assign directly.
        xlsx_dataset_kfold:     type=str or type=DataFrame_wrapper. if str: Reads xlsx file , else: assign directly.
        verbose:                type=bool.  if False: self.verbose_print will be muted.
        
        """
        
        # todo. change column name in excel file. 성별:sex, PET검사나이:PET_AGE, 교육연수:EDU
        # mist
        super(CIVET_DATASET, self).__init__()
        assert frame_type in ["VFI", "GT"]
        
        # predefined column lists
        self.col_CIVET = ["Cingulate", "Frontal", "Parietal", "Temporal", "Occipital"]
        self.col_DEMO = ["sex", "PET_AGE", "EDU"]  # Demographic
        self.col_APOE = ["APOE_e4"]
        self.col_to_NORMALIZE = ["PET_AGE", "EDU", "Cingulate", "Frontal", "Parietal", "Temporal", "Occipital"]
        
        # unique key to match between multiple dataframes
        self.unique_key = "UNIQUE_KEY"
        self.get_unique_key_fn = lambda x: x.split("_nuc_")[0]
        
        # dataset meta-data
        self.verbose = verbose
        self.frame_type = frame_type
        self.metadata = {
            "error_count": 0,
            "fold_idxs": fold_idxs,
            
            "col_DEMO": self.col_DEMO,
            "col_APOE": self.col_APOE,
            "col_CIVET": self.col_CIVET,
            "col_to_NORMALIZE": self.col_to_NORMALIZE,
            
            "unique_key": self.unique_key,
            "frame_type": self.frame_type,
        }

        # read dataset assets
        self.dataframes = dict()
        for k, v in {"dataset":xlsx_dataset, "dataset_kfold":xlsx_dataset_kfold, "dataset_demo_apoe":xlsx_dataset_demo_apoe}.items():
            if isinstance(v, DataFrame_wrapper):
                self.metadata[k] = v.filename
                self.dataframes[k] = v.dataframe.copy()
            
            elif isinstance(v, str):
                self.verbose_print(f"reading {v}")
                preload = preload_xlsx(v)
                self.metadata[k] = preload.filename
                self.dataframes[k] = preload.dataframe.copy()
            else:
                raise NotImplementedError(f"Unkown type for {k}. Got {type(v)}")

        # preprocess, merge, select k-fold ... etc
        self.collect_n_fold()
        self.set_unique_keys()
        self.preprocess()
        self.match_k_fold()

        # wrap-up and clean
        self.dataframes["dataset"].dropna(axis=0, how="any", inplace=True)  # drop any row(axis=0) where nan exists
        self.dataframes["dataset"].reset_index(drop=True, inplace=True)
        self.y = self.dataframes["dataset"][["label"]].copy().to_numpy(dtype=np.float32)
        self.x = self.dataframes["dataset"][
            self.col_CIVET +
            self.col_DEMO +
            self.col_APOE
            ].copy().to_numpy(dtype=np.float32)

        self.x = torch.FloatTensor(self.x).cuda()
        self.y = torch.FloatTensor(self.y).cuda()

        # clear redundant dataset
        del self.dataframes
        
    
    def verbose_print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    
    def set_unique_keys(self):
        # DataFrame merging is based on keys.
        # But, given xlsx files (from ICT) dont have any matching keys.
        
        self.dataframes["dataset_kfold"] = self.dataframes["dataset_kfold"].to_frame(self.unique_key)
        self.dataframes["dataset_kfold"][self.unique_key] = self.dataframes["dataset_kfold"][self.unique_key].apply(self.get_unique_key_fn)

        self.dataframes["dataset"][self.unique_key] = self.dataframes["dataset"]["Cth"].apply(self.get_unique_key_fn)
        self.dataframes["dataset_demo_apoe"][self.unique_key] = self.dataframes["dataset_demo_apoe"]["Cth"].apply(self.get_unique_key_fn)


    def collect_n_fold(self):
        
        # concat all cols in kfold.xlsx w.r.t self.metadata["fold_idx"]

        self.dataframes["dataset_kfold"] = pd.concat([self.dataframes["dataset_kfold"][str(idx)] for idx in self.metadata["fold_idxs"]])


    def preprocess(self):
        
        self.verbose_print("preprocessing")
        
        # remove row if it contains error
        len_prev = len(self.dataframes["dataset"].index)
        self.dataframes["dataset"].dropna(axis=0, how="any", inplace=True)  # drop any row(axis=0) where nan exists
        self.metadata["error_count"] = len_prev - len(self.dataframes["dataset"].index)  # how many nans (=errors)
    
        
        # merge dataset (without demo, apoe) with dataset_with_demo_apoe.
        
        #############################################################################################################################################
        #############################################################################################################################################
        # IMPORTANT! duplicating keys while merging changes the key name automatically. Dup_Key --> Dup_Key_L, Dup_Key_R based on pd.merge left/right
        # thus must manually select only the desired keys.
        #############################################################################################################################################
        #############################################################################################################################################

        # Get CIVET columns from self.dataframes["dataset"]
        if self.frame_type == "VFI":
            self.dataframes["dataset"] = self.dataframes["dataset"][[self.unique_key]+self.col_CIVET]  # manually select only the desired keys.
            self.dataframes["dataset_demo_apoe"] = self.dataframes["dataset_demo_apoe"] \
                [[self.unique_key, "label"] + self.col_APOE + self.col_DEMO]  # manually select only the desired keys.
            self.dataframes["dataset"] = pd.merge(
                left=self.dataframes["dataset"],
                right=self.dataframes["dataset_demo_apoe"],
                on=self.unique_key,
                how="inner"  # Drop if not matching keys (= where errors occurred).
            )

        # Get CIVET columns from self.dataframes["dataset_demo_apoe"] which contains CIVET info from GT (not VFI) MRI frames
        elif self.frame_type == "GT":
            self.dataframes["dataset"] = self.dataframes["dataset"][self.unique_key]  # manually select only the desired keys.
            self.dataframes["dataset_demo_apoe"] = self.dataframes["dataset_demo_apoe"] \
                [[self.unique_key, "label"] + self.col_APOE + self.col_DEMO + self.col_CIVET]  # manually select only the desired keys.
            
            self.dataframes["dataset"] = pd.merge(
                left=self.dataframes["dataset"],
                right=self.dataframes["dataset_demo_apoe"],
                on=self.unique_key,
                how="inner"  # Drop if not matching keys (= where errors occurred).
            )
            

        # convert sex to 01. M:1, F:0
        col_name = "sex"
        self.dataframes["dataset"].loc[self.dataframes["dataset"][col_name] == "M", col_name] = 1
        self.dataframes["dataset"].loc[self.dataframes["dataset"][col_name] == "F", col_name] = 0
        
        # convert label to 01. ADD:1, NC:0
        col_name = "label"
        self.dataframes["dataset"].loc[self.dataframes["dataset"][col_name] == "ADD", col_name] = 1
        self.dataframes["dataset"].loc[self.dataframes["dataset"][col_name] == "NC", col_name] = 0
        
        
        # normalize
        cols_to_normalize = self.col_to_NORMALIZE
        for col in cols_to_normalize:
            m = self.dataframes["dataset"][col].mean()
            s = self.dataframes["dataset"][col].std()
            self.dataframes["dataset"][col] = (self.dataframes["dataset"][col] - m) / s  # normalize to Z distribution
        
        
    def match_k_fold(self):
        """
        merge inner (to drop rows of self.dataframes["dataset"] not in self.dataframes["dataset_fold"])
        """
        # only leave matching rows with how="inner"
        self.dataframes["dataset"] = pd.merge(
            left=self.dataframes["dataset_kfold"],
            right=self.dataframes["dataset"],
            on=self.unique_key,
            how="inner"
        )

    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        
        return {
            "x": self.x[idx],
            "y": self.y[idx]
        }

    def __repr__(self):
        return json.dumps(self.metadata, indent=4)


def preload_xlsx(filename, verbose=True)->DataFrame_wrapper:
    """
    type(filename) == path_like string or DataFrame_wrapper
    """
    
    if isinstance(filename, DataFrame_wrapper):
        if verbose:
            print(f"reading {filename.filename}")
        return filename
    
    elif isinstance(filename, str):
        if verbose:
            print(f"reading {filename}")
        if "@" in filename:  # if explicitly defined sheet name (=@). Default=0.
            dataframe = pd.read_excel(
                io=filename.split("@")[0],
                sheet_name=filename.split("@")[1],
            )
        else:
            dataframe = pd.read_excel(filename)
    
    else:
        raise NotImplementedError(f"Unknown type of filename. Got {type(filename)} ")
    
    return DataFrame_wrapper(dataframe=dataframe, filename=filename)


def define_dataset(xlsx_dataset, xlsx_dataset_kfold, xlsx_dataset_demo_apoe, frame_type, training_columns, fold):
    
    
    # define k-fold. ex: train [0, 1, 3, 4] / val [2]
    k_fold_idx, n_fold = fold.split("@")
    train_k_fold_idxs = set([i for i in range(int(n_fold))])
    train_k_fold_idxs.remove(int(k_fold_idx))
    train_k_fold_idxs = list((train_k_fold_idxs))
    valid_k_fold_idxs = [int(k_fold_idx)]
    
    
    train_dataset = CIVET_DATASET(
        frame_type=frame_type,
        fold_idxs=train_k_fold_idxs,
        xlsx_dataset=xlsx_dataset,
        xlsx_dataset_kfold=xlsx_dataset_kfold,
        xlsx_dataset_demo_apoe=xlsx_dataset_demo_apoe,
        verbose=False
    )
    valid_dataset = CIVET_DATASET(
        frame_type=frame_type,
        fold_idxs=valid_k_fold_idxs,
        xlsx_dataset=xlsx_dataset,
        xlsx_dataset_kfold=xlsx_dataset_kfold,
        xlsx_dataset_demo_apoe=xlsx_dataset_demo_apoe,
        verbose=False
    )
    
    # select only desired columns (CIVET, CIVET+APOE, CIVET+DEMO, CIVET+APOE+DEMO)
    select_column_fn_mapping = {
        "CIVET": lambda x: x[:, [0, 1, 2, 3, 4]],
        "CIVET+DEMO": lambda x: x[:, [0, 1, 2, 3, 4, 5, 6, 7]],
        "CIVET+APOE": lambda x: x[:, [0, 1, 2, 3, 4, 8]],
        "CIVET+DEMO+APOE": lambda x: x,
        "CIVET+APOE+DEMO": lambda x: x,
    }
    
    select_column_fn = select_column_fn_mapping[training_columns]
    train_dataset.x = select_column_fn(train_dataset.x)
    valid_dataset.x = select_column_fn(valid_dataset.x)
    
    
    
    return train_dataset, valid_dataset


if __name__ == "__main__":
    import time
    
    start_time = time.time()
    
    dataset = preload_xlsx("./dataset/20221011/linear_221011.xlsx" + "@" + "linear_x2")
    dataset_kfold = preload_xlsx("./dataset/20221011/kfold.xlsx")
    dataset_demo_apoe = preload_xlsx("./dataset/20221011/220420_th_with_demo.xlsx")
    
    civet_dataset = CIVET_DATASET(
        fold_idxs=[1, 2, 3],
        xlsx_dataset=dataset,
        xlsx_dataset_kfold=dataset_kfold,
        xlsx_dataset_demo_apoe=dataset_demo_apoe,
        verbose=True
    )