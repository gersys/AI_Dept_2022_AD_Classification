from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
from ICT_AD_classification.model import MLP, SVM_wrapper
from ICT_AD_classification.civet_dataset import define_dataset, preload_xlsx
import os
import torch
import time

class Trainer():
    
    def __init__(self, classifier_type, train_dataset, valid_dataset):
    
        # misc
        assert classifier_type in ["MLP", "SVM"]
        self.classifier_type = classifier_type
        self.verbose = False
    
        # define dataloader
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        self.valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True, num_workers=0)
       
        # define model, train/valid functions
        input_dim = train_dataset.x.shape[-1]  # dynamic initialize classifier by dataset input dim
        self.model = MLP(input_dim).cuda() if classifier_type == "MLP" else SVM_wrapper()
        self.basic_train_valid_sequence = \
            self.basic_train_valid_sequence_MLP if classifier_type == "MLP" else self.basic_train_valid_sequence_SVM

        # train hyper-params / train assets
        if self.classifier_type == "MLP":
            # train params
            self.n_epoch = 300
            self.lr = 0.001
            self.step_size = 50
            self.gamma = 0.5
            self.criterion = torch.nn.BCEWithLogitsLoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.9)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

        # training status
        self.time_start = time.time()
        self.time_duration = 0
        self.best_model_acc = 0  # best model in terms of AU-ROC scores.
        self.best_model_auroc = 0
        self.curr_train_acc = 0
        self.curr_val_acc = 0
        self.curr_epoch = 0
        
        # train results
        self.save_PATH = './save/'
        os.makedirs(self.save_PATH, exist_ok=True)

    def verbose_print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
    
    def train_SVM(self):
        # cpu based :(
        x = self.train_loader.dataset.x.cpu().numpy()
        y = self.train_loader.dataset.y.cpu().numpy()
        
        self.model.fit(x, y.flatten())
        pred = self.model.predict_proba(x)[:, 1]
        
        self.train_acc = (y.flatten() == np.round(pred)).sum() / len(pred)
        
    def valid_SVM(self):
    
        # cpu-based svm.
        x = self.valid_loader.dataset.x.cpu().numpy()
        y = self.valid_loader.dataset.y.cpu().numpy()

        pred = self.model.predict_proba(x)[:, 1]
        self.best_model_acc = (y.flatten() == np.round(pred)).sum() / len(pred)
        self.best_model_auroc = roc_auc_score(y_true=y.flatten(), y_score=pred)
        


    def train_MLP(self):
        
        grad_scaler = torch.cuda.amp.GradScaler()
        labels_total = np.array([])
        preds_total = np.array([])
        
        self.model.train()
        for i, batch_data in enumerate(self.train_loader):
            # load data
            inputs = batch_data["x"]
            labels = batch_data["y"]
            
            # main train section
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            
            # outputs
            preds = torch.sigmoid(outputs)
            preds_total = np.append(preds_total, preds.flatten().detach().cpu().numpy())
            labels_total = np.append(labels_total, labels.flatten().detach().cpu().numpy())
            

        # log
        self.curr_train_acc = (labels_total == np.round(preds_total)).sum() / len(labels_total)
        _lr = self.optimizer.param_groups[0]["lr"]
        # print(f"e: {self.curr_epoch:5d} -- train acc: {self.curr_train_acc:.2f} -- lr: {_lr}"
        #       f" -- {len(preds_total), len(labels_total)}")
    
        # ...
        self.scheduler.step()
        
    
    
    def valid_MLP(self):

        labels_total = np.array([])
        preds_total = np.array([])
    
        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(self.valid_loader):
                # load data
                inputs = batch_data["x"]
                labels = batch_data["y"]
                
                outputs = self.model(inputs)

                # outputs
                preds = torch.sigmoid(outputs)
                preds_total = np.append(preds_total, preds.flatten().detach().cpu().numpy())
                labels_total = np.append(labels_total, labels.flatten().detach().cpu().numpy())

        # log
        self.curr_val_acc = (labels_total == np.round(preds_total)).sum() / len(labels_total)
        self.curr_val_auroc = roc_auc_score(y_true=labels_total, y_score=preds_total)
        
        # save model if best validation result
        if self.curr_val_acc >= self.best_model_acc:
            # torch.save(self.model.state_dict(), self.save_PATH + f"epoch_{self.curr_epoch}_acc_{self.curr_val_acc:.2f}.pth")
            # print("model saved")
            self.best_model_acc = self.curr_val_acc
            self.best_model_auroc = self.curr_val_auroc
            
        self.verbose_print(f"e: {self.curr_epoch:5d} -- val acc: {self.curr_val_acc:.2f} -- best acc: {self.best_model_acc:.2f}"
              f" -- val auroc {self.curr_val_auroc:.2f} -- best auroc {self.best_model_auroc:.2f}")


    def basic_train_valid_sequence_SVM(self):
        self.train_SVM()
        self.valid_SVM()
        _filename = self.train_loader.dataset.metadata["dataset"]

        self.time_duration = time.time() - self.time_start
        self.verbose_print(f"{_filename} ({len(self.train_loader.dataset):4d} items)")
        self.verbose_print(f"{self.classifier_type}: --- best_ACC: {self.best_model_acc}")


    def basic_train_valid_sequence_MLP(self):
    
        for epoch in range(self.n_epoch):
            self.train_MLP()
            self.valid_MLP()
            self.curr_epoch += 1
            
        _filename = self.train_loader.dataset.metadata["dataset"]

        self.time_duration = time.time() - self.time_start
        self.verbose_print(f"{_filename} ({len(self.train_loader.dataset):4d} items)")
        self.verbose_print(f"{self.classifier_type}: {_filename} ({len(self.train_loader.dataset):4d} items) --- best_ACC: {self.best_model_acc}")


class Logger():
    def __init__(self, log_file_name):
        self.log_file_name = log_file_name
        with open((self.log_file_name), mode="w") as log:
            log.write(",".join([
                "filename",
                "sheetname",
                "train_columns",
                "frame_type",
                "classifier_type",
                "fold",
                "train_dataset_len",
                "valid_dataset_len",
                "best_model_acc",
                "best_model_auroc",
                "\n"
            ]))

    def log(self, log_str):
        with open((self.log_file_name), mode="a") as log:
            log.write(log_str+"\n")


if __name__=="__main__":
    
    n_fold = 5
    preloaded = {
        "xlsx_dataset_kfold": preload_xlsx("./dataset/20221011/kfold.xlsx@Sheet1"),
        "xlsx_dataset_demo_apoe": preload_xlsx("./dataset/20221011/220420_th_with_demo.xlsx@AD_NC_Cth_segment"),
    }
    
    
    # path_like_str or Dataframe_wrapper
    xlsx_dataset_list = [
        preloaded["xlsx_dataset_demo_apoe"],
        # *[f"./dataset/20221011/linear_221011.xlsx@linear_x{i}" for i in [2, 4, 8, 16, 32]],
        # *[f"./dataset/20221011/laplacian_no_perceptual_221005.xlsx@laplacian_no_perceptual_x{i}" for i in [2, 4, 8, 16, 32]],
        *[f"./dataset/20221011/3_laplacian_perceptual.xlsx@x{i}" for i in [2, 4, 8, 16, 32]],
    ]

    logger = Logger("log_20221028.csv")
    for file_sheet_name in xlsx_dataset_list:
        for classifier_type in ["SVM", "MLP", ]:
            for training_columns in ["CIVET", "CIVET+DEMO", "CIVET+APOE", "CIVET+APOE+DEMO"]:
                for frame_type in ["VFI", "GT"]:
                    for fold in [f"{i}@{n_fold}" for i in range(n_fold)]:  # i@n_fold means: use i_th col (starting from 0) as val set, out of n_fold columns.
        
                        xlsx_dataset = preload_xlsx(file_sheet_name, verbose=False)
                        xlsx_dataset_kfold = preloaded["xlsx_dataset_kfold"]
                        xlsx_dataset_demo_apoe = preloaded["xlsx_dataset_demo_apoe"]
                        
                        train_dataset, valid_dataset = define_dataset(
                            xlsx_dataset=xlsx_dataset,
                            xlsx_dataset_kfold=xlsx_dataset_kfold,
                            xlsx_dataset_demo_apoe=xlsx_dataset_demo_apoe,
                            frame_type=frame_type,
                            training_columns=training_columns,
                            fold=fold,
                        )
                        trainer = Trainer(classifier_type, train_dataset, valid_dataset)
                        trainer.basic_train_valid_sequence()
               
                        fold_idx = fold.split("@")[0]
                        train_configs = ",".join([
                            os.path.basename(xlsx_dataset.filename.split("@")[0]),  # file name (VFI method)  3
                            xlsx_dataset.filename.split("@")[1],  # sheet name (VFI scale)  5
                            training_columns,  # CIVET or CIVET+alpha  4
                            frame_type,  # comparison (GT frames, VFI frames)  2
                            classifier_type,  # MLP or SVM  2
                            f"{fold_idx}",  # fold  5
                            f"{len(trainer.train_loader.dataset.x)}",  # train_data_len
                            f"{len(trainer.valid_loader.dataset.x)}",  # valid_data_len
                            f"{trainer.best_model_acc:.3f}",  # best acc
                            f"{trainer.best_model_auroc:.3f}",  # best auroc
                        ])
                        logger.log(train_configs)
                        
                        print(
                            f"| filename: {xlsx_dataset.filename:42s} "
                            f"| {classifier_type} "
                            f"| auroc: {trainer.best_model_auroc:.3f} "
                            f"| acc: {trainer.best_model_acc:.3f} "
                            f"| frame_type: {frame_type:3s} "
                            f"| fold: {fold:4s} "
                            f"| train_cols: {training_columns:16s} "
                            f"| train_len: {len(trainer.train_loader.dataset.x):5d}"  # train_data_len
                            f"| valid_len: {len(trainer.valid_loader.dataset.x):5d}"  # valid_data_len
                        )

