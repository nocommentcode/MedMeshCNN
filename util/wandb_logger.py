from datetime import datetime

import wandb
import os

os.environ["WANDB_SILENT"] = "true"


class WandbLogger:
    def __init__(self, opt):
        self.opt = opt
        self.is_active = opt.log_wandb
        self.run_name = opt.run_name
        self.is_initialised = False

    def initialise_logging(self):
        if self.is_active:
            name = self.generate_run_name()

            wandb.init(project="mms_cfd", entity="monash-deep-neuron", name=name, config=vars(self.opt))
            print(f"initialised logging to wandb under run name {name}")
            self.is_initialised = True

    def log_train_loss(self, loss):
        if not self.can_log():
            return

        wandb.log({'train_cda_loss': loss.cda, 'train_cla_loss': loss.cla, 'train_cop_loss': loss.cop,
                   'train_total_loss': loss.total_loss()})

    def log_test_loss(self, loss):
        if not self.can_log():
            return

        wandb.log({'test_cda_loss': loss.cda, 'test_cla_loss': loss.cla, 'test_cop_loss': loss.cop,
                   'test_total_loss': loss.total_loss()})

    def log_train_av_loss(self, loss, epoch):
        if not self.can_log():
            return

        wandb.log({'train_cda_loss': loss.cda, 'train_cla_loss': loss.cla, 'train_cop_loss': loss.cop,
                   'train_total_loss': loss.total_loss(), 'epoch': epoch})

    def log_test_av_loss(self, loss, epoch):
        if not self.can_log():
            return

        wandb.log({'test_cda_loss': loss.cda, 'test_cla_loss': loss.cla, 'test_cop_loss': loss.cop,
                   'test_total_loss': loss.total_loss(), 'epoch': epoch})

    def log_test_accuracy(self, acc, epoch):
        if not self.can_log():
            return

        wandb.log({'test_cda_acc': acc.cda_correct / acc.cda_count, 'test_cla_acc': acc.cla_correct / acc.cla_count,
                   'test_cop_acc': acc.cop_correct / acc.cop_count,
                   'test_total_acc': acc.total_accuracy(), 'epoch': epoch})

    def log_train_accuracy(self, acc, epoch):
        if not self.can_log():
            return

        wandb.log({'train_cda_acc': acc.cda_correct / acc.cda_count, 'train_cla_acc': acc.cla_correct / acc.cla_count,
                   'train_cop_acc': acc.cop_correct / acc.cop_count,
                   'train_total_acc': acc.total_accuracy(), 'epoch': epoch})

    def can_log(self):
        if not self.is_active:
            return False
        if not self.is_initialised:
            print("Wandb logging not initialised, skipping")
            return False
        return True

    def generate_run_name(self):
        if self.opt.run_name is None:
            date = datetime.now()
            data_folder = self.opt.dataroot.split("/")[-1]
            data_folder = data_folder.split("\\")[-1]
            return f"cfd | {date.month}/{date.day} {date.hour}:{date.minute} | {data_folder}"
        else:
            return self.opt.run_name

    def init_log_prediction(self):
        if not self.can_log():
            return
        self.pred_table = wandb.Table(columns=["Epoch", "Batch #", "Metric", "Predicted", "actual"])
        self.pred_initialised = True

    def log_predictions(self, epoch, iter, metric, predictions, labels):
        if not self.can_log():
            return

        if self.pred_initialised:
            self.pred_table.add_data(str(epoch), str(iter), str(metric), str(predictions), str(labels))
        else:
            print("tried to log preditions without initialising")

    def commit_predictions(self):
        if not self.can_log():
            return

        if self.pred_initialised:
            wandb.log({"prediction": self.pred_table})

            self.pred_table = None
            self.pred_initialised = False

    def log_model(self, epoch):
        if not self.can_log():
            return
        path = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        path = os.path.join(path, f'{epoch}_net.pth')
        wandb.save(path)
